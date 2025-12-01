"""
Implementation of GENIE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union, Callable, Any

import nerfacc
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch.cuda.amp import GradScaler

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
# from nerfstudio.model_components.losses import MSELoss, scale_gradients_by_distance_squared
from nerfstudio.model_components.ray_samplers import VolumetricSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, RGBRenderer
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps

from genie.field.field import GENIEField
from genie.knnx.knn_algorithms import BaseKNNConfig, BaseKNN
from genie.utils.viewer_utils import ViewerPointCloud, ViewerOccupancyGrid, ViewerAABB
from genie.utils.losses import distortion

@dataclass
class GENIEModelConfig(ModelConfig):
    """GENIE Model Config"""

    _target: Type = field(
        default_factory=lambda: GENIEModel
    )  # We can't write `NGPModel` directly, because `NGPModel` doesn't exist yet
    """target class to instantiate"""
    enable_collider: bool = False
    """Whether to create a scene collider to filter rays."""
    collider_params: Optional[Dict[str, float]] = None
    """Instant NGP doesn't use a collider."""
    grid_resolution: Union[int, List[int]] = 128
    """Resolution of the grid used for the field."""
    alpha_thre: float = 0.0
    """Threshold for opacity skipping."""
    cone_angle: float = 0.0
    """Should be set to 0.0 for blender scenes but 1./256 for real scenes."""
    render_step_size: Optional[float] = 0.005
    """Minimum step size for rendering."""
    near_plane: float = 0.0
    """How far along ray to start sampling."""
    far_plane: float = 1e10
    """How far along ray to stop sampling."""
    use_gradient_scaling: bool = True
    """Use gradient scaler where the gradients are lower for points closer to the camera."""
    use_appearance_embedding: bool = False
    """Whether to use an appearance embedding."""
    appearance_embedding_dim: int = 32
    """Dimension of the appearance embedding."""
    background_color: Literal["random", "black", "white"] = "white"
    """
    The color that is given to masked areas.
    These areas are used to force the density in those regions to be zero.
    """
    disable_scene_contraction: bool = True
    """Whether to disable scene contraction or not."""
    knn_algorithm: BaseKNNConfig = field(default_factory=lambda: BaseKNN())
    """KNN algorithm to use for nearest neighbor search."""
    max_gb: int = 20
    """Maximum amount of GPU memory to use for densification."""
    densify: bool = True
    """Whether to densify points or not. If False, the model will not densify."""
    prune: bool = True
    """Whether to prune the model or not. If False, the model will not prune."""
    unfreeze_means: bool = True
    """Whether to unfreeze the means of the encoder or not."""


class GENIEModel(Model):
    """Instant NGP model

    Args:
        config: instant NGP configuration to instantiate model
    """

    config: GENIEModelConfig
    field: GENIEField

    def __init__(self, config: GENIEModelConfig, **kwargs) -> None:
        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        # Get seed points
        seed_points = self.kwargs.get("seed_points", None)
        if seed_points is not None:
            seed_points = seed_points[0]

        # Initilize field
        self.knn_algorithm = self.config.knn_algorithm.setup()
        self.field = GENIEField(
            knn_algorithm=self.knn_algorithm,
            aabb=self.scene_box.aabb,
            appearance_embedding_dim=self.config.appearance_embedding_dim if self.config.use_appearance_embedding else 0,
            num_images=self.num_train_data,
            spatial_distortion=scene_contraction,
            seed_points=seed_points,
            densify=self.config.densify,
            prune=self.config.prune,
            unfreeze_means=self.config.unfreeze_means,
        )

        self.scene_aabb = Parameter(self.scene_box.aabb.flatten(), requires_grad=False)

        # Auto step size: ~1000 samples in the base level grid
        if self.config.render_step_size is None:
            self.config.render_step_size = ((self.scene_aabb[3:] - self.scene_aabb[:3]) ** 2).sum().sqrt().item() / 1000

        # Occupancy Grid.
        roi_aabb = self.scene_aabb if self.config.disable_scene_contraction else self.scene_aabb * 2
        self.occupancy_grid = nerfacc.OccGridEstimator(
            roi_aabb=roi_aabb, # self.scene_aabb,
            resolution=self.config.grid_resolution,
            levels=1,
        )

        # Sampler
        self.sampler = VolumetricSampler(
            occupancy_grid=self.occupancy_grid,
            density_fn=self.field.density_fn,
        )

        # Renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")

        # Losses
        self.rgb_loss = F.smooth_l1_loss

        # Metrics
        from torchmetrics.functional import structural_similarity_index_measure
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        # Point Cloud Viewer
        self.viewer_point_cloud_handle = ViewerPointCloud(
            name="means", 
            aabb=self.scene_box, 
            points=self.field.mlp_base.encoder.gauss_params["means"].detach().cpu().numpy(),
            confidence=self.field.mlp_base.encoder.confidence.detach().cpu().numpy(),
            gradients=self.field.mlp_base.encoder.xyz_gradient_accum.detach().cpu().numpy(),
            show_gradients=True,
        )
        self.viewer_occupancy_grid_handle = ViewerOccupancyGrid(
            name="occupancy_grid",
            aabb=self.scene_box,
            occ_grid=self.occupancy_grid.binaries.bool().squeeze(0).detach().cpu(),
        )
        self.viewer_aabb_handle = ViewerAABB(
            name="aabb",
            aabb=self.scene_box,
        )

        # GradScaler
        self.grad_scaler = GradScaler(2**10)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        
        def update_occupancy_grid(step: int):
            self.occupancy_grid.update_every_n_steps(
            step=step,
            occ_eval_fn=lambda x: self.field.density_fn(x) * self.config.render_step_size,
            )

        def update_viewer(step: int):
            self.viewer_occupancy_grid_handle.update(
                occ_grid=self.occupancy_grid.binaries.bool().squeeze(0).detach().cpu()
            )
            self.viewer_point_cloud_handle.update(
                points=self.field.mlp_base.encoder.gauss_params["means"].detach().cpu().numpy(),
                confidence=self.field.mlp_base.encoder.confidence.detach().cpu().numpy(),
                gradients=self.field.mlp_base.encoder.xyz_gradient_accum.detach().cpu().numpy(),
            )

        return [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=update_occupancy_grid,
            ),
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=update_viewer,
            ),
        ]

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_param_groups")

        fields = []
        for name, param in self.field.named_parameters():
            if name == "mlp_base.encoder.gauss_params.means":
                param_groups["means"] = [param]
            elif name == "mlp_base.encoder.gauss_params.log_covs":
                param_groups["log_covs"] = [param]
            elif name == "mlp_base.encoder.gauss_params.quats":
                param_groups["quats"] = [param]
            else:
                fields.append(param)

        param_groups["fields"] = fields

        return param_groups
    
    def densify_points(self, optimizers: Dict[str, torch.optim.Optimizer]) -> bool:
        # Check memory usage before densifying
        used_gb = torch.cuda.memory_reserved() / 1e9
        if used_gb > self.config.max_gb:
            print(f"[Densification] Skipped: CUDA memory usage {used_gb:.2f}GB > {self.config.max_gb}GB")
            return False
        
        if self.config.densify:
            # Calculate scene extent for splitting threshold
            extent = self.scene_aabb[3:] - self.scene_aabb[:3]
            scene_extent = extent.max().item()
            
            self.field.mlp_base.encoder.densify_and_split(optimizers, scene_extent=scene_extent)
            return True
            
        return False

    def get_outputs(self, ray_bundle: RayBundle):
        assert self.field is not None
        num_rays = len(ray_bundle)

        with torch.no_grad():
            ray_samples, ray_indices = self.sampler(
                ray_bundle=ray_bundle,
                near_plane=self.config.near_plane,
                far_plane=self.config.far_plane,
                render_step_size=self.config.render_step_size,
                alpha_thre=self.config.alpha_thre,
                cone_angle=self.config.cone_angle,
            )

        field_outputs = self.field(ray_samples)

        # accumulation
        packed_info = nerfacc.pack_info(ray_indices, num_rays)
        trans, alphas = nerfacc.render_transmittance_from_density(
            t_starts=ray_samples.frustums.starts[..., 0],
            t_ends=ray_samples.frustums.ends[..., 0],
            sigmas=field_outputs[FieldHeadNames.DENSITY][..., 0],
            packed_info=packed_info,
        )
        weights = trans * alphas
        weights = weights[..., None]

        # Removed densify_buffer collection logic here

        rgb = self.renderer_rgb(
            rgb=field_outputs[FieldHeadNames.RGB],
            weights=weights,
            ray_indices=ray_indices,
            num_rays=num_rays,
        )
        depth = self.renderer_depth(
            weights=weights, ray_samples=ray_samples, ray_indices=ray_indices, num_rays=num_rays
        )
        accumulation = self.renderer_accumulation(weights=weights, ray_indices=ray_indices, num_rays=num_rays)

        mip_loss = distortion(
            weights=weights.view(-1),
            t_starts=ray_samples.frustums.starts[..., 0],
            t_ends=ray_samples.frustums.ends[..., 0],
            ray_indices=ray_indices,
            n_rays=len(ray_bundle),
        )

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "num_samples_per_ray": packed_info[:, 1],
            "mip_loss": mip_loss,
        }
        return outputs

    def get_metrics_dict(self, outputs, batch):
        image = batch["image"].to(self.device)
        image = self.renderer_rgb.blend_background(image)
        metrics_dict = {}
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        metrics_dict["num_samples_per_batch"] = outputs["num_samples_per_ray"].sum()
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        image = batch["image"].to(self.device)
        pred_rgb, image = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )
        rgb_loss = self.rgb_loss(image, pred_rgb)
        mip_loss = outputs["mip_loss"].mean() * 1e-3

        loss = rgb_loss + mip_loss
        if self.config.use_gradient_scaling:
            loss = self.grad_scaler.scale(loss)

        loss_dict = {"loss": loss}
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)
        image = self.renderer_rgb.blend_background(image)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        image = image[:rgb.shape[0], :rgb.shape[1], ...]  # Ensure image and rgb have the same batch size

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim), "lpips": float(lpips)}  # type: ignore
        # TODO(ethan): return an image dictionary

        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
        }

        return metrics_dict, images_dict
