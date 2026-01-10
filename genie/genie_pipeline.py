import torch
import typing
import torch.distributed as dist
import torchvision.utils as vutils

from pathlib import Path
from time import time
from dataclasses import dataclass, field
from typing import Literal, Dict, Any, Optional, Type
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp.grad_scaler import GradScaler
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from nerfstudio.models.base_model import Model
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipelineConfig, DynamicBatchPipeline
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.utils import profiler

@dataclass 
class GENIEPipelineConfig(DynamicBatchPipelineConfig):
    """Configuration for the GENIEPipeline."""

    _target: Type = field(default_factory=lambda: GENIEPipeline)
    """target class to instantiate"""


class GENIEPipeline(DynamicBatchPipeline):

    config: GENIEPipelineConfig
    datamanager: VanillaDataManager

    def __init__(
        self,
        config: DynamicBatchPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        # Initialize Pipeline (nn.Module)
        Pipeline.__init__(self)
        
        self.config = config
        self.test_mode = test_mode
        self.datamanager: VanillaDataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )

        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
            seed_points=self.datamanager.train_dataparser_outputs.metadata,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])

        # DynamicBatchPipeline initialization
        assert isinstance(self.datamanager, VanillaDataManager), (
            "DynamicBatchPipeline only works with VanillaDataManager."
        )

        self.dynamic_num_rays_per_batch = self.config.target_num_samples // self.config.max_num_samples_per_ray
        self._update_pixel_samplers()

    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
            step: training step of the loaded checkpoint
        """
        state = {
            (key[len("module.") :] if key.startswith("module.") else key): value for key, value in loaded_state.items()
        }

        means_size = None
        for key, value in state.items():
            if key == "_model.field.mlp_base.encoder.gauss_params.means":
                means_size = value.shape[0]
                break
        
        self.model.field.mlp_base.encoder.reinitialize_params(means_size)

        self.model.update_to_step(step)
        self.load_state_dict(state)
        means = self.model.field.mlp_base.encoder.means
        scales = torch.sqrt(torch.exp(self.model.field.mlp_base.encoder.log_covs))
        quats = self.model.field.mlp_base.encoder.quats
        self.model.field.mlp_base.encoder.knn.fit(means, scales, quats)


    @profiler.time_function
    def get_average_image_metrics(
        self,
        data_loader,
        image_prefix: str,
        step: Optional[int] = None,
        output_path: Optional[Path] = None,
        get_std: bool = False,
    ):
        """Iterate over all the images in the dataset and get the average.

        Args:
            data_loader: the data loader to iterate over
            image_prefix: prefix to use for the saved image filenames
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        metrics_dict_list = []
        num_images = len(data_loader)
        if output_path is not None:
            output_path.mkdir(exist_ok=True, parents=True)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all images...", total=num_images)
            for idx, (camera, batch) in enumerate(data_loader):
                # if idx < 98:
                #     continue
                # time this the following line
                inner_start = time()
                outputs = self.model.get_outputs_for_camera(camera=camera)
                height, width = camera.height, camera.width
                num_rays = height * width
                metrics_dict, image_dict = self.model.get_image_metrics_and_images(outputs, batch)
                if output_path is not None:
                    for key in image_dict.keys():
                        image = image_dict[key]  # [H, W, C] order
                        vutils.save_image(
                            image.permute(2, 0, 1).cpu(), output_path / f"{image_prefix}_{key}_{idx:04d}.png"
                        )

                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = (num_rays / (time() - inner_start)).item()
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = (metrics_dict["num_rays_per_sec"] / (height * width)).item()
                print(f"Image {idx}/{num_images} - PSNR: {metrics_dict['psnr']:.2f}")
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)

        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            if get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
                )

        self.train()
        return metrics_dict

    