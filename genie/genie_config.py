"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.data.pixel_samplers import PixelSamplerConfig

from genie.data.dataparsers import GenieBlenderDataParserConfig, GenieNerfstudioDataParserConfig
from genie.genie_trainer import GenieTrainerConfig
from genie.genie_model import GenieModelConfig
from genie.knnx.knn_algorithms import OptixKNNConfig
from genie.utils.schedulers import ChainedSchedulerConfig
from genie.genie_pipeline import GeniePipelineConfig

MAX_NUM_ITERATIONS = 30000

genie = MethodSpecification(
    config=GenieTrainerConfig(
        method_name="genie",
        steps_per_eval_batch=500,
        steps_per_save=100,
        max_num_iterations=MAX_NUM_ITERATIONS,
        pipeline=GeniePipelineConfig(
            target_num_samples = 1 << 18,
            datamanager=VanillaDataManagerConfig(
                dataparser=GenieBlenderDataParserConfig(
                    alpha_color="black",
                ),
                pixel_sampler=PixelSamplerConfig(
                    rejection_sample_mask=False,
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=GenieModelConfig(
                knn_algorithm=OptixKNNConfig(
                    chi_squared_radius=2.0,
                    n_neighbours=16,
                ),
                eval_num_rays_per_chunk=8192,
                grid_levels=1,
                grid_resolution=256,
                alpha_thre=0.0,
                cone_angle=0.0,
                disable_scene_contraction=True,
                densify=True,
                prune=True,
                unfreeze_means=True,
                near_plane=2.0,
                far_plane=6.0,
                background_color="black",
                ),
        ),
        optimizers={
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-06),
                "scheduler": ChainedSchedulerConfig(max_steps=MAX_NUM_ITERATIONS),
            },
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1e-5, eps=1e-15),
                "scheduler": ChainedSchedulerConfig(max_steps=MAX_NUM_ITERATIONS),
            },
            "log_covs": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ChainedSchedulerConfig(max_steps=MAX_NUM_ITERATIONS),
            },
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ChainedSchedulerConfig(max_steps=MAX_NUM_ITERATIONS),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 12),
        vis="viewer",
    ),
    description="Gaussian Splatting Encoded Neural Radiance Fields",
)

genie_real = MethodSpecification(
    config=GenieTrainerConfig(
        method_name="genie-real",
        steps_per_eval_batch=500,
        steps_per_save=100,
        max_num_iterations=MAX_NUM_ITERATIONS,
        pipeline=GeniePipelineConfig(
            target_num_samples = 1 << 17,
            datamanager=VanillaDataManagerConfig(
                dataparser=GenieNerfstudioDataParserConfig(
                ),
                pixel_sampler=PixelSamplerConfig(
                    rejection_sample_mask=False,
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=GenieModelConfig(
                knn_algorithm=OptixKNNConfig(
                    chi_squared_radius=120.0,
                    n_neighbours=16,
                ),
                eval_num_rays_per_chunk=8192,
                grid_resolution=128,
                densify=False,
                prune=False,
                unfreeze_means=False,
                near_plane=0.05,
                far_plane=1e3,
                background_color="random",
                disable_scene_contraction=False,
                cone_angle=1.0 / 256.0,
                ),
        ),
        optimizers={
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-06),
                "scheduler": ChainedSchedulerConfig(max_steps=MAX_NUM_ITERATIONS),
            },
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1e-5, eps=1e-15),
                "scheduler": ChainedSchedulerConfig(max_steps=MAX_NUM_ITERATIONS),
            },
            "log_covs": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ChainedSchedulerConfig(max_steps=MAX_NUM_ITERATIONS),
            },
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ChainedSchedulerConfig(max_steps=MAX_NUM_ITERATIONS),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 12),
        vis="viewer",
    ),
    description="Gaussian Splatting Encoded Neural Radiance Fields for Real Scenes",
)