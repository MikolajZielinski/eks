import tyro
import numpy as np
import open3d as o3d

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Annotated, Union
from tqdm import tqdm

from nerfstudio.utils.eval_utils import eval_setup

from eks.eks_model import EksModel


@dataclass
class SineEdit:

    load_config: Path
    """Path to config YAML file."""
    eval_num_rays_per_chunk: Optional[int] = None
    """Specifies number of rays per chunk during eval. If None, use the value in the config file."""

    def edit(self):

        _, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="test",
        )

        assert isinstance(pipeline.model, EksModel), "DatasetRender only works with EksModel"
        means = pipeline.model.field.mlp_base.encoder.means.detach().cpu().numpy()
        num_images = len(pipeline.datamanager.eval_dataset)

        output_path = self.load_config.parent / Path("camera_path")
        output_path.mkdir(parents=True, exist_ok=True)

        for idx in tqdm(range(num_images), desc="Generating pointclouds"):
            # Apply sinusoidal modification to means in 100 bins along the y-axis
            new_means = means.copy()
            x_values = new_means[:, 1]
            x_min, x_max = x_values.min(), x_values.max()
            bins = np.linspace(x_min, x_max, 1001)
            bin_indices = np.digitize(x_values, bins) - 1

            # Shift the sine phase based on idx
            phase_shift = (idx / num_images) * 10 * np.pi

            # Apply sinusoidal modification based on bin index with phase shift
            for i in range(1000):
                mask = bin_indices == i
                if mask.any():
                    new_means[mask, 0] = new_means[mask, 0] + np.sin((i / 1000 * 2 * np.pi) + phase_shift) * 0.1
            
            # Create an Open3D point cloud from means
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(new_means)

            # Save the point cloud as a PLY file
            pcd_path = output_path / Path(f"{idx:05d}.ply")
            o3d.io.write_point_cloud(str(pcd_path), pcd)

        print(f"Saved point clouds to {output_path}")


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[SineEdit, tyro.conf.subcommand(name="sine")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).edit()


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(Commands)  # noqa