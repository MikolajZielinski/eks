import torch
import numpy as np

from dataclasses import dataclass, field
from typing import Type, Optional, Dict
from pathlib import Path

from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig, Blender
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig, Nerfstudio

from genie.utils.utils import rotmat_to_quat, quat_multiply
from genie.utils.ply_utils import read_ply


def extract_gaussians_from_ply(ply_gaussians: dict, transform_matrix: torch.Tensor, scale_factor: float, opacity_threshold: Optional[float] = None) -> Dict[str, torch.Tensor]:
    """Extracts 3D Gaussians from a PLY file data.

    Args:
        ply_gaussians: A dictionary containing the data read from a PLY file.
        transform_matrix: A 4x4 transformation matrix to apply to the Gaussian means.
        scale_factor: A scaling factor to apply to the Gaussian means and scales.
    
    Returns:
        A Gaussians object containing the extracted means, scales, quats, and colors.
    """

    assert transform_matrix.shape == (3, 4), "Transform matrix must be of shape (3, 4)"

    # Check fields using dtype.names
    field_names = ply_gaussians.dtype.names

    # Read means
    points3D = np.stack([ply_gaussians["x"], ply_gaussians["y"], ply_gaussians["z"]], axis=-1)
    points3D = (np.concatenate((points3D, np.ones_like(points3D[..., :1]),), -1,) @ transform_matrix.T.cpu().detach().numpy())
    points3D *= scale_factor
    points3D = torch.from_numpy(points3D.astype(np.float32))

    # Read opacity and create opacity mask
    mask = torch.ones(points3D.shape[0], dtype=torch.bool)
    if "opacity" in field_names and opacity_threshold is not None:
        print(f"Opacity field found, filtering points with opacity <= {opacity_threshold}")
        opacity = ply_gaussians["opacity"]
        mask = opacity > opacity_threshold
        points3D = points3D[mask]
        print(f"Removed {np.sum(~mask)} points, {points3D.shape[0]} points remaining")

    # Read colors
    if "f_dc_0" in field_names:
        sh_0 = ply_gaussians["f_dc_0"]
        sh_1 = ply_gaussians["f_dc_1"]
        sh_2 = ply_gaussians["f_dc_2"]
        # SH to RGB (DC component only)
        rgb = 0.5 + 0.28209479177387814 * np.stack([sh_0, sh_1, sh_2], axis=-1)
        points3D_rgb = torch.from_numpy(np.clip(rgb * 255, 0, 255).astype(np.uint8))
    elif "red" in field_names:
        points3D_rgb = np.stack([ply_gaussians["red"], ply_gaussians["green"], ply_gaussians["blue"]], axis=-1)
        points3D_rgb = torch.from_numpy(points3D_rgb.astype(np.uint8))
        points3D_rgb = points3D_rgb[mask]
    else:
        points3D_rgb = torch.zeros_like(points3D, dtype=torch.uint8)
        points3D_rgb = points3D_rgb[mask]

    # Read quaternions
    points3D_quats = None
    if "rot_0" in field_names:
        points3D_quats = torch.from_numpy(
            np.stack([ply_gaussians["rot_0"], ply_gaussians["rot_1"], ply_gaussians["rot_2"], ply_gaussians["rot_3"]], axis=-1).astype(np.float32)
        )
        # Apply transform_matrix rotation to quats
        R_tf = torch.as_tensor(transform_matrix[:3, :3], dtype=points3D_quats.dtype, device=points3D_quats.device)
        q_tf = rotmat_to_quat(R_tf.expand(points3D_quats.shape[0], -1, -1))
        points3D_quats = quat_multiply(q_tf, points3D_quats)
        points3D_quats = points3D_quats / points3D_quats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        points3D_quats = points3D_quats[mask]

    # Read scales
    points3D_scale = None
    if "scale_0" in field_names:
        points3D_scale = torch.exp(torch.from_numpy(np.stack([ply_gaussians["scale_0"], ply_gaussians["scale_1"], ply_gaussians["scale_2"]], axis=-1).astype(np.float32)))
        points3D_scale = points3D_scale[mask]
        points3D_scale = points3D_scale * scale_factor

    out = {
        "points3D_xyz": points3D,
        "points3D_rgb": points3D_rgb,
        "points3D_quat": points3D_quats,
        "points3D_scale": points3D_scale,
    }

    return out


@dataclass
class EksBlenderDataParserConfig(BlenderDataParserConfig):
    """Configuration for Eks Blender data parser."""

    _target: Type = field(default_factory=lambda: EksBlender)

class EksBlender(Blender):
    """Eks Blender data parser.

    This class extends the BlenderDataParser to handle Eks-specific data parsing.
    """

    def __init__(self, config: EksBlenderDataParserConfig):

        config.ply_path ="sparse_pc.ply"
        super().__init__(config)

    def _load_3D_points(self, ply_file_path: Path):

        gaussians = read_ply(ply_file_path)

        # transform_matrix = torch.tensor([
        #     [1.0, 0.0, 0.0, 1.5],
        #     [0.0, 1.0, 0.0, 1.5],
        #     [0.0, 0.0, 1.0, 1.5],
        # ])
        # scale_factor = 1.0 / 3.0

        transform_matrix = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])
        scale_factor = 1.0

        out = extract_gaussians_from_ply(gaussians, transform_matrix, scale_factor)

        print(out["points3D_xyz"].shape)
            
        return out
    

@dataclass
class EksNerfstudioDataParserConfig(NerfstudioDataParserConfig):
    """Configuration for Eks Nerfstudio data parser."""

    _target: Type = field(default_factory=lambda: EksNerfstudio)
    """target class to instantiate"""
    downscale_factor: Optional[int] = None
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    load_3D_points: bool = True
    """Whether to load the 3D points from the colmap reconstruction."""


class EksNerfstudio(Nerfstudio):
    """Eks Nerfstudio data parser.

    This class extends the NerfstudioDataParser to handle Eks-specific data parsing.
    """

    def __init__(self, config: EksNerfstudioDataParserConfig):

        config.ply_path ="sparse_pc.ply"
        super().__init__(config)

    def _load_3D_points(self, ply_file_path: Path, transform_matrix: torch.Tensor, scale_factor: float):

        ply_gaussians = read_ply(ply_file_path)

        print(transform_matrix)

        ox = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        transform_matrix = transform_matrix @ ox

        out = extract_gaussians_from_ply(ply_gaussians, transform_matrix, scale_factor)
            
        return out