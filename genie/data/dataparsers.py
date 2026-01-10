import math
import torch
import numpy as np

from dataclasses import dataclass, field
from typing import Type, Optional
from pathlib import Path

from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig, Blender
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig, Nerfstudio

from genie.utils.utils import rotmat_to_quat, quat_multiply, rotate_gaussians_x90


@dataclass
class GENIEBlenderDataParserConfig(BlenderDataParserConfig):
    """Configuration for GENIE Blender data parser."""

    _target: Type = field(default_factory=lambda: GENIEBlender)

class GENIEBlender(Blender):
    """GENIE Blender data parser.

    This class extends the BlenderDataParser to handle GENIE-specific data parsing.
    """

    def __init__(self, config: GENIEBlenderDataParserConfig):

        config.ply_path ="sparse_pc.ply"
        super().__init__(config)

    def _load_3D_points(self, ply_file_path: Path):

        with open(ply_file_path, "rb") as f:
            # Parse header
            properties = []
            vertex_count = 0
            
            while True:
                line = f.readline().decode("utf-8").strip()
                if line == "end_header":
                    break
                
                if line.startswith("element vertex"):
                    vertex_count = int(line.split()[-1])
                
                if line.startswith("property"):
                    parts = line.split()
                    # format: property <type> <name>
                    dtype_str = parts[1]
                    name = parts[2]
                    
                    if dtype_str == "float":
                        np_dtype = "f4"
                    elif dtype_str == "uchar":
                        np_dtype = "u1"
                    else:
                        # Handle other types if necessary
                        mapping = {
                            "double": "f8", "int": "i4", "uint": "u4", 
                            "short": "i2", "ushort": "u2", "char": "i1"
                        }
                        np_dtype = mapping.get(dtype_str, "f4")
                    
                    properties.append((name, np_dtype))

            # Read binary data
            dtype = np.dtype(properties)
            vertices = np.fromfile(f, dtype=dtype, count=vertex_count)

        points3D = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=-1)
        points3D = torch.from_numpy(points3D.astype(np.float32) * self.config.scale_factor)

        # Check fields using dtype.names
        field_names = vertices.dtype.names

        if "f_dc_0" in field_names:
            sh_0 = vertices["f_dc_0"]
            sh_1 = vertices["f_dc_1"]
            sh_2 = vertices["f_dc_2"]
            # SH to RGB (DC component only)
            rgb = 0.5 + 0.28209479177387814 * np.stack([sh_0, sh_1, sh_2], axis=-1)
            points3D_rgb = torch.from_numpy(np.clip(rgb * 255, 0, 255).astype(np.uint8))
        elif "red" in field_names:
            points3D_rgb = np.stack([vertices["red"], vertices["green"], vertices["blue"]], axis=-1)
            points3D_rgb = torch.from_numpy(points3D_rgb.astype(np.uint8))
        else:
            points3D_rgb = torch.zeros_like(points3D, dtype=torch.uint8)

        points3D_quat = None
        if "rot_0" in field_names:
            points3D_quat = torch.from_numpy(
                np.stack([vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"]], axis=-1).astype(np.float32)
            )

        # Normlaize means
        if "rot_0" in field_names or "scale_0" in field_names:
            points3D = (points3D / 3.0) + 0.5

        out = {
            "points3D_xyz": points3D,
            "points3D_rgb": points3D_rgb,
        }

        if "scale_0" in field_names:
            points3D_scale = torch.exp(torch.from_numpy(np.stack([vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]], axis=-1).astype(np.float32)))
            out["points3D_scale"] = points3D_scale

        if points3D_quat is not None:
            out["points3D_quat"] = points3D_quat
            
        return out
    

@dataclass
class GENIENerfstudioDataParserConfig(NerfstudioDataParserConfig):
    """Configuration for GENIE Nerfstudio data parser."""

    _target: Type = field(default_factory=lambda: GENIENerfstudio)
    """target class to instantiate"""
    downscale_factor: Optional[int] = None
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    load_3D_points: bool = True
    """Whether to load the 3D points from the colmap reconstruction."""


class GENIENerfstudio(Nerfstudio):
    """GENIE Nerfstudio data parser.

    This class extends the NerfstudioDataParser to handle GENIE-specific data parsing.
    """

    def __init__(self, config: GENIENerfstudioDataParserConfig):

        config.ply_path ="sparse_pc.ply"
        super().__init__(config)

    def _load_3D_points(self, ply_file_path: Path, transform_matrix: torch.Tensor, scale_factor: float):

        with open(ply_file_path, "rb") as f:
            # Parse header
            properties = []
            vertex_count = 0
            
            while True:
                line = f.readline().decode("utf-8").strip()
                if line == "end_header":
                    break
                
                if line.startswith("element vertex"):
                    vertex_count = int(line.split()[-1])
                
                if line.startswith("property"):
                    parts = line.split()
                    # format: property <type> <name>
                    dtype_str = parts[1]
                    name = parts[2]
                    
                    if dtype_str == "float":
                        np_dtype = "f4"
                    elif dtype_str == "uchar":
                        np_dtype = "u1"
                    else:
                        # Handle other types if necessary
                        mapping = {
                            "double": "f8", "int": "i4", "uint": "u4", 
                            "short": "i2", "ushort": "u2", "char": "i1"
                        }
                        np_dtype = mapping.get(dtype_str, "f4")
                    
                    properties.append((name, np_dtype))

            # Read binary data
            dtype = np.dtype(properties)
            vertices = np.fromfile(f, dtype=dtype, count=vertex_count)

        points3D = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=-1)
        points3D = (
            np.concatenate(
                (
                    points3D,
                    np.ones_like(points3D[..., :1]),
                ),
                -1,
            )
            @ transform_matrix.T.cpu().detach().numpy()
        )
        points3D *= scale_factor
        points3D = torch.from_numpy(points3D.astype(np.float32))

        # Check fields using dtype.names
        field_names = vertices.dtype.names

        if "f_dc_0" in field_names:
            sh_0 = vertices["f_dc_0"]
            sh_1 = vertices["f_dc_1"]
            sh_2 = vertices["f_dc_2"]
            # SH to RGB (DC component only)
            rgb = 0.5 + 0.28209479177387814 * np.stack([sh_0, sh_1, sh_2], axis=-1)
            points3D_rgb = torch.from_numpy(np.clip(rgb * 255, 0, 255).astype(np.uint8))
        elif "red" in field_names:
            points3D_rgb = np.stack([vertices["red"], vertices["green"], vertices["blue"]], axis=-1)
            points3D_rgb = torch.from_numpy(points3D_rgb.astype(np.uint8))
        else:
            points3D_rgb = torch.zeros_like(points3D, dtype=torch.uint8)

        points3D_quats = None
        if "rot_0" in field_names:
            points3D_quats = torch.from_numpy(
                np.stack([vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"]], axis=-1).astype(np.float32)
            )
            # Apply transform_matrix rotation to quats
            R_tf = torch.as_tensor(transform_matrix[:3, :3], dtype=points3D_quats.dtype, device=points3D_quats.device)
            q_tf = rotmat_to_quat(R_tf.expand(points3D_quats.shape[0], -1, -1))
            points3D_quats = quat_multiply(q_tf, points3D_quats)

        # Rotate -90° about x to keep consistent with points
        points3D, points3D_quats = rotate_gaussians_x90(points3D, points3D_quats)

        max_points = min(1000000, points3D.shape[0])
        rand_indices = np.random.permutation(points3D.shape[0])[:max_points]
        rand_indices_t = torch.from_numpy(rand_indices).long()

        print(f"Loaded {points3D.shape[0]} 3D points from {ply_file_path}")
        print(f"Downsampling to {rand_indices_t.shape[0]} 3D points")

        out = {
            "points3D_xyz": points3D[rand_indices_t],
            "points3D_rgb": points3D_rgb[rand_indices_t],
        }

        if "scale_0" in field_names:
            points3D_scale = torch.exp(torch.from_numpy(np.stack([vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]], axis=-1).astype(np.float32)))
            out["points3D_scale"] = points3D_scale[rand_indices_t] * scale_factor

        if "rot_0" in field_names:
            assert points3D_quats is not None
            out["points3D_quat"] = points3D_quats[rand_indices_t]
            
        return out