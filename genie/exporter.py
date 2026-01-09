"""
Script for exporting NeRF into other formats.
"""

from __future__ import annotations

import torch
import tyro
import trimesh
import glob
import os
import numpy as np
import open3d as o3d

from dataclasses import dataclass
from typing import Union, Optional
from typing_extensions import Annotated
from sklearn.neighbors import KDTree
from pathlib import Path
from tqdm import tqdm

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE

from genie.genie_model import GenieModel


@dataclass
class ExportTriangleSoup():
    """
    Export a triangle soup from a gaussians.
    """
    load_config: Path
    """Path to the configuration file."""
    output_filename: str = "triangle_soup.ply"
    """Name of the output file."""
    scale: float = 0.1
    """Scale factor for the triangles."""
    scale_mesh: float = 1.0

    def main(self) -> None:
        """Export triangle soup"""

        assert self.load_config.exists(), f"Configuration file {self.load_config} does not exist."
        _, pipeline, _, _ = eval_setup(self.load_config)

        model = pipeline.model
        assert isinstance(model, GenieModel), "Pipeline model must be GenieModel for triangle soup export."
        
        with torch.no_grad():
            means = model.field.mlp_base.encoder.means
            covs = torch.exp(model.field.mlp_base.encoder.log_covs)

            CONSOLE.print(f"Exporting triangle soup with {means.shape[0]} gaussians.")

            if covs.shape[1] == 2:
                CONSOLE.print("Covariance matrices are 2D, conveerting to 3D.")
                covs = torch.cat([covs, torch.ones(covs.shape[0], 1, device=covs.device) * 1e-6], dim=-1)

            if covs.ndim == 2:
                CONSOLE.print("Creating full covariance matrices from diagonal entries.")
                covs = torch.diag_embed(covs)

            # Compute eigenvalues and eigenvectors for each covariance matrix
            eigenvalues, eigenvectors = torch.linalg.eigh(covs)
            sigmas = torch.sqrt(torch.clamp(eigenvalues, min=0.0))

            idx = torch.argsort(sigmas, descending=True, dim=1)[:, :2]
            batch_indices = torch.arange(eigenvectors.shape[0], device=eigenvectors.device).unsqueeze(-1).expand(-1, 2)
            top2_vecs = eigenvectors[batch_indices, :, idx]
            top2_sigmas = sigmas[batch_indices, idx]

            # Compute triangles
            top2_vecs = top2_vecs.transpose(1, 2)
            arms = top2_vecs * top2_sigmas.unsqueeze(1) * self.scale
            center_np = (means * self.scale_mesh).detach().cpu().numpy()
            arm1 = arms[:, :, 0].detach().cpu().numpy()
            arm2 = arms[:, :, 1].detach().cpu().numpy()
            v0 = center_np
            v1 = center_np + arm1
            v2 = center_np + arm2
            triangles = np.stack([v0, v1, v2], axis=1).tolist()

            # Convert to Open3D TriangleMesh and export
            vertices = []
            faces = []
            for idx, tri in enumerate(triangles):
                base = len(vertices)
                vertices.extend(tri)
                faces.append([base, base + 1, base + 2])

            vertices_np = np.array(vertices)

            # Save triangle soup
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices_np))
            mesh.triangles = o3d.utility.Vector3iVector(np.array(faces))
            o3d.io.write_triangle_mesh(str(self.load_config.parent / self.output_filename), mesh)

            CONSOLE.print(f"Exported triangle soup to {self.load_config.parent / self.output_filename}")


@dataclass
class ExportTetrahedronSoup():
    """
    Export a triangle soup from a gaussians.
    """
    load_config: Path
    """Path to the configuration file."""
    output_filename: str = "tetrahedron_soup.ply"
    """Name of the output file."""
    scale: float = 0.1
    """Scale factor for the triangles."""
    scale_mesh: float = 1.0

    def main(self) -> None:
        """Export triangle soup"""

        assert self.load_config.exists(), f"Configuration file {self.load_config} does not exist."
        _, pipeline, _, _ = eval_setup(self.load_config)

        model = pipeline.model
        assert isinstance(model, GenieModel), "Pipeline model must be GenieModel for triangle soup export."
        
        with torch.no_grad():
            means = model.field.mlp_base.encoder.means
            covs = torch.exp(model.field.mlp_base.encoder.log_covs)
            quats = model.field.mlp_base.encoder.quats

            # Normalize quaternions to ensure valid rotation matrices
            quats = quats / quats.norm(dim=-1, keepdim=True)

            CONSOLE.print(f"Exporting tetrahedron soup with {means.shape[0]} gaussians.")

            if covs.shape[1] == 2:
                CONSOLE.print("Covariance matrices are 2D, conveerting to 3D.")
                covs = torch.cat([covs, torch.ones(covs.shape[0], 1, device=covs.device) * 1e-6], dim=-1)

            # Compute rotation matrices from quaternions
            w, x, y, z = quats.unbind(-1)
            xx = x * x
            yy = y * y
            zz = z * z
            xy = x * y
            xz = x * z
            yz = y * z
            wx = w * x
            wy = w * y
            wz = w * z
            
            row0 = torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1)
            row1 = torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1)
            row2 = torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1)
            
            R = torch.stack([row0, row1, row2], dim=-2)
            sigmas = torch.sqrt(covs)
            eigenvectors = R

            # Compute tetrahedrons
            arms = eigenvectors * sigmas.unsqueeze(1) * self.scale
            center_np = (means * self.scale_mesh).detach().cpu().numpy()
            arm1 = arms[:, :, 0].detach().cpu().numpy()
            arm2 = arms[:, :, 1].detach().cpu().numpy()
            arm3 = arms[:, :, 2].detach().cpu().numpy()
            
            v0 = center_np
            v1 = center_np + arm1
            v2 = center_np + arm2
            v3 = center_np + arm3

            # Stack all vertices: (N, 4, 3) -> (N*4, 3)
            all_vertices = np.stack([v0, v1, v2, v3], axis=1).reshape(-1, 3)
            num_gaussians = means.shape[0]
            base_indices = np.arange(num_gaussians) * 4
            
            f1 = np.stack([base_indices, base_indices + 2, base_indices + 1], axis=1)
            f2 = np.stack([base_indices, base_indices + 1, base_indices + 3], axis=1)
            f3 = np.stack([base_indices, base_indices + 3, base_indices + 2], axis=1)
            f4 = np.stack([base_indices + 1, base_indices + 2, base_indices + 3], axis=1)
            
            all_faces = np.concatenate([f1, f2, f3, f4], axis=0)

            # Save triangle soup
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(all_vertices)
            mesh.triangles = o3d.utility.Vector3iVector(all_faces)
            o3d.io.write_triangle_mesh(str(self.load_config.parent / self.output_filename), mesh)

            CONSOLE.print(f"Exported tetrahedron soup to {self.load_config.parent / self.output_filename}")


@dataclass
class ExportPlyFromObj():
    """
    Export a point cloud sampled from a mesh .obj file to a .ply file, or batch process all .obj/.ply files in a folder.
    If ply_mode is True, load .ply files, normalize, and save.
    """
    obj_path: str = None
    """Path to the .obj mesh file (used if batch_folder is not set)."""
    output_filename: str = "seed_points.ply"
    """Name of the output .ply file (used if batch_folder is not set)."""
    gausses_per_face: int = 3
    """Number of points to sample per mesh face."""
    batch_folder: str = None
    """If set, process all .obj files in this folder."""
    output_folder: Optional[str] = None
    """If batch_folder is set, save .ply files here with the same base names."""
    ply_mode: bool = False
    """If True, load .ply files instead of .obj and normalize them."""
    scale: float = 1.0
    """Scale factor to apply to the output point cloud or mesh."""

    def process_obj(self, obj_path, ply_path):
        mesh = trimesh.load(obj_path, process=False, force='mesh')
        vertices = mesh.vertices  # (V, 3) numpy array
        faces = mesh.faces  # (F, 3) numpy array
        rot_mat = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ], dtype=np.float32)
        vertices = vertices @ rot_mat.T
        F = faces.shape[0]
        K = self.gausses_per_face
        weights = np.random.dirichlet([1, 1, 1], size=(F, K))
        face_vertices = vertices[faces]  # (F, 3, 3)
        pts = np.einsum('fkj,fjd->fkd', weights, face_vertices)
        pts = pts.reshape(-1, 3)
        pts = (np.array(pts, dtype=np.float32) / 3) + 0.5
        # Scale around the center point (0.5, 0.5, 0.5)
        center = np.array([0.5, 0.5, 0.5])
        pts = (pts - center) * self.scale + center
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        o3d.io.write_point_cloud(str(ply_path), pcd)
        print(f"Exported point cloud to {ply_path}")

    def process_ply(self, ply_path_in, ply_path_out):
        # Try to read as mesh first, fallback to point cloud
        try:
            mesh = o3d.io.read_triangle_mesh(ply_path_in)
            if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
                # It's a mesh, process as mesh
                vertices = np.asarray(mesh.vertices)
                vertices = (vertices / 3) + 0.5
                # Scale around the center point (0.5, 0.5, 0.5)
                center = np.array([0.5, 0.5, 0.5])
                vertices = (vertices - center) * self.scale + center
                mesh.vertices = o3d.utility.Vector3dVector(vertices)
                mesh.compute_vertex_normals()
                o3d.io.write_triangle_mesh(str(ply_path_out), mesh)
                print(f"Exported normalized mesh to {ply_path_out} with {len(vertices)} vertices and {len(mesh.triangles)} faces")
            else:
                # No triangles, treat as point cloud
                raise ValueError("No triangles found, treating as point cloud")
        except:
            # Fallback to point cloud processing
            pcd = o3d.io.read_point_cloud(ply_path_in)
            pts = np.asarray(pcd.points)
            pts = (pts / 3) + 0.5
            # Scale around the center point (0.5, 0.5, 0.5)
            center = np.array([0.5, 0.5, 0.5])
            pts = (pts - center) * self.scale + center
            pcd.points = o3d.utility.Vector3dVector(pts)
            o3d.io.write_point_cloud(str(ply_path_out), pcd)
            print(f"Exported normalized point cloud to {ply_path_out}")

    def main(self) -> None:
        import glob
        import os
        if self.batch_folder:
            assert self.output_folder, "output_folder must be set if batch_folder is used."
            os.makedirs(self.output_folder, exist_ok=True)
            ext = '*.ply' if self.ply_mode else '*.obj'
            files = sorted(glob.glob(os.path.join(self.batch_folder, ext)))
            print(f"Found {len(files)} {ext} files in {self.batch_folder}")
            for file_path in tqdm(files):
                base = os.path.splitext(os.path.basename(file_path))[0]
                ply_path = os.path.join(self.output_folder, base + '.ply')
                if self.ply_mode:
                    self.process_ply(file_path, ply_path)
                else:
                    self.process_obj(file_path, ply_path)
        else:
            # Single file mode
            if self.ply_mode:
                ply_path_in = os.path.expanduser(self.obj_path)
                ply_path_out = self.output_filename
                self.process_ply(ply_path_in, ply_path_out)
            else:
                obj_path = os.path.expanduser(self.obj_path)
                ply_path = self.output_filename
                self.process_obj(obj_path, ply_path)


@dataclass
class ExportPlyFromEdits():

    load_config: Path
    """Path to the configuration file."""
    scale: int = 1
    """Scale factor for the edited mesh."""

    def write_ply_pointcloud(self, points, filepath, verbose=False):
        """Write points as PLY point cloud file"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(str(filepath), pcd)
        if verbose:
            print(f'Point cloud saved to: {filepath} with {len(points)} points')

    def calc_new_vertices_position(self,alpha, normal, vec_1, vec_2, vertice_1):
        vertices = torch.bmm(
            alpha.permute(0, 2, 1),torch.stack((normal, vec_1, vec_2), dim=1)
        ).reshape(-1, 3)  + vertice_1
        return vertices

    def main(self):

        output_dir = self.load_config.parent
        triangle_soup_path = output_dir / "triangle_soup.ply"
        mesh_path = output_dir / "reference_meshes/00000.ply"

        assert triangle_soup_path.exists(), f"Triangle soup file {triangle_soup_path} does not exist."
        assert mesh_path.exists(), f"Reference mesh file {mesh_path} does not exist."

        if not (output_dir / "camera_path").exists():
            os.makedirs(output_dir / "camera_path")

        # Triangle_soup is interpreted as psuedomesh
        triangle_soup = trimesh.load(str(triangle_soup_path), force='mesh')
        triangle_soup = torch.tensor(triangle_soup.triangles).cuda().float()
        mesh = trimesh.load(str(mesh_path), force='mesh')
        mesh_triangles = torch.tensor(mesh.triangles).cuda().float()


        files = sorted(glob.glob(os.path.join(mesh_path.parent, "*.ply")))
        print(f"Found {len(files)} files in {mesh_path.parent}")
        for edited_mesh_path in tqdm(files):

            # Pseudomesh transformation based on mesh use triangle
            mesh_edited = trimesh.load(edited_mesh_path,  force='mesh')
            mesh_edited_triangles = torch.tensor(mesh_edited.triangles).cuda().float()

            # Find the closest face (triangle)
            tree = KDTree(torch.mean(mesh_triangles, dim = 1).cpu())
            index_of_closest = tree.query(
                torch.mean(triangle_soup, dim = 1).cpu(), k = 1, return_distance = False
            )
            closest_triangle = mesh_triangles[index_of_closest.flatten()]

            # Vertices of the closest face from references mesh to init psuedomesh
            v1 = closest_triangle[:, 0,:]
            v2 = closest_triangle[:, 1,:]
            v3 = closest_triangle[:, 2,:]

            v2_v1 = v2 - v1
            v3_v1 = v3 - v1

            # Use linalg.cross instead of deprecated cross
            normal = torch.linalg.cross(v2_v1, v3_v1, dim=-1)
            
            # Add small regularization to prevent division by zero
            eps = 1e-8
            v2_v1_norm = torch.linalg.vector_norm(v2_v1, dim=-1, keepdim=True)
            v3_v1_norm = torch.linalg.vector_norm(v3_v1, dim=-1, keepdim=True)
            normal_norm = torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
            
            v2_v1 = v2_v1 / torch.clamp(v2_v1_norm, min=eps)
            v3_v1 = v3_v1 / torch.clamp(v3_v1_norm, min=eps)
            normal = normal / torch.clamp(normal_norm, min=eps)
            
            A_T = torch.stack([normal, v2_v1, v3_v1]).permute(1, 2, 0)

            # Vertices psuedomesh
            w1 = triangle_soup[:, 0,:]
            w2 = triangle_soup[:, 1,:]
            w3 = triangle_soup[:, 2,:]

            # Calculate alpha with error handling for singular matrices
            try:
                # Add small regularization to diagonal for numerical stability
                reg = 1e-6 * torch.eye(3, device=A_T.device, dtype=A_T.dtype).unsqueeze(0).expand(A_T.shape[0], -1, -1)
                A_T_reg = A_T + reg
                
                alpha_w1 = torch.linalg.solve(A_T_reg, w1 - v1).reshape(A_T.shape[0], 3, 1)
            except torch._C._LinAlgError:
                # Fallback: use pseudoinverse for singular matrices
                print("Warning: Using pseudoinverse due to singular matrices")
                alpha_w1 = torch.linalg.pinv(A_T) @ (w1 - v1).unsqueeze(-1)
                alpha_w1 = alpha_w1.reshape(A_T.shape[0], 3, 1)

            # Find referenced triangle based on edited mesh and init mesh
            referenced_triangle = mesh_edited_triangles[index_of_closest.flatten()]

            v1_referenced = referenced_triangle[:, 0,:]
            v2_referenced = referenced_triangle[:, 1,:]
            v3_referenced = referenced_triangle[:, 2,:]

            referenced_v2_v1 = v2_referenced - v1_referenced
            referenced_v3_v1 = v3_referenced - v1_referenced
            normal = torch.linalg.cross(referenced_v2_v1, referenced_v3_v1, dim=-1)

            # Normalize with regularization
            referenced_v2_v1 = referenced_v2_v1 / torch.clamp(torch.linalg.vector_norm(referenced_v2_v1, dim=-1, keepdim=True), min=eps)
            referenced_v3_v1 = referenced_v3_v1 / torch.clamp(torch.linalg.vector_norm(referenced_v3_v1, dim=-1, keepdim=True), min=eps)
            normal = normal / torch.clamp(torch.linalg.vector_norm(normal, dim=-1, keepdim=True), min=eps)

            # Calculate new vertices of edited psuedomesh
            w1_edited = self.calc_new_vertices_position(alpha_w1, normal, referenced_v2_v1, referenced_v3_v1, v1_referenced)
            vertices = w1_edited

            filename = str(edited_mesh_path).replace("reference_meshes", "camera_path")
            self.write_ply_pointcloud(points=(vertices * self.scale).detach().cpu().numpy(), filepath=filename, verbose=True)


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[ExportTriangleSoup, tyro.conf.subcommand(name="triangles")],
        Annotated[ExportTetrahedronSoup, tyro.conf.subcommand(name="tetrahedrons")],
        Annotated[ExportPlyFromObj, tyro.conf.subcommand(name="ply-from-obj")],
        Annotated[ExportPlyFromEdits, tyro.conf.subcommand(name="ply-from-edits")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(Commands)  # noqa