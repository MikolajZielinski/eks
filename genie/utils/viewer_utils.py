import numpy as np
import torch
import trimesh

from nerfstudio.viewer.viewer_elements import ViewerElement
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.viewer.viewer import VISER_NERFSTUDIO_SCALE_RATIO
from viser import ViserServer
from viser._scene_handles import PointCloudHandle


class ViewerPointCloud(ViewerElement[bool]):
    """A point cloud in the viewer

    Args:
        name: The name of the point cloud
        visible: If the point cloud is visible
    """

    scene_handle: PointCloudHandle

    def __init__(self, name: str, aabb: SceneBox, points: np.ndarray, confidence: np.ndarray, gradients: np.ndarray = None, show_gradients: bool = False, visible: bool = True):
        self.aabb = aabb
        self.points = points
        self.confidence = confidence
        self.gradients = gradients
        self.show_gradients = show_gradients
        super().__init__(name, visible=visible)

    def update(self, points: np.ndarray, confidence: np.ndarray, gradients: np.ndarray = None) -> None:
        """Update the point cloud with new points."""
        self.points = points
        self.confidence = confidence
        self.gradients = gradients
        if self.viser_server is not None:
            self._create_scene_handle(self.viser_server)

    def _create_scene_handle(self, viser_server: ViserServer) -> None:
        points = self.points.reshape(-1, self.points.shape[-1])
        aabb =  self.aabb.aabb.reshape(2, 3).detach().cpu().numpy()
        aabb_min, aabb_max = aabb[0] * VISER_NERFSTUDIO_SCALE_RATIO, aabb[1] * VISER_NERFSTUDIO_SCALE_RATIO
        points = points * (aabb_max - aabb_min) + aabb_min

        pcd = trimesh.PointCloud(points)

        if self.show_gradients and self.gradients is not None:
            # Normalize gradients for visualization
            grads = self.gradients
            max_grad = np.max(grads) if grads.size > 0 else 1.0
            if max_grad == 0: max_grad = 1.0
            norm_grads = grads / max_grad
            
            # Map gradients to Red (high gradient) -> Blue (low gradient)
            # Or simply use intensity in Red channel
            colors = np.zeros((pcd.vertices.shape[0], 3))
            colors[:, 0] = norm_grads * 255.0  # Red channel proportional to gradient
            colors[:, 2] = (1.0 - norm_grads) * 255.0 # Blue channel inverse proportional
            
        else:
            color_coeffs = np.random.uniform(0.4, 1.0, size=(pcd.vertices.shape[0]))
            colors = np.tile((0, 255, 255), pcd.vertices.shape[0]).reshape(-1, 3) * color_coeffs[:, None]
            colors[:, 1] *= (1 - self.confidence)
            colors[:, 2] *= self.confidence

        self.scene_handle = viser_server.scene.add_point_cloud(
            f"/{self.name}",
            points=pcd.vertices,
            colors=colors,
            point_size=0.02,
            point_shape="circle"
        )

    def install(self, viser_server: ViserServer) -> None:
        self.viser_server = viser_server
        self._create_scene_handle(viser_server)


class ViewerOccupancyGrid(ViewerElement[bool]):
    """A occupancy grid in the viewer

    Args:
        name: The name of the occupancy grid
        visible: If the occupancy grid is visible
    """

    def __init__(self, name: str, occ_grid: torch.Tensor, aabb: SceneBox, visible: bool = True):
        self.aabb = aabb
        self.occ_grid = occ_grid
        super().__init__(name, visible=visible)

    def update(self, occ_grid: torch.Tensor) -> None:
        """Update the occupancy grid with new occupancy values."""
        self.occ_grid = occ_grid
        if self.viser_server is not None:
            self._create_scene_handle(self.viser_server)

    def _create_scene_handle(self, viser_server: ViserServer) -> None:

        if torch.any(self.occ_grid):
            # Generate voxel grid indices
            res = self.occ_grid.shape[0]
            device = self.occ_grid.device

            grid_coords = torch.stack(torch.meshgrid(
                torch.arange(res, device=device),
                torch.arange(res, device=device),
                torch.arange(res, device=device),
                indexing='ij'
            ), dim=-1).reshape(-1, 3)  # (res^3, 3)

            # Select occupied voxels
            occupied_indices = grid_coords[self.occ_grid.view(-1)]  # (N, 3)

            # Convert to world coordinates
            aabb =  self.aabb.aabb.reshape(2, 3).detach().cpu().numpy() * VISER_NERFSTUDIO_SCALE_RATIO
            voxel_size = (aabb[1] - aabb[0]) / res
            occupied_centers = aabb[0] + (occupied_indices.cpu().numpy() + 0.5) * voxel_size  # (N, 3)

            # Convert to NumPy and visualize using trimesh + your viewer
            occupied_cloud = trimesh.PointCloud(occupied_centers)
            viser_server.scene.add_point_cloud(
                "/occupied_voxels",
                points=occupied_cloud.vertices,
                colors=np.tile((255, 0, 0), occupied_cloud.vertices.shape[0]).reshape(-1, 3),  # red color
                point_size=0.03,
                point_shape="circle",
                visible=True
            )

    def install(self, viser_server: ViserServer) -> None:
        self.viser_server = viser_server
        self._create_scene_handle(viser_server)


class ViewerAABB(ViewerElement[bool]):
    """A bounding box in the viewer

    Args:
        name: The name of the aabb
        visible: If the aabb is visible
    """

    def __init__(self, name: str, aabb: SceneBox, visible: bool = True):
        self.aabb = aabb
        super().__init__(name, visible=visible)

    def _create_scene_handle(self, viser_server: ViserServer) -> None:
        aabb =  self.aabb.aabb.reshape(2, 3).detach().cpu().numpy() * VISER_NERFSTUDIO_SCALE_RATIO
        mesh = trimesh.creation.box(tuple((aabb[1] - aabb[0])))
        viser_server.scene.add_mesh_simple(
            name=f"/{self.name}",
            vertices=mesh.vertices,
            faces=mesh.faces,
            color=(0, 0, 0),
            wireframe=True,
            visible=False
        )

    def install(self, viser_server: ViserServer) -> None:
        self._create_scene_handle(viser_server)