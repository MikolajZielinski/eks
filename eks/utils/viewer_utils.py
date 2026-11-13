import numpy as np
import torch
import trimesh

from nerfstudio.viewer.viewer_elements import ViewerElement
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.viewer.viewer import VISER_NERFSTUDIO_SCALE_RATIO
from viser import ViserServer
from viser._scene_handles import PointCloudHandle, GaussianSplatHandle


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


class ViewerGaussianSplats(ViewerElement[bool]):
    """A point cloud in the viewer

    Args:
        name: The name of the point cloud
        visible: If the point cloud is visible
    """

    scene_handle: GaussianSplatHandle

    def __init__(self, name: str, aabb: SceneBox, means: np.ndarray, covariances: np.ndarray, quats: np.ndarray, confidence: np.ndarray, gradients: np.ndarray = None, show_gradients: bool = False, visible: bool = True):
        self.aabb = aabb
        self.means = means
        self.covariances = covariances
        self.quats = quats
        self.confidence = confidence
        self.gradients = gradients
        self.show_gradients = show_gradients
        super().__init__(name, visible=visible)

    def update(self, means: np.ndarray, covariances: np.ndarray, quats: np.ndarray, confidence: np.ndarray, gradients: np.ndarray = None) -> None:
        """Update the point cloud with new points."""
        self.means = means
        self.covariances = covariances
        self.quats = quats
        self.confidence = confidence
        self.gradients = gradients
        if self.viser_server is not None:
            self._create_scene_handle(self.viser_server)

    def _create_scene_handle(self, viser_server: ViserServer) -> None:
        means = self.means.reshape(-1, self.means.shape[-1])
        aabb =  self.aabb.aabb.reshape(2, 3).detach().cpu().numpy()
        aabb_min, aabb_max = aabb[0] * VISER_NERFSTUDIO_SCALE_RATIO, aabb[1] * VISER_NERFSTUDIO_SCALE_RATIO
        means = means * (aabb_max - aabb_min) + aabb_min

        # Convert scales and quats to covariance matrices
        scales = np.sqrt(self.covariances) * VISER_NERFSTUDIO_SCALE_RATIO
        quats = self.quats

        # Normalize quaternions
        quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)

        # Create rotation matrices (assuming w, x, y, z convention)
        w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

        R = np.zeros((quats.shape[0], 3, 3))
        R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
        R[:, 0, 1] = 2 * (x * y - w * z)
        R[:, 0, 2] = 2 * (x * z + w * y)
        R[:, 1, 0] = 2 * (x * y + w * z)
        R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
        R[:, 1, 2] = 2 * (y * z - w * x)
        R[:, 2, 0] = 2 * (x * z - w * y)
        R[:, 2, 1] = 2 * (y * z + w * x)
        R[:, 2, 2] = 1 - 2 * (x**2 + y**2)

        # Compute covariance matrices: Sigma = R * S^2 * R^T
        S_squared = scales ** 2
        # Multiply columns of R by S_squared (broadcasting)
        R_scaled = R * S_squared[:, None, :]
        covariances = R_scaled @ np.transpose(R, (0, 2, 1))

        if self.show_gradients and self.gradients is not None:
             # Normalize gradients for visualization
             grads = self.gradients
             max_grad = np.max(grads) if grads.size > 0 else 1.0
             if max_grad == 0: max_grad = 1.0
             norm_grads = grads / max_grad
             
             # Map gradients to Red (high gradient) -> Blue (low gradient)
             colors = np.zeros((means.shape[0], 3))
             colors[:, 0] = norm_grads * 255.0  # Red
             colors[:, 2] = (1.0 - norm_grads) * 255.0 # Blue
             
        else:
             color_coeffs = np.random.uniform(0.4, 1.0, size=(means.shape[0]))
             colors = np.tile((0, 1, 1), means.shape[0]).reshape(-1, 3) * color_coeffs[:, None]
             colors[:, 1] *= (1 - self.confidence)
             colors[:, 2] *= self.confidence

        opacities = np.ones((means.shape[0], 1))

        self.scene_handle = viser_server.scene.add_gaussian_splats(
            f"/{self.name}",
            centers=means,
            covariances=covariances,
            rgbs=colors,
            opacities=opacities
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
            # Convert base AABB to numpy once
            base_aabb = self.aabb.aabb.reshape(2, 3).detach().cpu().numpy() * VISER_NERFSTUDIO_SCALE_RATIO
            
            points_list = []
            colors_list = []
            
            # Colors for levels: 0:Red, 1:Green, 2:Blue, 3:Yellow, 4:Cyan, 5:Magenta
            level_colors = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)
            ]

            if self.occ_grid.ndim == 4:
                # Multi-level grid [Levels, Res, Res, Res]
                num_levels = self.occ_grid.shape[0]
                res = self.occ_grid.shape[1]
                device = self.occ_grid.device
                
                # Pre-calculate grid coordinates (same for all levels)
                grid_coords = torch.stack(torch.meshgrid(
                    torch.arange(res, device=device),
                    torch.arange(res, device=device),
                    torch.arange(res, device=device),
                    indexing='ij'
                ), dim=-1).reshape(-1, 3)

                for i in range(num_levels):
                    level_occ = self.occ_grid[i]
                    if not torch.any(level_occ):
                        continue

                    # Select occupied voxels for this level
                    mask = level_occ.reshape(-1)
                    occupied_indices = grid_coords[mask].cpu().numpy()
                    
                    # Metric scale for this level (2^i)
                    level_scale = 2.0 ** i
                    level_aabb = base_aabb * level_scale
                    
                    # Compute world centers
                    voxel_size = (level_aabb[1] - level_aabb[0]) / res
                    occupied_centers = level_aabb[0] + (occupied_indices + 0.5) * voxel_size
                    
                    points_list.append(occupied_centers)
                    
                    # Color coding
                    color = level_colors[i % len(level_colors)]
                    points_colors = np.tile(color, occupied_centers.shape[0]).reshape(-1, 3)
                    colors_list.append(points_colors)

            else:
                # Single level grid
                res = self.occ_grid.shape[0]
                device = self.occ_grid.device

                grid_coords = torch.stack(torch.meshgrid(
                    torch.arange(res, device=device),
                    torch.arange(res, device=device),
                    torch.arange(res, device=device),
                    indexing='ij'
                ), dim=-1).reshape(-1, 3)

                mask = self.occ_grid.view(-1)
                occupied_indices = grid_coords[mask].cpu().numpy()

                voxel_size = (base_aabb[1] - base_aabb[0]) / res
                occupied_centers = base_aabb[0] + (occupied_indices + 0.5) * voxel_size

                points_list.append(occupied_centers)
                colors_list.append(np.tile((255, 0, 0), occupied_centers.shape[0]).reshape(-1, 3))

            if len(points_list) > 0:
                all_points = np.concatenate(points_list, axis=0)
                all_colors = np.concatenate(colors_list, axis=0)
                
                viser_server.scene.add_point_cloud(
                    "/occupied_voxels",
                    points=all_points,
                    colors=all_colors,
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