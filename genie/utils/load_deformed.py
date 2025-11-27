import torch
import numpy as np
import open3d as o3d
from genie.genie_model import GENIEModel
from scipy.spatial.transform import Rotation as R_scipy


def matrix_to_quaternion(M):
    # M: (B, 3, 3) torch tensor
    M_np = M.detach().cpu().numpy()
    
    # Create rotation object
    r = R_scipy.from_matrix(M_np)
    
    # Get quaternions (x, y, z, w) format in scipy
    q_np = r.as_quat()
    q_wxyz = np.stack([q_np[:, 3], q_np[:, 0], q_np[:, 1], q_np[:, 2]], axis=1)
    
    return torch.tensor(q_wxyz, dtype=torch.float32, device=M.device)

def load_deformed_tetrahedrons(model: GENIEModel, ply_path: str, scale: float = 0.1, scale_mesh: float = 1.0):
    """
    Load deformed tetrahedrons from a PLY file and update the model's Gaussians.
    
    Args:
        model: The GENIEModel to update.
        ply_path: Path to the PLY file containing the deformed tetrahedron soup.
        scale: The scale factor used during export for the arms.
        scale_mesh: The scale factor used during export for the means.
    """
    
    # Load the mesh
    mesh = o3d.io.read_triangle_mesh(ply_path)
    vertices = np.asarray(mesh.vertices)
    
    num_vertices = vertices.shape[0]
    num_gaussians = num_vertices // 4
    print(f"Loading {num_gaussians} Gaussians.")
    
    # The exporter stacks vertices as [v0, v1, v2, v3] for each Gaussian.
    vertices_reshaped = vertices.reshape(num_gaussians, 4, 3)
    v0 = vertices_reshaped[:, 0, :] # Center
    v1 = vertices_reshaped[:, 1, :] # Center + Arm 1
    v2 = vertices_reshaped[:, 2, :] # Center + Arm 2
    v3 = vertices_reshaped[:, 3, :] # Center + Arm 3

    # Convert to torch tensors
    device = model.field.mlp_base.encoder.means.device
    v0 = torch.tensor(v0, dtype=torch.float32, device=device)
    v1 = torch.tensor(v1, dtype=torch.float32, device=device)
    v2 = torch.tensor(v2, dtype=torch.float32, device=device)
    v3 = torch.tensor(v3, dtype=torch.float32, device=device)
    
    # Recover means
    new_means = v0 / scale_mesh    
    arm1 = v1 - v0
    arm2 = v2 - v0
    arm3 = v3 - v0
    
    # Stack arms to form the matrix of principal axes (scaled eigenvectors)
    M = torch.stack([arm1, arm2, arm3], dim=2)
    M_scaled = M / scale
    
    # Perform SVD on the scaled arms matrix directly to recover the new Gaussian parameters
    # The arms represent the transformed axes. If M is the matrix of arms, the covariance is M @ M.T
    # SVD: M = U @ S @ Vh
    # Covariance = (U @ S @ Vh) @ (Vh.T @ S @ U.T) = U @ S^2 @ U.T
    # Thus, U represents the rotation (orientation) and S represents the scales (sigmas)
    U, S, Vh = torch.linalg.svd(M_scaled)
    
    new_sigmas = S
    R_clean = U
    
    # Check determinant to ensure right-handed system (proper rotation matrix)
    # Since the Gaussian is symmetric, flipping an axis doesn't change the shape, 
    # but we need a valid rotation matrix (det=1) for the quaternion conversion.
    det = torch.linalg.det(R_clean)
    mask = det < 0
    if mask.any():
        # Flip the last column (z-axis) for matrices with negative determinant
        R_clean[mask, :, 2] *= -1

    new_quats = matrix_to_quaternion(R_clean)
        
    new_covs = new_sigmas ** 2
    new_log_covs = torch.log(torch.clamp(new_covs, min=1e-6))
    
    # Update the model
    encoder = model.field.mlp_base.encoder    
    encoder.means.data = new_means
    encoder.log_covs.data = new_log_covs
    encoder.quats.data = new_quats
    
    print("Model updated with deformed tetrahedrons.")
