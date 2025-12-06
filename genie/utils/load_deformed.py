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

def load_deformed_tetrahedrons(model: GENIEModel, old_quats: torch.Tensor, old_variances: torch.Tensor, ply_path: str, ref_ply_path: str, scale: float = 0.1, scale_mesh: float = 1.0):
    """
    Load deformed tetrahedrons from a PLY file and update the model's Gaussians using deformation gradient.
    
    Args:
        model: The GENIEModel to update.
        ply_path: Path to the PLY file containing the deformed tetrahedron soup.
        ref_ply_path: Path to the PLY file containing the reference (undeformed) tetrahedron soup.
        scale: The scale factor used during export for the arms.
        scale_mesh: The scale factor used during export for the means.
    """
    
    # Load the deformed mesh
    mesh = o3d.io.read_triangle_mesh(ply_path)
    vertices = np.asarray(mesh.vertices)
    
    # Load the reference mesh
    ref_mesh = o3d.io.read_triangle_mesh(ref_ply_path)
    ref_vertices = np.asarray(ref_mesh.vertices)
    
    num_vertices = vertices.shape[0]
    num_gaussians = num_vertices // 4
    print(f"Loading {num_gaussians} Gaussians.")
    
    assert ref_vertices.shape[0] == num_vertices, "Reference and deformed meshes must have the same number of vertices"

    # The exporter stacks vertices as [v0, v1, v2, v3] for each Gaussian.
    # v0 is center, v1, v2, v3 are tips of the arms
    
    def get_means_and_arms(verts):
        verts_reshaped = verts.reshape(num_gaussians, 4, 3)
        v0 = verts_reshaped[:, 0, :] # Center
        v1 = verts_reshaped[:, 1, :] # Center + Arm 1
        v2 = verts_reshaped[:, 2, :] # Center + Arm 2
        v3 = verts_reshaped[:, 3, :] # Center + Arm 3
        
        means = v0
        arm1 = v1 - v0
        arm2 = v2 - v0
        arm3 = v3 - v0
        
        # Stack arms to form the matrix M where columns are arms
        # M shape: (N, 3, 3)
        M = np.stack([arm1, arm2, arm3], axis=2)
        return means, M

    means_def_np, M_def_np = get_means_and_arms(vertices)
    _, M_ref_np = get_means_and_arms(ref_vertices)

    device = model.field.mlp_base.encoder.means.device
    
    # Convert to tensors
    means_def = torch.tensor(means_def_np, dtype=torch.float32, device=device)
    M_def = torch.tensor(M_def_np, dtype=torch.float32, device=device)
    M_ref = torch.tensor(M_ref_np, dtype=torch.float32, device=device)
    
    # Compute Deformation Gradient A
    # M_def = A @ M_ref  =>  A = M_def @ M_ref^-1
    M_ref_inv = torch.linalg.inv(M_ref)
    A = torch.matmul(M_def, M_ref_inv)

    # Get current model covariance
    encoder = model.field.mlp_base.encoder
    
    # Construct rotation matrix
    R_old = encoder.quat_to_rotmat(old_quats) # (N, 3, 3)
    
    # Construct full covariance matrix Sigma_old
    # Sigma = R * diag(var) * R^T
    Sigma_old = torch.matmul(R_old, torch.matmul(torch.diag_embed(old_variances), R_old.transpose(-2, -1)))
    
    # Update Covariance: Sigma_new = A * Sigma_old * A^T
    Sigma_new = torch.matmul(A, torch.matmul(Sigma_old, A.transpose(-2, -1)))
    
    # Decompose Sigma_new to get new parameters
    U, S, _ = torch.linalg.svd(Sigma_new)
    
    new_variances = S
    new_R = U
    
    # Ensure right-handed rotation
    det = torch.linalg.det(new_R)
    mask = det < 0
    if mask.any():
        new_R[mask, :, 2] *= -1
        
    new_quats = matrix_to_quaternion(new_R)
    new_log_covs = torch.log(torch.clamp(new_variances, min=1e-6))
    
    # Update means
    new_means = means_def / scale_mesh
    
    # Update model parameters
    encoder.means.data = new_means
    encoder.log_covs.data = new_log_covs
    encoder.quats.data = new_quats
    
    return A
