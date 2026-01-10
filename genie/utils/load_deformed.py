import torch
import numpy as np
import open3d as o3d

from genie.genie_model import GENIEModel
from genie.utils.utils import rotmat_to_quat

def load_deformed_tetrahedrons(model: GENIEModel, ply_path: str, ref_ply_path: str, scale: float = 0.1, scale_mesh: float = 1.0):
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
    print(f"Loaded {vertices.shape[0] // 4} tetrahedrons from {ply_path}")
    
    # Load the reference mesh
    ref_mesh = o3d.io.read_triangle_mesh(ref_ply_path)
    ref_vertices = np.asarray(ref_mesh.vertices)
    print(f"Loaded {ref_vertices.shape[0] // 4} tetrahedrons from {ref_ply_path}")
    
    num_vertices = vertices.shape[0]
    num_gaussians = num_vertices // 4
    print(f"Loading {num_gaussians} Gaussians.")
    
    assert ref_vertices.shape[0] == num_vertices, f"Reference and deformed meshes must have the same number of vertices. Reference {ref_vertices.shape[0]}, Deformed {num_vertices}"

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

    # Get current model covariance
    encoder = model.field.mlp_base.encoder

    # Perform SVD on M_def to get new rotation and scales
    U, S, Vh = torch.linalg.svd(M_def.double())
    U = U.float()
    S = S.float()

    # Ensure right-handed rotation
    det = torch.linalg.det(U)
    mask = det < 0
    if mask.any():
        U[mask, :, 2] *= -1

    new_quats = rotmat_to_quat(U)
    
    # Update means
    new_means = means_def / scale_mesh
    
    # Update model parameters
    encoder.means.data = new_means
    encoder.log_covs.data = torch.log(torch.square(torch.clamp(S / scale, min=1e-6)))
    encoder.quats.data = new_quats

    # Compute direction transform (Deformed -> Reference)
    M_ref_pinv = torch.linalg.pinv(M_ref.double())
    A = torch.matmul(M_def.double(), M_ref_pinv)
    U_dt, _, Vh_dt = torch.linalg.svd(A)
    R_dt = torch.matmul(U_dt, Vh_dt)

    # Enforce det = +1
    det_dt = torch.linalg.det(R_dt)
    mask_dt = det_dt < 0
    if mask_dt.any():
        U_dt[mask_dt, :, 2] *= -1
    R_dt = torch.matmul(U_dt, Vh_dt)

    return R_dt.float()
