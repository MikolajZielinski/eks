import torch
import math

def quat_to_rotmat(quats):
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

    return torch.stack([row0, row1, row2], dim=-2)

def rotmat_to_quat(R):
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    t = m00 + m11 + m22

    w = torch.sqrt((1 + t).clamp_min(1e-12)) * 0.5
    x = torch.sign(m21 - m12) * torch.sqrt((1 + m00 - m11 - m22).clamp_min(1e-12)) * 0.5
    y = torch.sign(m02 - m20) * torch.sqrt((1 - m00 + m11 - m22).clamp_min(1e-12)) * 0.5
    z = torch.sign(m10 - m01) * torch.sqrt((1 - m00 - m11 + m22).clamp_min(1e-12)) * 0.5

    quat = torch.stack([w, x, y, z], dim=-1)
    return quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-12)

def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of quaternions in (w, x, y, z) order."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)

    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )

def rotate_gaussians_x90(points: torch.Tensor, quats: torch.Tensor | None = None):
    """Rotate Gaussian means (and optional quats) by -90 deg about the x-axis."""
    rot_xm90 = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=points.dtype,
        device=points.device,
    )
    rotated_points = points @ rot_xm90.T

    rotated_quats = None
    if quats is not None:
        half_angle = math.sqrt(0.5)
        # q for -90° about +x
        q_rot = torch.tensor(
            [half_angle, -half_angle, 0.0, 0.0],
            dtype=quats.dtype,
            device=quats.device,
        )
        rotated_quats = quat_multiply(q_rot, quats)
        rotated_quats = rotated_quats / rotated_quats.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    return rotated_points, rotated_quats