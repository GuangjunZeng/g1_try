# SPDX-License-Identifier: MIT

import torch
from torch import Tensor
from typing import Tuple
import numpy as np


def quat_apply_yaw(quat, vec):
    """
    Apply yaw rotation from quaternion to vector.
    
    Args:
        quat: Quaternion (N, 4) in xyzw format
        vec: Vector (N, 3)
        
    Returns:
        Rotated vector (N, 3)
    """
    quat_yaw = quat.clone()
    quat_yaw[:, 0] = 0.
    quat_yaw[:, 1] = 0.
    quat_yaw = quat_yaw / torch.norm(quat_yaw, dim=-1, keepdim=True)
    return quat_apply(quat_yaw, vec)


def quat_apply(q, v):
    """
    Apply quaternion rotation to vector.
    
    Args:
        q: Quaternion (N, 4) in xyzw format
        v: Vector (N, 3)
        
    Returns:
        Rotated vector (N, 3)
    """
    shape = q.shape
    q_w = q[:, 3]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


def quat_rotate_inverse(q, v):
    """
    Rotate vector by inverse of quaternion.
    
    Args:
        q: Quaternion (N, 4) in xyzw format
        v: Vector (N, 3)
        
    Returns:
        Rotated vector (N, 3)
    """
    q_inv = quat_conjugate(q)
    return quat_apply(q_inv, v)


def quat_conjugate(q):
    """
    Compute quaternion conjugate.
    
    Args:
        q: Quaternion (N, 4) in xyzw format
        
    Returns:
        Conjugate quaternion (N, 4)
    """
    q_conj = q.clone()
    q_conj[:, :3] *= -1
    return q_conj


def wrap_to_pi(angles):
    """
    Wrap angles to [-pi, pi].
    
    Args:
        angles: Angles in radians
        
    Returns:
        Wrapped angles
    """
    angles = angles % (2 * np.pi)
    angles = torch.where(angles > np.pi, angles - 2 * np.pi, angles)
    return angles


def normalize_angle(x):
    """Normalize angle to [-pi, pi]."""
    return torch.atan2(torch.sin(x), torch.cos(x))


def quat_mul(q1, q2):
    """
    Multiply two quaternions.
    
    Args:
        q1: First quaternion (N, 4) in xyzw format
        q2: Second quaternion (N, 4) in xyzw format
        
    Returns:
        Product quaternion (N, 4)
    """
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack([x, y, z, w], dim=-1)


def quat_from_angle_axis(angle, axis):
    """
    Create quaternion from angle-axis representation.
    
    Args:
        angle: Rotation angle in radians (N,)
        axis: Rotation axis (N, 3)
        
    Returns:
        Quaternion (N, 4) in xyzw format
    """
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return torch.cat([xyz, w], dim=-1)


def normalize(v, eps=1e-8):
    """
    Normalize vector.
    
    Args:
        v: Vector (..., N)
        eps: Small value for numerical stability
        
    Returns:
        Normalized vector
    """
    return v / (torch.norm(v, dim=-1, keepdim=True) + eps)


def axis_angle_from_quat(q):
    """
    Convert quaternion to axis-angle representation.
    
    Args:
        q: Quaternion (N, 4) in xyzw format
        
    Returns:
        Tuple of (angle, axis)
    """
    # Normalize quaternion
    q = q / torch.norm(q, dim=-1, keepdim=True)
    
    # Extract angle
    angle = 2 * torch.acos(torch.clamp(q[:, 3], -1, 1))
    
    # Extract axis
    sin_half_angle = torch.sqrt(1 - q[:, 3] ** 2)
    axis = q[:, :3] / (sin_half_angle.unsqueeze(-1) + 1e-8)
    
    # Handle small angles
    small_angle = sin_half_angle < 1e-6
    axis[small_angle] = torch.tensor([1., 0., 0.], device=q.device)
    
    return angle, axis


def quat_to_rotation_matrix(q):
    """
    Convert quaternion to rotation matrix.
    
    Args:
        q: Quaternion (N, 4) in xyzw format
        
    Returns:
        Rotation matrix (N, 3, 3)
    """
    batch_size = q.shape[0]
    
    # Normalize quaternion
    q = q / torch.norm(q, dim=-1, keepdim=True)
    
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    # Build rotation matrix
    R = torch.zeros(batch_size, 3, 3, device=q.device, dtype=q.dtype)
    
    R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    R[:, 0, 1] = 2 * (x*y - w*z)
    R[:, 0, 2] = 2 * (x*z + w*y)
    
    R[:, 1, 0] = 2 * (x*y + w*z)
    R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    R[:, 1, 2] = 2 * (y*z - w*x)
    
    R[:, 2, 0] = 2 * (x*z - w*y)
    R[:, 2, 1] = 2 * (y*z + w*x)
    R[:, 2, 2] = 1 - 2 * (x**2 + y**2)
    
    return R


def rotation_matrix_to_quat(R):
    """
    Convert rotation matrix to quaternion.
    
    Args:
        R: Rotation matrix (N, 3, 3)
        
    Returns:
        Quaternion (N, 4) in xyzw format
    """
    batch_size = R.shape[0]
    q = torch.zeros(batch_size, 4, device=R.device, dtype=R.dtype)
    
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    
    # Case 1: trace > 0
    mask1 = trace > 0
    s = torch.sqrt(trace[mask1] + 1.0) * 2
    q[mask1, 3] = 0.25 * s
    q[mask1, 0] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s
    q[mask1, 1] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s
    q[mask1, 2] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s
    
    # Case 2: R[0,0] is max
    mask2 = (~mask1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    s = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2
    q[mask2, 3] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s
    q[mask2, 0] = 0.25 * s
    q[mask2, 1] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s
    q[mask2, 2] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s
    
    # Case 3: R[1,1] is max
    mask3 = (~mask1) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    s = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2
    q[mask3, 3] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s
    q[mask3, 0] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s
    q[mask3, 1] = 0.25 * s
    q[mask3, 2] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s
    
    # Case 4: R[2,2] is max
    mask4 = (~mask1) & (~mask2) & (~mask3)
    s = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2
    q[mask4, 3] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s
    q[mask4, 0] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s
    q[mask4, 1] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s
    q[mask4, 2] = 0.25 * s
    
    return q

