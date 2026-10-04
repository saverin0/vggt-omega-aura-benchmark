"""Python reference geometry. NumPy only. The C++ core in cpp/ mirrors this file.

CONVENTIONS, stated once and used everywhere:

- A transform is a 4x4 matrix named destination_from_source. It maps a point
  written in source coordinates to the same point in destination coordinates:
      p_destination = T_destination_from_source @ p_source
  Chaining reads right to left:  a_from_c = a_from_b @ b_from_c.

- AURA's base_link is the vehicle frame: X forward, Y left, Z up (ROS).

- A camera frame, in AURA and in VGGT-Omega alike, is the OPTICAL frame:
  X right, Y down, Z forward (OpenCV). Depth means the Z coordinate, not the
  distance along the ray. Checked on real data: front_medium's Z axis points
  along base_link +X.

- VGGT-Omega extrinsics are camera_from_world, 3x4, where "world" is the
  first input camera. So the first extrinsic is the identity, and a camera's
  position in the world is  -R.T @ t,  NOT t.

- A pixel (u, v) has u to the right and v down, origin at the top-left.
"""

from __future__ import annotations

import numpy as np


def to_4x4(matrix) -> np.ndarray:
    """Accept 3x4 or 4x4 (single or stacked) and return 4x4."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[-2:] == (4, 4):
        return matrix
    if matrix.shape[-2:] != (3, 4):
        raise ValueError(f"expected (..., 3, 4) or (..., 4, 4), got {matrix.shape}")
    bottom = np.zeros(matrix.shape[:-2] + (1, 4))
    bottom[..., 0, 3] = 1.0
    return np.concatenate([matrix, bottom], axis=-2)


def invert_se3(matrix) -> np.ndarray:
    """Invert rigid transforms in closed form: [R t] -> [R.T  -R.T t]."""
    matrix = to_4x4(matrix)
    rotation = matrix[..., :3, :3]
    translation = matrix[..., :3, 3]
    inverse = np.zeros_like(matrix)
    rotation_t = np.swapaxes(rotation, -1, -2)
    inverse[..., :3, :3] = rotation_t
    inverse[..., :3, 3] = -np.einsum("...ij,...j->...i", rotation_t, translation)
    inverse[..., 3, 3] = 1.0
    return inverse


def camera_centers(camera_from_world) -> np.ndarray:
    """Camera positions in world coordinates, shape (N, 3), from camera_from_world matrices."""
    return invert_se3(camera_from_world)[..., :3, 3]


def relative_to_first(world_from_camera) -> np.ndarray:
    """Re-express camera poses in the frame of the first camera.

    Input:  world_from_camera_i, shape (N, 4, 4), any world (for AURA: odom).
    Output: camera0_from_camera_i, shape (N, 4, 4). Entry 0 is the identity.
    """
    world_from_camera = to_4x4(world_from_camera)
    return invert_se3(world_from_camera[0]) @ world_from_camera


def scale_intrinsics(fx, fy, cx, cy, source_wh, target_wh) -> tuple[float, float, float, float]:
    """Intrinsics of a resized image. x and y scale independently.

    Uses the pixel-centre convention of PIL and OpenCV resizing, where pixel i
    covers [i, i+1) and its centre is i + 0.5:
        u_target = (u_source + 0.5) * s - 0.5
    Focal lengths scale by s. The principal point follows the formula above.
    For a 3x downscale the half-pixel term moves cx by a third of a pixel,
    which is small but not zero, so it is kept.
    """
    sx = target_wh[0] / source_wh[0]
    sy = target_wh[1] / source_wh[1]
    return (fx * sx, fy * sy, (cx + 0.5) * sx - 0.5, (cy + 0.5) * sy - 0.5)


def fov_degrees(focal_px: float, size_px: float) -> float:
    """Full field of view for a focal length and image extent in the same pixel units."""
    return float(np.degrees(2.0 * np.arctan((size_px / 2.0) / focal_px)))


def quat_xyzw_to_matrix(quaternion) -> np.ndarray:
    """Rotation matrix from a quaternion given as (x, y, z, w), the order AURA uses.

    Many libraries use (w, x, y, z). Mixing the two orders gives a valid but
    WRONG rotation with no error, so the order is part of the function's name.
    The quaternion is normalised first.
    """
    x, y, z, w = (np.asarray(quaternion, dtype=np.float64) / np.linalg.norm(quaternion)).tolist()
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rotation_angle_deg(rotation) -> np.ndarray:
    """Angle of a rotation matrix (or a stack of them), in degrees, from its trace."""
    rotation = np.asarray(rotation, dtype=np.float64)
    cosine = (np.trace(rotation, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def angle_between_deg(a, b) -> np.ndarray:
    """Angle between vectors, row by row, in degrees. NaN where either vector is (near) zero."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    norms = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = np.where(norms > 1e-12, np.sum(a * b, axis=-1) / norms, np.nan)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def umeyama_similarity(source, target, with_scale: bool = True):
    """Least-squares similarity transform taking `source` points onto `target` points.

    Returns (scale, rotation, translation) with  target ~= scale * rotation @ source + translation.
    Umeyama, IEEE PAMI 1991. The determinant check keeps the result a proper
    rotation and never a reflection.
    """
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError("need two (N, 3) arrays with N >= 3")
    mean_s, mean_t = source.mean(axis=0), target.mean(axis=0)
    cs, ct = source - mean_s, target - mean_t
    covariance = ct.T @ cs / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vt
    variance = (cs ** 2).sum() / len(source)
    scale = float((singular * sign).sum() / variance) if with_scale and variance > 1e-18 else 1.0
    translation = mean_t - scale * rotation @ mean_s
    return scale, rotation, translation


def points_in_box(points, center, size_lwh, rotation_xyzw, margin: float = 0.0) -> np.ndarray:
    """Which points lie inside an oriented 3D box. Points and box must share one frame.

    AURA boxes: centre origin, FULL sizes (length along box X, width along Y,
    height along Z), and a quaternion (x, y, z, w) that rotates box axes into
    the frame the box is written in.
    """
    rotation = quat_xyzw_to_matrix(rotation_xyzw)
    local = (np.asarray(points, dtype=np.float64) - np.asarray(center, dtype=np.float64)) @ rotation   # = R.T @ (p - c)
    half = np.asarray(size_lwh, dtype=np.float64) / 2.0 + margin
    return np.all(np.abs(local) <= half, axis=1)
