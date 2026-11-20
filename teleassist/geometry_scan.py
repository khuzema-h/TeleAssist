"""Scan-time geometry: depth unproject, object pose, grasp orientation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEPTH_RANGE = (0.05, 3.0)
STANDOFF_Z = 0.10
TOP_DOWN_WXYZ = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
SIDE_GRASP_TILT_DEG = 15.0


@dataclass
class ApproachGoal:
    position: np.ndarray
    quaternion: np.ndarray
    center_world: np.ndarray
    top_z: float


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    rot = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(rot))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rot[2, 1] - rot[1, 2]) / scale
        y = (rot[0, 2] - rot[2, 0]) / scale
        z = (rot[1, 0] - rot[0, 1]) / scale
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        scale = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        w = (rot[2, 1] - rot[1, 2]) / scale
        x = 0.25 * scale
        y = (rot[0, 1] + rot[1, 0]) / scale
        z = (rot[0, 2] + rot[2, 0]) / scale
    elif rot[1, 1] > rot[2, 2]:
        scale = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        w = (rot[0, 2] - rot[2, 0]) / scale
        x = (rot[0, 1] + rot[1, 0]) / scale
        y = 0.25 * scale
        z = (rot[1, 2] + rot[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        w = (rot[1, 0] - rot[0, 1]) / scale
        x = (rot[0, 2] + rot[2, 0]) / scale
        y = (rot[1, 2] + rot[2, 1]) / scale
        z = 0.25 * scale
    quat = np.array([w, x, y, z], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm > 1e-12:
        quat /= norm
    return quat


def adaptive_approach_pose(
    target_point: np.ndarray,
    quaternion_world: np.ndarray,
    standoff: float,
    finger_reach: float = 0.11,
) -> np.ndarray:
    rot = quat_to_matrix(quaternion_world)
    z_hand = rot[:, 2]
    target = np.asarray(target_point, dtype=np.float64).reshape(3)
    return target - (float(finger_reach) + float(standoff)) * z_hand


def adaptive_grasp_orientation(
    points: np.ndarray,
    object_center: np.ndarray | None = None,
    grasp_style: str = "auto",
    tilt_angle_deg: float = 35.0,
) -> np.ndarray:
    """Compute adaptive grasp quaternion [w, x, y, z] in world coordinates."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    finite = pts[np.isfinite(pts).all(axis=1)]
    if finite.shape[0] < 3:
        return TOP_DOWN_WXYZ.copy()

    mean_pt = np.mean(finite, axis=0)
    style = grasp_style.lower()
    if style == "auto":
        if object_center is not None:
            radial_dist = float(
                np.linalg.norm(mean_pt[:2] - np.asarray(object_center, dtype=np.float64)[:2])
            )
            style = "tilted" if radial_dist > 0.02 else "top_down"
        else:
            style = "top_down"

    if style in {"tilted", "side"}:
        if object_center is not None:
            radial_vec = mean_pt[:2] - np.asarray(object_center, dtype=np.float64)[:2]
        else:
            radial_vec = mean_pt[:2]
        rad_norm = float(np.linalg.norm(radial_vec))
        if rad_norm < 1e-4:
            u_rad = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            u_rad = np.array(
                [radial_vec[0] / rad_norm, radial_vec[1] / rad_norm, 0.0], dtype=np.float64
            )
        u_tan = np.array([-u_rad[1], u_rad[0], 0.0], dtype=np.float64)
        alpha = np.radians(SIDE_GRASP_TILT_DEG if style == "side" else np.clip(tilt_angle_deg, 10.0, 80.0))
        cos_a = np.cos(alpha)
        sin_a = np.sin(alpha)
        z_hand = -cos_a * u_rad - sin_a * np.array([0.0, 0.0, 1.0], dtype=np.float64)
        z_hand /= np.linalg.norm(z_hand)
        y_hand = u_tan
        x_hand = np.cross(y_hand, z_hand)
        x_hand /= np.linalg.norm(x_hand)
        if x_hand[2] < 0:
            y_hand = -y_hand
            x_hand = np.cross(y_hand, z_hand)
        return matrix_to_quat_wxyz(np.stack([x_hand, y_hand, z_hand], axis=1))

    z_hand = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    centered = finite - mean_pt
    cov = centered.T @ centered / finite.shape[0]
    cov_xy = cov[:2, :2]
    eig_xy, vec_xy = np.linalg.eigh(cov_xy)
    if eig_xy[1] > 2.0 * max(eig_xy[0], 1e-6):
        major = np.array([vec_xy[0, 1], vec_xy[1, 1], 0.0], dtype=np.float64)
        major /= np.linalg.norm(major)
        y_hand = np.array([-major[1], major[0], 0.0], dtype=np.float64)
        x_hand = np.cross(y_hand, z_hand)
        x_hand /= np.linalg.norm(x_hand)
        return matrix_to_quat_wxyz(np.stack([x_hand, y_hand, z_hand], axis=1))
    return TOP_DOWN_WXYZ.copy()


def tracked_surface_top(points: np.ndarray) -> float:
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("Tracked points must be a non-empty Nx3 array")
    z_values = points[:, 2]
    z_values = z_values[np.isfinite(z_values)]
    if z_values.size == 0:
        raise ValueError("Tracked points contain no finite depths")
    median = float(np.median(z_values))
    mad = float(np.median(np.abs(z_values - median)))
    upper_bound = median + max(3.0 * 1.4826 * mad, 0.015)
    inliers = z_values[z_values <= upper_bound]
    return float(np.max(inliers))


def pose_from_depth_points(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 4:
        raise ValueError("Depth points must be an Nx3 array with at least 4 points")
    finite = points[np.isfinite(points).all(axis=1)]
    if finite.shape[0] < 4:
        raise ValueError("Depth points contain fewer than 4 finite samples")
    top_z = tracked_surface_top(finite)
    low = np.percentile(finite, 5, axis=0)
    high = np.percentile(finite, 95, axis=0)
    dims = np.maximum(high - low, 0.02).astype(np.float64)
    xy_size = float(np.median(dims[:2]))
    if dims[2] < 0.5 * xy_size:
        dims[2] = xy_size
    center = np.median(finite, axis=0).astype(np.float64)
    center[2] = top_z - float(dims[2]) / 2.0
    return center, top_z, dims


def sample_mask_points(mask: np.ndarray, count: int = 16) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise RuntimeError("Empty mask")
    side = int(np.ceil(np.sqrt(count)))
    y_samples = np.linspace(ys.min(), ys.max(), side)
    x_samples = np.linspace(xs.min(), xs.max(), side)
    chosen: list[tuple[float, float]] = []
    for y in y_samples:
        for x in x_samples:
            nearest = int(np.argmin((ys - y) ** 2 + (xs - x) ** 2))
            chosen.append((float(xs[nearest]), float(ys[nearest])))
            if len(chosen) >= count:
                break
        if len(chosen) >= count:
            break
    unique = list(dict.fromkeys(chosen))
    if len(unique) < count:
        order = np.linspace(0, len(xs) - 1, count, dtype=int)
        for index in order:
            unique.append((float(xs[index]), float(ys[index])))
            if len(unique) >= count:
                break
    return np.asarray(unique[:count], dtype=np.float32)


def quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    return quat_to_matrix(np.asarray(quat_wxyz, dtype=np.float64).reshape(4))


def unproject_pixels_lab(
    pixels_xy: np.ndarray,
    depth: np.ndarray,
    intrinsic_3x3: np.ndarray,
    cam_pos_w: np.ndarray,
    cam_quat_w_ros: np.ndarray,
) -> np.ndarray:
    """Unproject image pixels with ROS-convention camera pose into world XYZ.

    ROS camera frame: +X right, +Y down, +Z forward (along the optical axis).
    """
    height, width = depth.shape[:2]
    k = np.asarray(intrinsic_3x3, dtype=np.float64).reshape(3, 3)
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    rot = quat_wxyz_to_matrix(cam_quat_w_ros)
    origin = np.asarray(cam_pos_w, dtype=np.float64).reshape(3)
    points = []
    for u, v in np.asarray(pixels_xy, dtype=np.float64).reshape(-1, 2):
        ui = int(np.clip(round(float(u)), 0, width - 1))
        vi = int(np.clip(round(float(v)), 0, height - 1))
        z = float(depth[vi, ui])
        if not np.isfinite(z) or z < DEPTH_RANGE[0] or z > DEPTH_RANGE[1]:
            continue
        local = np.array([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=np.float64)
        points.append(origin + rot @ local)
    if not points:
        return np.zeros((0, 3), dtype=np.float64)
    return np.stack(points, axis=0)


def mask_depth_points_lab(
    mask: np.ndarray,
    depth: np.ndarray,
    intrinsic_3x3: np.ndarray,
    cam_pos_w: np.ndarray,
    cam_quat_w_ros: np.ndarray,
    count: int = 64,
) -> np.ndarray:
    if mask.shape != depth.shape[:2]:
        raise ValueError(f"Mask shape {mask.shape} does not match depth image {depth.shape[:2]}")
    pixels = sample_mask_points(mask, count=count)
    return unproject_pixels_lab(pixels, depth, intrinsic_3x3, cam_pos_w, cam_quat_w_ros)
