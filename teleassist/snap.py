"""Orientation snap assist: pull EE rotation toward GraspAssistMap grasp quats."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from teleassist.scan import GraspAssistEntry, GraspAssistMap

# World distance (m) to the grasp region where snap starts / is fully on.
SNAP_RADIUS = 0.20
FULL_SNAP_RADIUS = 0.08
# Max orientation correction per teleop step (rad), before action scale.
MAX_ROT_DELTA = 0.12
# Diff-IK arm action scale in the Franka IK-Rel cfg (raw action is multiplied by this).
DEFAULT_ACTION_SCALE = 0.5


@dataclass
class SnapDecision:
    object_id: str | None
    distance: float
    weight: float
    rot_delta: np.ndarray  # axis-angle in world/EE frame, ready for action[3:6]


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / norm


def _quat_conjugate_xyzw(q: np.ndarray) -> np.ndarray:
    q = _quat_normalize(q)
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def _quat_mul_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product for (x, y, z, w) quaternions."""
    x1, y1, z1, w1 = _quat_normalize(q1)
    x2, y2, z2, w2 = _quat_normalize(q2)
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def quat_error_axis_angle_xyzw(current_xyzw: np.ndarray, target_xyzw: np.ndarray) -> np.ndarray:
    """Axis-angle that rotates ``current`` into ``target`` (both xyzw)."""
    current = _quat_normalize(current_xyzw)
    target = _quat_normalize(target_xyzw)
    # Prefer the shorter arc.
    if float(np.dot(current, target)) < 0.0:
        target = -target
    # q_err = q_target * q_current^{-1}
    err = _quat_mul_xyzw(target, _quat_conjugate_xyzw(current))
    err = _quat_normalize(err)
    # xyzw → axis-angle
    xyz = err[:3]
    w = float(np.clip(err[3], -1.0, 1.0))
    sin_half = float(np.linalg.norm(xyz))
    if sin_half < 1e-8:
        return np.zeros(3, dtype=np.float64)
    axis = xyz / sin_half
    angle = 2.0 * float(np.arctan2(sin_half, w))
    if angle > np.pi:
        angle -= 2.0 * np.pi
    return axis * angle


def snap_weight(
    distance: float,
    snap_radius: float = SNAP_RADIUS,
    full_snap_radius: float = FULL_SNAP_RADIUS,
) -> float:
    if distance <= full_snap_radius:
        return 1.0
    if distance >= snap_radius:
        return 0.0
    return float(1.0 - (distance - full_snap_radius) / (snap_radius - full_snap_radius))


def _grasp_region_point(
    entry: GraspAssistEntry,
    live_centers: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """World point used for snap distance: OOAL grasp region, not object center."""
    scan_center = np.asarray(entry.center_world, dtype=np.float64).reshape(3)
    pts = np.asarray(entry.grasp_points_world, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] > 0:
        region = pts.mean(axis=0)
    else:
        region = scan_center.copy()
    if live_centers and entry.object_id in live_centers:
        live = np.asarray(live_centers[entry.object_id], dtype=np.float64).reshape(3)
        # Keep grasp offset relative to object while tracking the live body pose.
        region = region + (live - scan_center)
    return region


def nearest_entry(
    ee_pos_w: np.ndarray,
    assist_map: GraspAssistMap,
    live_centers: dict[str, np.ndarray] | None = None,
) -> tuple[GraspAssistEntry | None, float]:
    if not assist_map.entries:
        return None, float("inf")
    ee = np.asarray(ee_pos_w, dtype=np.float64).reshape(3)
    best: GraspAssistEntry | None = None
    best_dist = float("inf")
    for entry in assist_map.entries.values():
        target = _grasp_region_point(entry, live_centers=live_centers)
        dist = float(np.linalg.norm(ee - target))
        if dist < best_dist:
            best = entry
            best_dist = dist
    return best, best_dist


def compute_snap(
    ee_pos_w: np.ndarray,
    ee_quat_xyzw: np.ndarray,
    assist_map: GraspAssistMap,
    *,
    live_centers: dict[str, np.ndarray] | None = None,
    snap_radius: float = SNAP_RADIUS,
    full_snap_radius: float = FULL_SNAP_RADIUS,
    max_rot_delta: float = MAX_ROT_DELTA,
    action_scale: float = DEFAULT_ACTION_SCALE,
) -> SnapDecision:
    """Return a rotational action delta that pulls EE toward the nearest grasp quat."""
    entry, distance = nearest_entry(ee_pos_w, assist_map, live_centers=live_centers)
    weight = snap_weight(distance, snap_radius, full_snap_radius)
    if entry is None or weight <= 0.0:
        return SnapDecision(None, distance, 0.0, np.zeros(3, dtype=np.float64))

    target_xyzw = wxyz_to_xyzw(entry.grasp_quat_world)
    error_aa = quat_error_axis_angle_xyzw(ee_quat_xyzw, target_xyzw)
    # Raw teleop action is scaled by ``action_scale`` inside Diff-IK; invert that.
    scale = max(float(action_scale), 1e-6)
    # Cap per-step correction in EE space, then map back to action units.
    err_norm = float(np.linalg.norm(error_aa))
    if err_norm > max_rot_delta:
        error_aa = error_aa * (max_rot_delta / err_norm)
    desired_action_rot = error_aa / scale
    rot_delta = weight * desired_action_rot
    return SnapDecision(entry.object_id, distance, weight, rot_delta.astype(np.float64))


def apply_snap_to_action(
    action: np.ndarray | object,
    decision: SnapDecision,
) -> np.ndarray:
    """Blend snap rotation into a 7D teleop action (keeps translation + gripper)."""
    out = np.asarray(action, dtype=np.float64).reshape(-1).copy()
    if out.size < 6 or decision.weight <= 0.0:
        return out
    # Keep user translation; replace rotation with snap (user can still add rot keys —
    # we layer snap on top so assist always pulls when near).
    out[3:6] = out[3:6] + decision.rot_delta
    return out
