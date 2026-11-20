"""Read RGB-D and camera pose from an Isaac Lab Camera sensor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LabCameraFrame:
    rgb: np.ndarray
    depth: np.ndarray
    intrinsic: np.ndarray
    pos_w: np.ndarray
    quat_w_ros: np.ndarray


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "torch"):
        tensor = value.torch
        return tensor.detach().cpu().numpy()
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def read_table_camera(env, sensor_name: str = "table_cam", env_index: int = 0) -> LabCameraFrame:
    """Pull env-0 RGB-D + intrinsics + ROS pose from a Lab camera sensor."""
    sensors = getattr(env.scene, "sensors", None)
    if sensors is None:
        camera = env.scene[sensor_name]
    else:
        camera = sensors[sensor_name]

    output = camera.data.output
    rgb = _to_numpy(output["rgb"])[env_index]
    depth = _to_numpy(output["distance_to_image_plane"])[env_index]
    if depth.ndim == 3:
        depth = depth[..., 0]

    rgb = np.ascontiguousarray(rgb[..., :3])
    if rgb.dtype != np.uint8:
        if np.nanmax(rgb) <= 1.0:
            rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        else:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    intrinsic = _to_numpy(camera.data.intrinsic_matrices)[env_index].astype(np.float64)
    pos_w = _to_numpy(camera.data.pos_w)[env_index].astype(np.float64)
    quat_w_ros = _to_numpy(camera.data.quat_w_ros)[env_index].astype(np.float64)
    return LabCameraFrame(
        rgb=rgb,
        depth=depth.astype(np.float64),
        intrinsic=intrinsic.reshape(3, 3),
        pos_w=pos_w.reshape(3),
        quat_w_ros=quat_w_ros.reshape(4),
    )
