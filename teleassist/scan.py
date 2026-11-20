"""Build a GraspAssistMap from one RGB-D environment-camera frame."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from teleassist.affordance import (
    affordance_status,
    region_points_from_heatmap,
)
from teleassist.camera_lab import LabCameraFrame
from teleassist.catalog import TABLE_OBJECT_SPECS, object_id, object_phrase
from teleassist.geometry_scan import (
    adaptive_grasp_orientation,
    mask_depth_points_lab,
    pose_from_depth_points,
    sample_mask_points,
    unproject_pixels_lab,
)
from teleassist.grounding import ObjectGrounder
from teleassist.sam import SamSegmenter


@dataclass
class GraspAssistEntry:
    object_id: str
    center_world: np.ndarray
    dims: np.ndarray
    top_z: float
    box: tuple[float, float, float, float]
    affordance: str | None
    grasp_points_world: np.ndarray
    grasp_quat_world: np.ndarray
    grasp_style: str


@dataclass
class GraspAssistMap:
    entries: dict[str, GraspAssistEntry] = field(default_factory=dict)
    # Full-frame OOAL affordance heatmap (H, W), max-composited over objects.
    heatmap: np.ndarray | None = None
    grasp_pixels_xy: np.ndarray | None = None

    def summary_lines(self) -> list[str]:
        lines = []
        for entry in self.entries.values():
            lines.append(
                f"{entry.object_id}: center={entry.center_world.round(3).tolist()} "
                f"affordance={entry.affordance} style={entry.grasp_style} "
                f"quat={entry.grasp_quat_world.round(3).tolist()} "
                f"points={len(entry.grasp_points_world)}"
            )
        return lines


def _paste_heatmap(
    canvas: np.ndarray,
    crop: np.ndarray,
    y0: int,
    x0: int,
) -> None:
    y1 = min(canvas.shape[0], y0 + crop.shape[0])
    x1 = min(canvas.shape[1], x0 + crop.shape[1])
    if y1 <= y0 or x1 <= x0:
        return
    region = canvas[y0:y1, x0:x1]
    patch = crop[: y1 - y0, : x1 - x0]
    np.maximum(region, patch, out=region)


class SceneScanner:
    """OWL-ViT + SAM + OOAL scan pipeline for TeleAssist."""

    def __init__(self) -> None:
        self.grounder = ObjectGrounder()
        self.segmenter = SamSegmenter()

    def scan_frame(self, frame: LabCameraFrame, say=print) -> GraspAssistMap:
        from teleassist.affordance import affordance_heatmap

        rgb = frame.rgb
        depth = frame.depth
        assist = GraspAssistMap()
        if rgb.size == 0 or depth.size == 0:
            say("table camera has no RGB-D frame")
            return assist

        height, width = rgb.shape[:2]
        heatmap = np.zeros((height, width), dtype=np.float32)
        grasp_pixels: list[np.ndarray] = []

        requests = [(object_id(spec), spec, object_phrase(spec)) for spec in TABLE_OBJECT_SPECS]
        # Scene-wide priors for catalog affordances, then per-object crops.
        for affordance_name in {str(spec.get("affordance") or "pick_up") for _, spec, _ in requests}:
            scene_heat = affordance_heatmap(rgb, affordance_name)
            if scene_heat is not None:
                heatmap = np.maximum(heatmap, scene_heat.astype(np.float32))

        located = self.grounder.ground_all(rgb, [phrase for _, _, phrase in requests])

        for obj_id, spec, phrase in requests:
            found = located.get(phrase)
            if found is None:
                say(f"scan could not ground {obj_id} ({phrase})")
                continue
            box, score = found
            try:
                mask = self.segmenter.segment_region(rgb, box=box)
            except RuntimeError as error:
                say(f"scan could not segment {obj_id}: {error}")
                continue

            cloud = mask_depth_points_lab(
                mask, depth, frame.intrinsic, frame.pos_w, frame.quat_w_ros, count=64
            )
            if cloud.shape[0] < 4:
                say(f"scan depth has only {cloud.shape[0]} points on {obj_id}")
                continue
            center, top_z, dims = pose_from_depth_points(cloud)

            affordance = str(spec.get("affordance") or "hold")
            grasp_style = str(spec.get("grasp_style") or "auto")

            ys, xs = np.where(mask)
            pad = 8
            y0 = max(0, int(ys.min()) - pad)
            y1 = min(height, int(ys.max()) + pad + 1)
            x0 = max(0, int(xs.min()) - pad)
            x1 = min(width, int(xs.max()) + pad + 1)
            crop_heat = affordance_heatmap(rgb[y0:y1, x0:x1], affordance)
            if crop_heat is not None:
                _paste_heatmap(heatmap, crop_heat.astype(np.float32), y0, x0)
                region_xy = region_points_from_heatmap(
                    mask[y0:y1, x0:x1], crop_heat, count=16, origin=(x0, y0)
                )
            else:
                region_xy = None
            if region_xy is None:
                say(f"scan {obj_id}: no OOAL region for {affordance}; using full mask samples")
                affordance_used = None
                grasp_xy = sample_mask_points(mask, count=16)
            else:
                affordance_used = affordance
                grasp_xy = region_xy
                say(f"scan {obj_id}: OOAL {affordance} seeded {len(grasp_xy)} grasp pixels")
            grasp_pixels.append(np.asarray(grasp_xy, dtype=np.float64).reshape(-1, 2))

            grasp_points = unproject_pixels_lab(
                grasp_xy, depth, frame.intrinsic, frame.pos_w, frame.quat_w_ros
            )
            if grasp_points.shape[0] < 3:
                grasp_points = cloud
            quat = adaptive_grasp_orientation(
                grasp_points, object_center=center, grasp_style=grasp_style
            )
            assist.entries[obj_id] = GraspAssistEntry(
                object_id=obj_id,
                center_world=center,
                dims=dims,
                top_z=top_z,
                box=tuple(float(v) for v in box),
                affordance=affordance_used,
                grasp_points_world=grasp_points,
                grasp_quat_world=quat,
                grasp_style=grasp_style,
            )
            say(
                f"scan {obj_id} score={score:.2f} center={center.round(3).tolist()} "
                f"top_z={top_z:.3f}"
            )

        if float(np.nanmax(heatmap)) > 0:
            assist.heatmap = heatmap
        if grasp_pixels:
            assist.grasp_pixels_xy = np.concatenate(grasp_pixels, axis=0)

        self.segmenter.release_sam()
        say(f"OOAL status: {affordance_status()}")
        if assist.heatmap is not None:
            say(f"affordance heatmap ready peak={float(np.nanmax(assist.heatmap)):.3f}")
        else:
            say("affordance heatmap empty (OOAL unavailable)")
        return assist
