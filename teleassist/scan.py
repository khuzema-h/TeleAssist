"""Build a GraspAssistMap from one RGB-D environment-camera frame."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from teleassist.affordance import affordance_status, grasp_region_points
from teleassist.camera_lab import LabCameraFrame
from teleassist.catalog import STACK_CUBE_SPECS, object_id, object_phrase
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


class SceneScanner:
    """OWL-ViT + SAM + OOAL scan pipeline for TeleAssist."""

    def __init__(self) -> None:
        self.grounder = ObjectGrounder()
        self.segmenter = SamSegmenter()

    def scan_frame(self, frame: LabCameraFrame, say=print) -> GraspAssistMap:
        rgb = frame.rgb
        depth = frame.depth
        assist = GraspAssistMap()
        if rgb.size == 0 or depth.size == 0:
            say("table camera has no RGB-D frame")
            return assist

        requests = [(object_id(spec), spec, object_phrase(spec)) for spec in STACK_CUBE_SPECS]
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

            affordance = str(spec.get("affordance") or "pick_up")
            grasp_style = str(spec.get("grasp_style") or "auto")
            region_xy = grasp_region_points(rgb, mask, affordance)
            if region_xy is None:
                say(f"scan {obj_id}: no OOAL region for {affordance}; using full mask samples")
                affordance_used = None
                grasp_xy = sample_mask_points(mask, count=16)
            else:
                affordance_used = affordance
                grasp_xy = region_xy
                say(f"scan {obj_id}: OOAL {affordance} seeded {len(grasp_xy)} grasp pixels")

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

        self.segmenter.release_sam()
        say(f"OOAL status: {affordance_status()}")
        return assist
