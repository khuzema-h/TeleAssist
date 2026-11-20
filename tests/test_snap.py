"""Unit tests for orientation snap assist."""

from __future__ import annotations

import numpy as np

from teleassist.scan import GraspAssistEntry, GraspAssistMap
from teleassist.snap import (
    apply_snap_to_action,
    compute_snap,
    nearest_entry,
    quat_error_axis_angle_xyzw,
    snap_weight,
    wxyz_to_xyzw,
)


def _entry(object_id: str, center, quat_wxyz) -> GraspAssistEntry:
    return GraspAssistEntry(
        object_id=object_id,
        center_world=np.asarray(center, dtype=np.float64),
        dims=np.array([0.05, 0.05, 0.05], dtype=np.float64),
        top_z=float(center[2]) + 0.025,
        box=(0.0, 0.0, 10.0, 10.0),
        affordance="hold",
        grasp_points_world=np.asarray(center, dtype=np.float64).reshape(1, 3),
        grasp_quat_world=np.asarray(quat_wxyz, dtype=np.float64),
        grasp_style="side",
    )


def test_wxyz_to_xyzw():
    assert np.allclose(wxyz_to_xyzw([1, 0, 0, 0]), [0, 0, 0, 1])
    assert np.allclose(wxyz_to_xyzw([0.7071, 0, 0, 0.7071]), [0, 0, 0.7071, 0.7071], atol=1e-3)


def test_snap_weight_falloff():
    assert snap_weight(0.05) == 1.0
    assert snap_weight(0.25) == 0.0
    mid = snap_weight(0.14)
    assert 0.0 < mid < 1.0


def test_nearest_entry():
    assist = GraspAssistMap(
        entries={
            "mug": _entry("mug", [0.5, 0.0, 0.1], [0, 1, 0, 0]),
            "other": _entry("other", [1.0, 0.0, 0.1], [0, 1, 0, 0]),
        }
    )
    entry, dist = nearest_entry([0.52, 0.0, 0.1], assist)
    assert entry is not None and entry.object_id == "mug"
    assert dist < 0.05


def test_nearest_uses_grasp_region_not_center():
    # Grasp points offset from object center → distance measured to grasp region.
    entry = _entry("mug", [0.5, 0.0, 0.1], [0, 1, 0, 0])
    entry.grasp_points_world = np.array([[0.5, 0.08, 0.12], [0.5, 0.10, 0.12]], dtype=np.float64)
    assist = GraspAssistMap(entries={"mug": entry})
    ee_near_grasp = np.array([0.5, 0.09, 0.12])
    ee_near_center = np.array([0.5, 0.0, 0.1])
    _, d_grasp = nearest_entry(ee_near_grasp, assist)
    _, d_center = nearest_entry(ee_near_center, assist)
    assert d_grasp < 0.03
    assert d_center > 0.08


def test_quat_error_zero_and_nonzero():
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    assert np.allclose(quat_error_axis_angle_xyzw(identity, identity), 0.0, atol=1e-6)
    # 90° about Z: xyzw = (0, 0, sin45, cos45)
    target = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
    err = quat_error_axis_angle_xyzw(identity, target)
    assert abs(float(np.linalg.norm(err)) - np.pi / 2) < 1e-3
    assert abs(err[2]) > abs(err[0]) and abs(err[2]) > abs(err[1])


def test_compute_snap_and_apply():
    assist = GraspAssistMap(
        entries={"mug": _entry("mug", [0.5, 0.0, 0.05], [0.0, 1.0, 0.0, 0.0])}  # 180° about X in wxyz
    )
    # EE at object, identity orientation → should snap hard.
    decision = compute_snap(
        ee_pos_w=np.array([0.5, 0.0, 0.05]),
        ee_quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        assist_map=assist,
        action_scale=0.5,
    )
    assert decision.object_id == "mug"
    assert decision.weight == 1.0
    assert float(np.linalg.norm(decision.rot_delta)) > 0.0

    action = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    out = apply_snap_to_action(action, decision)
    assert np.allclose(out[:3], action[:3])
    assert not np.allclose(out[3:6], 0.0)
    assert out[6] == 1.0


def test_compute_snap_out_of_range():
    assist = GraspAssistMap(entries={"mug": _entry("mug", [0.5, 0.0, 0.05], [0, 1, 0, 0])})
    decision = compute_snap(
        ee_pos_w=np.array([1.5, 0.0, 0.05]),
        ee_quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        assist_map=assist,
    )
    assert decision.weight == 0.0
    assert np.allclose(decision.rot_delta, 0.0)
