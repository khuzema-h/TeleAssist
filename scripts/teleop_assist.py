#!/usr/bin/env python3
"""Franka SE(3) teleop with GraspAssistMap scanning (start + P rescan)."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

# Allow running without an editable install when launched via isaaclab.sh -p.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import setup_preset_cli

parser = argparse.ArgumentParser(description="TeleAssist Franka SE(3) teleop with scene scan.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--teleop_device", type=str, default="keyboard")
parser.add_argument("--task", type=str, default="TeleAssist-Mug-Plate-Franka-IK-Rel-RGBD")
parser.add_argument("--sensitivity", type=float, default=1.0)
parser.add_argument(
    "--disable_external_cameras",
    action="store_true",
    default=False,
    help="Strip cameras (disables scanning).",
)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
args_cli.enable_cameras = not args_cli.disable_external_cameras
args_cli.require_kit = True
sys.argv = [sys.argv[0]] + hydra_args

import gymnasium as gym
import torch
from isaaclab_physx.renderers import IsaacRtxRendererGlobalSettingsCfg

from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs import ManagerBasedRLEnvCfg

import teleassist.envs  # noqa: F401 — register TeleAssist gym ids
from isaaclab_tasks.utils import resolve_task_config
from teleassist.camera_lab import read_table_camera, read_wrist_camera
from teleassist.scan import GraspAssistMap, SceneScanner
from teleassist.snap import DEFAULT_ACTION_SCALE, SnapMemory, apply_snap_to_action, compute_snap
from teleassist.view import TableCameraWindows

logger = logging.getLogger(__name__)


@dataclass
class AssistState:
    map: GraspAssistMap = field(default_factory=GraspAssistMap)
    should_reset: bool = False
    should_rescan: bool = False
    teleoperation_active: bool = True
    snap_enabled: bool = True
    last_rgb: object | None = None
    last_wrist_rgb: object | None = None
    last_boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    last_heatmap: object | None = None
    last_grasp_pixels: object | None = None
    last_snap_object: str | None = None
    last_snap_mode: str = "idle"
    snap_memory: SnapMemory = field(default_factory=SnapMemory)


def _rtx_rendering_requested(args: argparse.Namespace) -> bool:
    visualizers = getattr(args, "visualizer", None) or []
    external_cameras = not getattr(args, "disable_external_cameras", False)
    return external_cameras or ("kit" in visualizers) or bool(getattr(args, "xr", False))


def _cam_prim(env, sensor_name: str, fallback: str) -> str:
    camera = env.scene.sensors[sensor_name] if hasattr(env.scene, "sensors") else env.scene[sensor_name]
    path = getattr(camera, "cfg", None)
    if path is not None and hasattr(path, "prim_path"):
        # Expand the first env namespace used by Lab.
        return str(path.prim_path).replace("{ENV_REGEX_NS}", "/World/envs/env_0")
    return fallback


def _table_cam_prim(env) -> str:
    return _cam_prim(env, "table_cam", "/World/envs/env_0/table_cam")


def _wrist_cam_prim(env) -> str:
    return _cam_prim(env, "wrist_cam", "/World/envs/env_0/Robot/panda_hand/wrist_cam")


def _log_wrist_cam_parent(env) -> None:
    """Confirm wrist_cam is under a moving panda_hand body (not a static sibling)."""
    try:
        from pxr import Usd

        import omni.usd

        stage = omni.usd.get_context().get_stage()
        path = _wrist_cam_prim(env)
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            # Regex prim_path may resolve to Geometry/.../panda_hand/wrist_cam.
            for candidate in stage.Traverse():
                if candidate.GetName() == "wrist_cam":
                    path = str(candidate.GetPath())
                    prim = candidate
                    break
        parent = str(prim.GetParent().GetPath()) if prim.IsValid() else "?"
        print(f"[TeleAssist] wrist_cam prim={path} parent={parent}", flush=True)
        if "panda_hand" not in parent:
            print(
                "[TeleAssist] WARNING: wrist_cam is not under panda_hand — "
                "image will not track the EE",
                flush=True,
            )
    except Exception as error:
        print(f"[TeleAssist] wrist_cam parent check failed: {error}", flush=True)


def _update_camera_ui(view: TableCameraWindows | None, state: AssistState) -> None:
    if view is None or state.last_rgb is None:
        return
    view.update(
        state.last_rgb,
        boxes=state.last_boxes or None,
        points_xy=state.last_grasp_pixels,
        heatmap=state.last_heatmap,
        wrist_rgb=state.last_wrist_rgb,
    )


def _ee_pose_world(env) -> tuple[object, object]:
    """Return (position xyz, quaternion xyzw) for the end-effector in world frame."""
    ee_frame = env.scene["ee_frame"]
    pos = ee_frame.data.target_pos_w.torch[0, 0, :].detach().cpu().numpy()
    quat_xyzw = ee_frame.data.target_quat_w.torch[0, 0, :].detach().cpu().numpy()
    return pos, quat_xyzw


def _live_object_centers(env, state: AssistState) -> dict[str, object]:
    """World positions from spawned rigid bodies when they match catalog ids."""
    centers: dict[str, object] = {}
    for object_id in state.map.entries:
        try:
            asset = env.scene[object_id]
        except Exception:
            continue
        try:
            centers[object_id] = asset.data.root_pos_w.torch[0].detach().cpu().numpy()
        except Exception:
            continue
    return centers


def _arm_action_scale(env) -> float:
    try:
        return float(env.cfg.actions.arm_action.scale)
    except Exception:
        return DEFAULT_ACTION_SCALE


def _maybe_apply_snap(env, action, state: AssistState, frame_i: int = 0):
    if not state.snap_enabled:
        if state.snap_memory.baseline_quat_xyzw is not None:
            state.snap_memory.clear()
        if frame_i % 60 == 0:
            print("[TeleAssist] snap disabled (press O to enable)", flush=True)
        return action
    if not state.map.entries:
        if frame_i % 60 == 0:
            print("[TeleAssist] snap idle: GraspAssistMap empty (press P to rescan)", flush=True)
        return action

    ee_pos, ee_quat = _ee_pose_world(env)
    live_centers = _live_object_centers(env, state)
    decision = compute_snap(
        ee_pos,
        ee_quat,
        state.map,
        memory=state.snap_memory,
        live_centers=live_centers,
        action_scale=_arm_action_scale(env),
    )
    if frame_i % 30 == 0:
        center_note = ""
        if decision.object_id and decision.object_id in live_centers:
            c = live_centers[decision.object_id]
            center_note = f" live=[{c[0]:.2f},{c[1]:.2f},{c[2]:.2f}]"
        print(
            f"[TeleAssist] snap check ee=[{ee_pos[0]:.2f},{ee_pos[1]:.2f},{ee_pos[2]:.2f}] "
            f"near={decision.object_id or '-'} dist={decision.distance:.3f} m "
            f"weight={decision.weight:.2f} mode={decision.mode}{center_note}",
            flush=True,
        )
    if decision.mode != state.last_snap_mode or decision.object_id != state.last_snap_object:
        if decision.mode == "snap" and decision.object_id is not None:
            print(
                f"[TeleAssist] snap → {decision.object_id} "
                f"(dist={decision.distance:.3f} m weight={decision.weight:.2f})",
                flush=True,
            )
        elif decision.mode == "restore":
            print(
                f"[TeleAssist] snap restore → pre-snap orientation "
                f"({decision.object_id or 'object'})",
                flush=True,
            )
        elif state.last_snap_mode in ("snap", "restore"):
            print(f"[TeleAssist] snap idle ({state.last_snap_object or '-'})", flush=True)
        state.last_snap_mode = decision.mode
        state.last_snap_object = decision.object_id if decision.mode != "idle" else None
    if decision.weight <= 0.0:
        return action
    snapped = apply_snap_to_action(action.detach().cpu().numpy(), decision)
    return torch.as_tensor(snapped, dtype=action.dtype, device=action.device)


def _run_scan(env, scanner: SceneScanner, state: AssistState, view: TableCameraWindows | None) -> None:
    for _ in range(3):
        env.sim.render()
    frame = read_table_camera(env)
    print(
        f"[TeleAssist] scanning table_cam rgb={frame.rgb.shape} depth={frame.depth.shape}",
        flush=True,
    )
    state.map = scanner.scan_frame(frame, say=lambda msg: print(f"[TeleAssist] {msg}", flush=True))
    state.last_rgb = frame.rgb
    state.last_boxes = [entry.box for entry in state.map.entries.values()]
    state.last_heatmap = state.map.heatmap
    state.last_grasp_pixels = state.map.grasp_pixels_xy
    lines = state.map.summary_lines()
    if not lines:
        print("[TeleAssist] GraspAssistMap is empty", flush=True)
    else:
        print(f"[TeleAssist] GraspAssistMap ({len(lines)} objects):", flush=True)
        for line in lines:
            print(f"  {line}", flush=True)
    _update_camera_ui(view, state)


def main() -> None:
    # Resolve through Isaac Lab presets so Physics USD variants and backends are concrete.
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = args_cli.num_envs
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise ValueError(f"Expected ManagerBasedRLEnvCfg, got {type(env_cfg).__name__}")
    env_cfg.terminations.time_out = None
    if not hasattr(env_cfg.scene, "table_cam"):
        raise RuntimeError("TeleAssist RGB-D task is missing scene.table_cam")
    if not hasattr(env_cfg.scene, "wrist_cam"):
        raise RuntimeError("TeleAssist RGB-D task is missing scene.wrist_cam")

    # Bring the main Kit viewport close to the table (default eye is far away).
    env_cfg.viewer.eye = (1.6, 1.2, 1.1)
    env_cfg.viewer.lookat = (0.5, 0.0, 0.15)

    with launch_simulation(env_cfg, args_cli), ExitStack():
        if _rtx_rendering_requested(args_cli):
            from isaaclab_physx.renderers.isaac_rtx_renderer_utils import apply_isaac_rtx_global_settings

            apply_isaac_rtx_global_settings(
                IsaacRtxRendererGlobalSettingsCfg(antialiasing_mode="DLSS")
            )

        try:
            env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        except Exception as error:
            logger.error(f"Failed to create environment: {error}")
            return

        state = AssistState()
        scanner = SceneScanner()
        view = None

        def request_reset() -> None:
            state.should_reset = True
            print("[TeleAssist] reset requested", flush=True)

        def request_rescan() -> None:
            state.should_rescan = True
            print("[TeleAssist] rescan requested", flush=True)

        def toggle_snap() -> None:
            state.snap_enabled = not state.snap_enabled
            state.last_snap_object = None
            state.last_snap_mode = "idle"
            state.snap_memory.clear()
            print(
                f"[TeleAssist] orientation snap {'ON' if state.snap_enabled else 'OFF'}",
                flush=True,
            )

        callbacks: dict[str, Callable[[], None]] = {
            "R": request_reset,
            "RESET": request_reset,
            "P": request_rescan,
            "O": toggle_snap,
        }

        device_name = args_cli.teleop_device
        if hasattr(env_cfg, "teleop_devices") and device_name in env_cfg.teleop_devices.devices:
            teleop_interface = create_teleop_device(
                device_name, env_cfg.teleop_devices.devices, callbacks
            )
        else:
            from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg

            sensitivity = args_cli.sensitivity
            teleop_interface = Se3Keyboard(
                Se3KeyboardCfg(
                    pos_sensitivity=0.05 * sensitivity,
                    rot_sensitivity=0.05 * sensitivity,
                )
            )
            for key, callback in callbacks.items():
                try:
                    teleop_interface.add_callback(key, callback)
                except (ValueError, TypeError) as error:
                    logger.warning(f"Failed to add callback for {key}: {error}")

        print(f"Using teleop device: {teleop_interface}", flush=True)
        print(
            "TeleAssist keys: WASD/QE move, ZX/TG/CV rotate, K gripper, "
            "R reset+rescan, P rescan, O toggle orientation snap",
            flush=True,
        )
        print("[TeleAssist] orientation snap ON (near scanned objects)", flush=True)

        def _seat_mug_after_mesh_ready() -> None:
            # Reset-time USD bounds are often empty; seat once meshes have extents,
            # then let PhysX settle while re-zeroing velocity so it doesn't bounce up.
            import torch
            from teleassist.envs.events import seat_rigid_on_table

            env_ids = torch.arange(env.num_envs, device=env.device)
            mug = env.scene["mug"]
            for _ in range(6):
                env.sim.render()
            for _ in range(8):
                seat_rigid_on_table(env, env_ids, surface_z=0.0, clearance=-0.001)
                for _ in range(4):
                    env.sim.step(render=False)
                mug.write_root_velocity_to_sim_index(
                    root_velocity=torch.zeros(env.num_envs, 6, device=env.device),
                    env_ids=env_ids,
                )
            env.sim.render()

        env.reset()
        teleop_interface.reset()
        _seat_mug_after_mesh_ready()

        # Open panels after the env/UI exist. Prefer Kit image windows; OpenCV fallback.
        if not args_cli.disable_external_cameras:
            try:
                _log_wrist_cam_parent(env)
                view = TableCameraWindows(
                    resolution=(
                        int(env_cfg.scene.table_cam.width),
                        int(env_cfg.scene.table_cam.height),
                    ),
                    table_camera_prim=_table_cam_prim(env),
                    wrist_resolution=(
                        int(env_cfg.scene.wrist_cam.width),
                        int(env_cfg.scene.wrist_cam.height),
                    ),
                    wrist_camera_prim=_wrist_cam_prim(env),
                )
            except Exception as error:
                import traceback

                print(f"[TeleAssist] Could not create camera panels: {error}", flush=True)
                traceback.print_exc()
            _run_scan(env, scanner, state, view)
        else:
            print("[TeleAssist] cameras disabled; skipping scan", flush=True)

        frame_i = 0
        while env.sim.is_running():
            try:
                with torch.inference_mode():
                    action = teleop_interface.advance()
                    if state.teleoperation_active and action is not None:
                        action = _maybe_apply_snap(env, action, state, frame_i=frame_i)
                        env.step(action.repeat(env.num_envs, 1))
                    else:
                        env.sim.render()

                    if state.should_reset:
                        env.reset()
                        teleop_interface.reset()
                        _seat_mug_after_mesh_ready()
                        state.should_reset = False
                        state.should_rescan = True
                        print("[TeleAssist] environment reset complete", flush=True)

                    if state.should_rescan and not args_cli.disable_external_cameras:
                        state.should_rescan = False
                        _run_scan(env, scanner, state, view)
                    elif view is not None and frame_i % 6 == 0:
                        # Keep table panels + wrist live RGB fresh; overlays reuse last scan.
                        try:
                            frame = read_table_camera(env)
                            state.last_rgb = frame.rgb
                        except Exception:
                            pass
                        try:
                            wrist = read_wrist_camera(env)
                            state.last_wrist_rgb = wrist.rgb
                        except Exception:
                            pass
                        _update_camera_ui(view, state)
                    frame_i += 1
            except Exception as error:
                logger.error(f"Error during simulation step: {error}")
                break

        env.close()
        print("Environment closed", flush=True)


if __name__ == "__main__":
    main()
