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
parser.add_argument("--task", type=str, default="TeleAssist-Stack-Cube-Franka-IK-Rel-RGBD")
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
from teleassist.camera_lab import read_table_camera
from teleassist.scan import GraspAssistMap, SceneScanner
from teleassist.view import TableCameraWindows

logger = logging.getLogger(__name__)


@dataclass
class AssistState:
    map: GraspAssistMap = field(default_factory=GraspAssistMap)
    should_reset: bool = False
    should_rescan: bool = False
    teleoperation_active: bool = True
    last_rgb: object | None = None
    last_boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    last_heatmap: object | None = None
    last_grasp_pixels: object | None = None


def _rtx_rendering_requested(args: argparse.Namespace) -> bool:
    visualizers = getattr(args, "visualizer", None) or []
    external_cameras = not getattr(args, "disable_external_cameras", False)
    return external_cameras or ("kit" in visualizers) or bool(getattr(args, "xr", False))


def _table_cam_prim(env) -> str:
    camera = env.scene.sensors["table_cam"] if hasattr(env.scene, "sensors") else env.scene["table_cam"]
    path = getattr(camera, "cfg", None)
    if path is not None and hasattr(path, "prim_path"):
        # Expand the first env namespace used by Lab.
        return str(path.prim_path).replace("{ENV_REGEX_NS}", "/World/envs/env_0")
    return "/World/envs/env_0/table_cam"


def _update_camera_ui(view: TableCameraWindows | None, state: AssistState) -> None:
    if view is None or state.last_rgb is None:
        return
    view.update(
        state.last_rgb,
        boxes=state.last_boxes or None,
        points_xy=state.last_grasp_pixels,
        heatmap=state.last_heatmap,
    )


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

        callbacks: dict[str, Callable[[], None]] = {
            "R": request_reset,
            "RESET": request_reset,
            "P": request_rescan,
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
            "TeleAssist keys: WASD/QE move, ZX/TG/CV rotate, K gripper, R reset+rescan, P rescan",
            flush=True,
        )

        env.reset()
        teleop_interface.reset()

        # Open panels after the env/UI exist. Prefer Kit image windows; OpenCV fallback.
        if not args_cli.disable_external_cameras:
            for _ in range(4):
                env.sim.render()
            try:
                view = TableCameraWindows(
                    resolution=(
                        int(env_cfg.scene.table_cam.width),
                        int(env_cfg.scene.table_cam.height),
                    ),
                    table_camera_prim=_table_cam_prim(env),
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
                        env.step(action.repeat(env.num_envs, 1))
                    else:
                        env.sim.render()

                    if state.should_reset:
                        env.reset()
                        teleop_interface.reset()
                        state.should_reset = False
                        state.should_rescan = True
                        print("[TeleAssist] environment reset complete", flush=True)

                    if state.should_rescan and not args_cli.disable_external_cameras:
                        state.should_rescan = False
                        _run_scan(env, scanner, state, view)
                    elif view is not None and frame_i % 6 == 0:
                        # Keep the overlay panel alive with a fresh RGB frame + last boxes.
                        try:
                            frame = read_table_camera(env)
                            state.last_rgb = frame.rgb
                            _update_camera_ui(view, state)
                        except Exception:
                            pass
                    frame_i += 1
            except Exception as error:
                logger.error(f"Error during simulation step: {error}")
                break

        env.close()
        print("Environment closed", flush=True)


if __name__ == "__main__":
    main()
