"""Franka IK-Rel stack env with a table RGB-D camera for TeleAssist scanning."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab_tasks.contrib.stack.config.franka.stack_ik_rel_env_cfg import FrankaCubeStackEnvCfg
from isaaclab_tasks.utils.presets import set_isaac_rtx_global_settings


@configclass
class TeleAssistFrankaStackRgbdEnvCfg(FrankaCubeStackEnvCfg):
    """IK-Rel Franka stack with a 480x480 table RGB-D camera."""

    def __post_init__(self):
        super().__post_init__()

        # StackEnvCfg leaves sim.physics as a preset selector; Lab requires a concrete backend.
        self.sim.physics = self.sim.physics.default

        self.scene.table_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam",
            update_period=0.0,
            height=480,
            width=480,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 2.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(1.0, 0.0, 0.4),
                rot=(-0.61237, -0.61237, 0.35355, 0.35355),
                convention="ros",
            ),
        )
        self.num_rerenders_on_reset = 3
        set_isaac_rtx_global_settings(self.scene.table_cam.renderer_cfg, antialiasing_mode="DLAA")
