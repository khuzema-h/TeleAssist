"""TeleAssist Isaac Lab environment registrations."""

from __future__ import annotations

import gymnasium as gym

gym.register(
    id="TeleAssist-Stack-Cube-Franka-IK-Rel-RGBD",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "teleassist.envs.stack_ik_rel_rgbd_env_cfg:TeleAssistFrankaStackRgbdEnvCfg"
        ),
    },
    disable_env_checker=True,
)
