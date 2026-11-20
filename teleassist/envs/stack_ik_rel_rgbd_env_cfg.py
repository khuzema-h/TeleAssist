"""Franka IK-Rel env with a kitchen mug on the table and a table RGB-D camera."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_physx.sim.schemas import PhysxCollisionCfg, PhysxRigidBodyCfg
from isaaclab_tasks.contrib.stack import mdp
from isaaclab_tasks.contrib.stack.config.franka.stack_ik_rel_env_cfg import FrankaCubeStackEnvCfg
from isaaclab_tasks.contrib.stack.stack_env_cfg import StackEventCfg
from isaaclab_tasks.utils.presets import set_isaac_rtx_global_settings

from teleassist.envs import events as teleassist_events

# Same Isaac mug mesh referenced by pseudo_vla/kitchen.py (MUG_ASSET).
MUG_USD = f"{ISAAC_NUCLEUS_DIR}/Props/Mugs/SM_Mug_A2.usd"

_PROP_RIGID = [
    PhysxRigidBodyCfg(
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=1,
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=5.0,
        disable_gravity=False,
    )
]

_MUG_MASS = 0.30
# SM_Mug_A2 is cm-authored; 0.01 → meters. After scale the body is ~9 cm tall.
_MUG_SCALE = (0.01, 0.01, 0.01)
_MUG_HEIGHT_M = 0.09
# Center of table workspace (plate removed for now).
_MUG_POS = (0.50, 0.0, _MUG_HEIGHT_M * 0.5)

# Isaac Lab root rot is (x, y, z, w) — not (w, x, y, z).
# +90° from the previous kitchen -90° yaw (handle facing).
_MUG_YAW_RAD = math.radians(0.0)
_MUG_QUAT = (
    0.0,
    0.0,
    math.sin(_MUG_YAW_RAD / 2.0),
    math.cos(_MUG_YAW_RAD / 2.0),
)


@configclass
class TeleAssistFrankaStackRgbdEnvCfg(FrankaCubeStackEnvCfg):
    """IK-Rel Franka table scene with a mug and a 480x480 table RGB-D camera."""

    def __post_init__(self):
        super().__post_init__()

        # StackEnvCfg leaves sim.physics as a preset selector; Lab requires a concrete backend.
        self.sim.physics = self.sim.physics.default

        # Replace stacking cubes with the kitchen mug only (plate removed for now).
        self.scene.cube_1 = None
        self.scene.cube_2 = None
        self.scene.cube_3 = None
        self.scene.plate = None
        self.scene.mug = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Mug",
            init_state=RigidObjectCfg.InitialStateCfg(pos=_MUG_POS, rot=_MUG_QUAT),
            spawn=UsdFileCfg(
                usd_path=MUG_USD,
                scale=_MUG_SCALE,
                rigid_props=_PROP_RIGID,
                mass_props=sim_utils.MassCfg(mass=_MUG_MASS),
                collision_props=[
                    sim_utils.UsdPhysicsCollisionCfg(),
                    PhysxCollisionCfg(contact_offset=0.001, rest_offset=0.0),
                    sim_utils.UsdPhysicsMeshCollisionCfg(mesh_approximation_name="convexHull"),
                ],
                semantic_tags=[("class", "mug")],
            ),
        )

        # Reset: keep joint noise; seat mug on the table from live AABB.
        self.events = StackEventCfg()
        self.events.randomize_cube_positions = None
        self.events.seat_mug = EventTerm(
            func=teleassist_events.seat_rigid_on_table,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("mug"),
                "table_cfg": SceneEntityCfg("table"),
                # Visual AABB sits flush at 0, but the convex-hull collider is slightly
                # larger and PhysX depenetrates upward — bias down so it rests on the top.
                "surface_z": 0.0,
                "clearance": -0.001,
            },
        )

        # Drop cube/plate-specific observations / subtasks (teleop does not need them).
        self.observations.policy.object = None
        self.observations.policy.cube_positions = None
        self.observations.policy.cube_orientations = None
        self.observations.subtask_terms = None

        self.terminations.cube_1_dropping = None
        self.terminations.cube_2_dropping = None
        self.terminations.cube_3_dropping = None
        self.terminations.success = None
        self.terminations.plate_dropping = None
        self.terminations.mug_dropping = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("mug")},
        )

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
