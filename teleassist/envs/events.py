"""Reset helpers for TeleAssist table props."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _env_ids_list(env: ManagerBasedEnv, env_ids) -> list[int]:
    if env_ids is None:
        return list(range(env.num_envs))
    if isinstance(env_ids, slice):
        return list(range(env.num_envs)[env_ids])
    if hasattr(env_ids, "tolist"):
        return env_ids.tolist()
    return list(env_ids)


def _world_aabb_min_z(stage, prim_path: str) -> float | None:
    """Return world-space AABB min Z for a prim, or None if unavailable."""
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    cache.Clear()
    bound = cache.ComputeWorldBound(prim)
    box = bound.ComputeAlignedBox()
    minimum = box.GetMin()
    maximum = box.GetMax()
    # Empty / unloaded mesh often reports a degenerate box at the origin.
    if (maximum - minimum).GetLength() < 1e-4:
        return None
    return float(minimum[2])


def seat_rigid_on_table(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("mug"),
    table_cfg: SceneEntityCfg = SceneEntityCfg("table"),
    surface_z: float | None = 0.0,
    clearance: float = 0.0,
) -> None:
    """Lower/raise a rigid object so its world AABB bottom rests on the table top.

    Lab's mug pivot is not the mesh bottom, and the USD bounds are often not ready
    during the first reset event — call this again after a few renders if needed.
    """
    import omni.usd

    asset = env.scene[asset_cfg.name]
    stage = omni.usd.get_context().get_stage()
    ids = _env_ids_list(env, env_ids)

    for env_id in ids:
        mug_path = asset.cfg.prim_path.replace("{ENV_REGEX_NS}", f"/World/envs/env_{env_id}")
        min_z = _world_aabb_min_z(stage, mug_path)
        if min_z is None:
            print(f"[TeleAssist] seat skipped: no mug bounds yet at {mug_path}", flush=True)
            continue

        if surface_z is None:
            try:
                table_asset = env.scene[table_cfg.name]
                table_path = table_asset.cfg.prim_path.replace(
                    "{ENV_REGEX_NS}", f"/World/envs/env_{env_id}"
                )
            except Exception:
                table_path = f"/World/envs/env_{env_id}/Table"
            table_min = _world_aabb_min_z(stage, table_path)
            # Prefer max Z of table if we can read a real box.
            from pxr import Usd, UsdGeom

            table_prim = stage.GetPrimAtPath(table_path)
            if table_prim.IsValid():
                cache = UsdGeom.BBoxCache(
                    Usd.TimeCode.Default(),
                    [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                    useExtentsHint=True,
                )
                cache.Clear()
                table_box = cache.ComputeWorldBound(table_prim).ComputeAlignedBox()
                resolved_surface = float(table_box.GetMax()[2])
            else:
                resolved_surface = 0.0 if table_min is None else table_min
        else:
            resolved_surface = float(surface_z)

        lift = resolved_surface + clearance - min_z
        pos = asset.data.root_pos_w.torch[env_id].clone()
        quat = asset.data.root_quat_w.torch[env_id].clone()
        print(
            f"[TeleAssist] seat mug env={env_id} min_z={min_z:.4f} surface={resolved_surface:.4f} "
            f"lift={lift:.4f} root_z={float(pos[2]):.4f}",
            flush=True,
        )
        if abs(lift) < 1e-5:
            continue

        pos[2] += lift
        env_ids_t = torch.tensor([env_id], device=env.device)
        root_pose = torch.cat([pos.unsqueeze(0), quat.unsqueeze(0)], dim=-1)
        asset.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids_t)
        asset.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros(1, 6, device=env.device),
            env_ids=env_ids_t,
        )
        # Keep the seated pose as the default so later resets do not float it again.
        try:
            asset.data.default_root_state.torch[env_id, 0:3] = pos
            asset.data.default_root_state.torch[env_id, 3:7] = quat
        except Exception:
            pass
