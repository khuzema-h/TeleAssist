# TeleAssist

Cartesian teleop assistant for a Franka arm in **Isaac Lab**. Milestone A is SE(3) keyboard teleop only; grasp snap-assist comes later.

## Prerequisites

- Conda env `ps_vla` with pip Isaac Sim 6.2 and Isaac Lab 3.0 installed
- Isaac Lab checkout at `~/IsaacLab` (`release/3.0.0`), or set `ISAACLAB_PATH`
- Accept the Omniverse EULA (this repo’s launchers set `OMNI_KIT_ACCEPT_EULA=YES`)

### Install Isaac Lab into `ps_vla` (one-time)

```bash
git clone git@github.com:isaac-sim/IsaacLab.git --branch release/3.0.0 ~/IsaacLab
conda activate ps_vla
cd ~/IsaacLab
./isaaclab.sh -i
```

Smoke-test Kit:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export ISAACLAB_PATH=~/IsaacLab
./scripts/verify_lab_kit.sh
```

## Run Franka SE(3) keyboard teleop

```bash
conda activate ps_vla
export OMNI_KIT_ACCEPT_EULA=YES
export ISAACLAB_PATH=~/IsaacLab
cd /path/to/TeleAssist
./scripts/teleop_franka_se3.sh
```

Defaults:

| Variable | Default |
| --- | --- |
| `ISAACLAB_PATH` | `~/IsaacLab` |
| `TASK` | `IsaacContrib-Stack-Cube-Franka-IK-Rel` |
| `TELEOP_DEVICE` | `keyboard` |
| `NUM_ENVS` | `1` |

Extra flags are forwarded to Isaac Lab’s teleop agent (for example `--viz kit`).

The launcher prepends `site-packages/nvidia/cu13/lib` to `LD_LIBRARY_PATH` so Torch CUDA 13 can find `libnvrtc-builtins.so.13.0`.

### Keyboard bindings

Focus the Kit viewport window, then:

| Keys | Action |
| --- | --- |
| `W` / `S` | Move along X |
| `A` / `D` | Move along Y |
| `Q` / `E` | Move along Z |
| `Z` / `X` | Rotate roll |
| `T` / `G` | Rotate pitch |
| `C` / `V` | Rotate yaw |
| `K` | Toggle gripper |
| `R` | Reset teleop commands |

## Layout

```
TeleAssist/
  README.md
  scripts/teleop_franka_se3.sh   # Franka IK-Rel keyboard launcher
  scripts/verify_lab_kit.py      # Short Kit smoke test
  teleassist/                    # Package stub for later snap-assist code
```

## Notes

- Lab 3.0 registers the relative-IK stack task as `IsaacContrib-Stack-Cube-Franka-IK-Rel` (not the older `Isaac-Stack-Cube-Franka-IK-Rel-v0` name).
- Installing Isaac Lab into `ps_vla` can change shared deps (PyTorch CUDA build, `transformers`, etc.) used by other projects in that env.
