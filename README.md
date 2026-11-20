# TeleAssist

Cartesian teleop assistant for a Franka arm in **Isaac Lab**. The table scene currently has the kitchen **mug** only (plate temporarily removed). Milestone B adds a scene **GraspAssistMap** scan (OWL-ViT + SAM + OOAL + grasp orientation) during teleop. Near a scanned object, **orientation snap** blends the EE toward that object’s grasp quaternion (`O` toggles).

## Prerequisites

- Conda env `ps_vla` with pip Isaac Sim 6.2 and Isaac Lab 3.0 (`~/IsaacLab`, `release/3.0.0`)
- SAM 2 installed and checkpoint at `../checkpoints/sam2_hiera_base_plus.pt` (or `TELEASSIST_SAM_CKPT`)
- OOAL checkout at `../OOAL` and weights at `../checkpoints/ooal-models/seen_best` (or `TELEASSIST_OOAL_ROOT`)
- `transformers>=4.46,<5` for OWL-ViT (pin if Lab upgraded transformers to 5.x)

```bash
conda activate ps_vla
cd /path/to/TeleAssist
pip install -e .
# if needed:
pip install 'transformers>=4.46,<5'
```

Accept the Omniverse EULA (launchers set `OMNI_KIT_ACCEPT_EULA=YES`).

## Run Franka SE(3) teleop + scan

```bash
conda activate ps_vla
export OMNI_KIT_ACCEPT_EULA=YES
export ISAACLAB_PATH=~/IsaacLab
cd /path/to/TeleAssist
./scripts/teleop_franka_se3.sh
```

The launcher defaults to `--viz kit` (GUI). Pass another `--viz` / `--visualizer` only if you intentionally want a different viewer.

Defaults:

| Variable | Default |
| --- | --- |
| `ISAACLAB_PATH` | `~/IsaacLab` |
| `TASK` | `TeleAssist-Mug-Plate-Franka-IK-Rel-RGBD` |
| `TELEOP_DEVICE` | `keyboard` |
| `TELEASSIST_SAM_CKPT` | `../checkpoints/sam2_hiera_base_plus.pt` |
| `TELEASSIST_OOAL_ROOT` | `../OOAL` |

The launcher prepends `site-packages/nvidia/cu13/lib` to `LD_LIBRARY_PATH` for Torch CUDA 13 NVRTC.

### Keyboard bindings

Focus the Kit viewport window:

| Keys | Action |
| --- | --- |
| `W` / `S` | Move along X |
| `A` / `D` | Move along Y |
| `Q` / `E` | Move along Z |
| `Z` / `X` | Rotate roll |
| `T` / `G` | Rotate pitch |
| `C` / `V` | Rotate yaw |
| `K` | Toggle gripper |
| `R` | Reset env and rescan |
| `P` | Rescan GraspAssistMap |
| `O` | Toggle orientation snap |

On start (and after `R` / `P`), the table RGB-D camera is scanned. Logs list each object’s center, affordance, grasp quaternion, and point count (mug). Three image panels open on the right of the Kit viewport: **Table Camera** (live RGB), **Affordance Heatmap** (OOAL JET overlay + grasp pixels), and **Table Camera Scan** (grounding boxes).

When the EE is within ~20 cm of the object’s grasp region (full strength inside ~8 cm), orientation snap adds a rotational delta toward that grasp quaternion. Distance is to OOAL grasp points (tracked with the live mug pose), not the object center. Log lines: `[TeleAssist] snap check ...` and `[TeleAssist] snap → mug ...`. Press `O` to disable; press `P` after restart so the map (and grasp quat) refresh.

## Layout

```
TeleAssist/
  README.md
  pyproject.toml
  scripts/teleop_franka_se3.sh   # launcher
  scripts/teleop_assist.py       # teleop + scan loop
  scripts/verify_lab_kit.sh      # Kit smoke test
  teleassist/
    envs/                        # RGB-D Franka IK-Rel gym task
    grounding.py                 # OWL-ViT + color boxes
    sam.py                       # SAM 2 segment_region
    affordance.py                # OOAL heatmaps (no Tev1)
    geometry_scan.py             # Lab unproject + grasp orientation
    camera_lab.py                # read table_cam buffers
    catalog.py                   # mug (kitchen phrase; plate removed for now)
    scan.py                      # GraspAssistMap builder
    snap.py                      # orientation snap toward grasp quats
  tests/test_geometry_scan.py
  tests/test_snap.py
```

## Notes

- Lab’s stock cube-stack task is reused as the Franka/table base; TeleAssist swaps the cubes for the kitchen mug USD (plate temporarily omitted) and registers `TeleAssist-Mug-Plate-Franka-IK-Rel-RGBD` with a 480×480 table RGB-D camera.
- Scanning loads OWL-ViT, SAM, and optionally OOAL on GPU; SAM is released after each scan to free VRAM.
- Installing Isaac Lab into `ps_vla` can change shared deps used by other projects in that env.
