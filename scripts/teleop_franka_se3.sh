#!/usr/bin/env bash
# Launch TeleAssist Franka SE(3) teleop with GraspAssistMap scanning.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_PATH="${ISAACLAB_PATH:-${HOME}/IsaacLab}"
TASK="${TASK:-TeleAssist-Stack-Cube-Franka-IK-Rel-RGBD}"
TELEOP_DEVICE="${TELEOP_DEVICE:-keyboard}"
NUM_ENVS="${NUM_ENVS:-1}"

if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "ps_vla" ]]; then
  echo "Activate conda env ps_vla first: conda activate ps_vla" >&2
  exit 1
fi

if [[ ! -x "${ISAACLAB_PATH}/isaaclab.sh" ]]; then
  echo "Isaac Lab not found at ISAACLAB_PATH=${ISAACLAB_PATH}" >&2
  echo "Clone release/3.0.0 to ~/IsaacLab or set ISAACLAB_PATH." >&2
  exit 1
fi

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Torch cu130 needs NVIDIA CUDA 13 NVRTC builtins on the dynamic loader path.
_PY_SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
_CU13_LIB="${_PY_SITE}/nvidia/cu13/lib"
if [[ -d "${_CU13_LIB}" ]]; then
  export LD_LIBRARY_PATH="${_CU13_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# Perception assets live next to TeleAssist inside the pseudo-vla checkout by default.
export TELEASSIST_SAM_CKPT="${TELEASSIST_SAM_CKPT:-${ROOT}/../checkpoints/sam2_hiera_base_plus.pt}"
export TELEASSIST_OOAL_ROOT="${TELEASSIST_OOAL_ROOT:-${ROOT}/../OOAL}"

echo "TeleAssist Franka SE(3) teleop + scan"
echo "  ISAACLAB_PATH=${ISAACLAB_PATH}"
echo "  task=${TASK}"
echo "  device=${TELEOP_DEVICE}"
echo "  cwd=${ROOT}"
echo
echo "Keyboard (focus the Kit window):"
echo "  W/S A/D Q/E  translate X/Y/Z"
echo "  Z/X T/G C/V  rotate roll/pitch/yaw"
echo "  K            toggle gripper"
echo "  R            reset environment + rescan"
echo "  P            rescan GraspAssistMap"
echo

# Default to a Kit viewport so teleop is not headless. Pass your own --viz/--visualizer
# to override (e.g. omit GUI intentionally).
EXTRA_ARGS=("$@")
HAS_VIZ=0
for arg in "${EXTRA_ARGS[@]}"; do
  case "${arg}" in
    --viz|--viz=*|--visualizer|--visualizer=*) HAS_VIZ=1 ;;
  esac
done
if [[ "${HAS_VIZ}" -eq 0 ]]; then
  EXTRA_ARGS=(--viz kit "${EXTRA_ARGS[@]}")
fi

cd "${ISAACLAB_PATH}"
exec ./isaaclab.sh -p "${ROOT}/scripts/teleop_assist.py" \
  --task "${TASK}" \
  --num_envs "${NUM_ENVS}" \
  --teleop_device "${TELEOP_DEVICE}" \
  "${EXTRA_ARGS[@]}"
