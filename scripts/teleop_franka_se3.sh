#!/usr/bin/env bash
# Launch Franka SE(3) keyboard teleop via Isaac Lab.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_PATH="${ISAACLAB_PATH:-${HOME}/IsaacLab}"
TASK="${TASK:-IsaacContrib-Stack-Cube-Franka-IK-Rel}"
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

# Torch cu130 needs NVIDIA CUDA 13 NVRTC builtins on the dynamic loader path.
_PY_SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
_CU13_LIB="${_PY_SITE}/nvidia/cu13/lib"
if [[ -d "${_CU13_LIB}" ]]; then
  export LD_LIBRARY_PATH="${_CU13_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

echo "TeleAssist Franka SE(3) teleop"
echo "  ISAACLAB_PATH=${ISAACLAB_PATH}"
echo "  task=${TASK}"
echo "  device=${TELEOP_DEVICE}"
echo "  cwd=${ROOT}"
echo
echo "Keyboard (focus the Kit window):"
echo "  W/S A/D Q/E  translate X/Y/Z"
echo "  Z/X T/G C/V  rotate roll/pitch/yaw"
echo "  K            toggle gripper"
echo "  R            reset commands"
echo

cd "${ISAACLAB_PATH}"
exec ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task "${TASK}" \
  --num_envs "${NUM_ENVS}" \
  --teleop_device "${TELEOP_DEVICE}" \
  "$@"
