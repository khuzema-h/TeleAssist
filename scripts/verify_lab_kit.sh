#!/usr/bin/env bash
# Short Isaac Lab Kit smoke test with CUDA 13 NVRTC on the loader path.
set -euo pipefail

ISAACLAB_PATH="${ISAACLAB_PATH:-${HOME}/IsaacLab}"

if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "ps_vla" ]]; then
  echo "Activate conda env ps_vla first: conda activate ps_vla" >&2
  exit 1
fi

if [[ ! -x "${ISAACLAB_PATH}/isaaclab.sh" ]]; then
  echo "Isaac Lab not found at ISAACLAB_PATH=${ISAACLAB_PATH}" >&2
  exit 1
fi

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

_PY_SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
_CU13_LIB="${_PY_SITE}/nvidia/cu13/lib"
if [[ -d "${_CU13_LIB}" ]]; then
  export LD_LIBRARY_PATH="${_CU13_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ISAACLAB_PATH}"
exec ./isaaclab.sh -p "${ROOT}/scripts/verify_lab_kit.py" --viz kit "$@"
