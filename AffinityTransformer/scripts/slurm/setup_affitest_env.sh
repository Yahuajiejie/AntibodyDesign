#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/slurm_common.sh"

affinity_enable_error_trap
affinity_resolve_project_dir
export LOAD_CUDA="${LOAD_CUDA:-0}"
affinity_load_modules

ENV_NAME="${AFFINITY_CONDA_ENV:-affitest}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

echo "CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS}"
echo "CONDA_ENVS_PATH=${CONDA_ENVS_PATH}"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "conda env already exists: ${ENV_NAME}"
else
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

conda activate "${ENV_NAME}"

python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -m pip check

echo "conda env ready: ${ENV_NAME}"
