#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/slurm_common.sh"

affinity_enable_error_trap
affinity_resolve_project_dir
affinity_load_modules

ENV_NAME="${AFFINITY_CONDA_ENV:-affitest}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "conda env already exists: ${ENV_NAME}"
else
  conda create -n "$ENV_NAME" "python=${PYTHON_VERSION}" -y
fi

conda activate "$ENV_NAME"
python -m pip install --upgrade pip

# Keep PyTorch explicit so the GPU build is installed before requirements.txt.
python -m pip install torch --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -m pip check

echo "conda env ready: ${ENV_NAME}"
echo "submit scripts/slurm/smoke_test.sbatch to validate the environment on a compute node"
