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

# Install via conda (uses cluster's local mirror, no public internet needed)
conda install -n "${ENV_NAME}" -y \
  pandas pyarrow pyyaml scipy pytest

# pytorch with CUDA 11.8 — conda channel takes priority over pip
conda install -n "${ENV_NAME}" -y \
  pytorch torchvision torchaudio pytorch-cuda=11.8 \
  -c pytorch -c nvidia || \
  echo "[WARN] conda pytorch install failed, will try pip fallback below"

# transformers not in standard conda channels, try pip (may need mirror)
pip install transformers huggingface_hub safetensors tokenizers \
  --no-deps \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple || \
  pip install transformers huggingface_hub safetensors tokenizers --no-deps

# python -m pip check  # skipped: base Anaconda packages cause false positives in conda envs

echo "conda env ready: ${ENV_NAME}"
