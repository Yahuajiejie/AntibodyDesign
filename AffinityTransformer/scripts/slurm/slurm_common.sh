#!/usr/bin/env bash

affinity_enable_error_trap() {
  trap 'code=$?; echo "[ERROR] ${BASH_SOURCE[1]:-${0}} failed near line ${LINENO}, exit=${code}" >&2' ERR
}

affinity_resolve_project_dir() {
  local submit_dir="${SLURM_SUBMIT_DIR:-$(pwd)}"

  if [[ -n "${AFFINITY_PROJECT_DIR:-}" ]]; then
    PROJECT_DIR="${AFFINITY_PROJECT_DIR}"
  elif [[ -f "${submit_dir}/train.py" && -d "${submit_dir}/affinity_transformer" ]]; then
    PROJECT_DIR="${submit_dir}"
  elif [[ -f "${submit_dir}/AffinityTransformer/train.py" && -d "${submit_dir}/AffinityTransformer/affinity_transformer" ]]; then
    PROJECT_DIR="${submit_dir}/AffinityTransformer"
  else
    echo "[FATAL] Cannot locate AffinityTransformer project root." >&2
    echo "        Submit from the project root, or set:" >&2
    echo "        AFFINITY_PROJECT_DIR=/path/to/AffinityTransformer sbatch ..." >&2
    echo "        SLURM_SUBMIT_DIR=${submit_dir}" >&2
    exit 2
  fi

  if [[ ! -d "${PROJECT_DIR}" ]]; then
    echo "[FATAL] Project directory does not exist: ${PROJECT_DIR}" >&2
    exit 2
  fi

  cd "${PROJECT_DIR}"
  mkdir -p logs/slurm outputs cache/huggingface cache/torch
  export PROJECT_DIR
}

affinity_load_modules() {
  MODULE_INIT="${MODULE_INIT:-/gpfs/share/software/module/tools/modules/init/profile.sh}"
  CONDA_INIT="${CONDA_INIT:-/gpfs/share/software/anaconda/3-2023.09-0/etc/profile.d/conda.sh}"
  ANACONDA_MODULE="${ANACONDA_MODULE:-anaconda/3-2023.09-0}"
  CUDA_MODULE="${CUDA_MODULE:-cuda/11.8}"
  LOAD_CUDA="${LOAD_CUDA:-1}"

  if [[ ! -f "${MODULE_INIT}" ]]; then
    echo "[FATAL] Missing module init script: ${MODULE_INIT}" >&2
    exit 3
  fi
  source "${MODULE_INIT}"
  if ! command -v module >/dev/null 2>&1; then
    echo "[FATAL] module command is unavailable after sourcing ${MODULE_INIT}" >&2
    exit 4
  fi

  module purge || true
  module load "${ANACONDA_MODULE}"
  if [[ "${LOAD_CUDA}" == "1" && -n "${CUDA_MODULE}" ]]; then
    module load "${CUDA_MODULE}"
  fi
  module list

  if [[ ! -f "${CONDA_INIT}" ]]; then
    echo "[FATAL] Missing conda init script: ${CONDA_INIT}" >&2
    exit 5
  fi
  source "${CONDA_INIT}"
  if ! command -v conda >/dev/null 2>&1; then
    echo "[FATAL] conda command is unavailable after sourcing ${CONDA_INIT}" >&2
    exit 6
  fi
}

affinity_activate_conda() {
  ENV_NAME="${AFFINITY_CONDA_ENV:-affitest}"
  if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[FATAL] Conda env not found: ${ENV_NAME}" >&2
    echo "        Create it first with scripts/slurm/setup_affitest_env.sh or setup_affitest_env.sbatch." >&2
    exit 7
  fi

  conda activate "${ENV_NAME}"
  echo "[env] python=$(command -v python)"
  echo "[env] CONDA_PREFIX=${CONDA_PREFIX}"

  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
  export HF_HOME="${HF_HOME:-${PROJECT_DIR}/cache/huggingface}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
  export TORCH_HOME="${TORCH_HOME:-${PROJECT_DIR}/cache/torch}"
  mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${TORCH_HOME}"
}

affinity_print_header() {
  echo "========================================"
  echo "Job ID:      ${SLURM_JOB_ID:-manual}"
  echo "NodeList:    ${SLURM_NODELIST:-local}"
  echo "Host:        $(hostname)"
  echo "Start:       $(date)"
  echo "Project:     ${PROJECT_DIR:-$(pwd)}"
  echo "Conda env:   ${AFFINITY_CONDA_ENV:-affitest}"
  echo "CUDA module: ${CUDA_MODULE:-cuda/11.8}"
  echo "Partition:   ${SLURM_JOB_PARTITION:-local}"
  echo "GPUs:        ${SLURM_GPUS:-${CUDA_VISIBLE_DEVICES:-none}}"
  echo "========================================"
}

affinity_check_gpu_runtime() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
  fi

  python - <<'PY'
import importlib.util
import sys

required = ["torch", "transformers", "pandas", "pyarrow", "yaml", "scipy"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing Python package(s): {missing}")

import torch
import transformers
import pandas
import pyarrow
import scipy
import yaml

print(f"Python:       {sys.version.split()[0]}")
print(f"PyTorch:      {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Transformers: {transformers.__version__}")
print(f"pandas:       {pandas.__version__}")
print(f"pyarrow:      {pyarrow.__version__}")
print(f"SciPy:        {scipy.__version__}")
print(f"PyYAML:       {yaml.__version__}")
print(f"CUDA ready:   {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is False; check Slurm GPU allocation and torch CUDA build.")
print(f"GPU:          {torch.cuda.get_device_name(0)}")
PY
}

affinity_check_cpu_runtime() {
  python - <<'PY'
import importlib.util

required = ["pandas", "pyarrow", "yaml"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing Python package(s): {missing}")
print("CPU runtime check passed")
PY
}
