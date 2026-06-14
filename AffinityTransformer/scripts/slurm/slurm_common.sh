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
    echo "        Submit from the project root, or set AFFINITY_PROJECT_DIR=/path/to/AffinityTransformer." >&2
    echo "        SLURM_SUBMIT_DIR=${submit_dir}" >&2
    exit 2
  fi

  cd "${PROJECT_DIR}"
  mkdir -p logs/slurm outputs cache/huggingface cache/torch
  export PROJECT_DIR
}

affinity_configure_user_conda_dirs() {
  # Shared Anaconda is read-only for normal users. Keep conda's writable
  # package/env cache in the normal user location unless overridden.
  mkdir -p "${HOME}/.conda/pkgs" "${HOME}/.conda/envs"
  export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${HOME}/.conda/pkgs}"
  export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-${HOME}/.conda/envs}"
}

affinity_load_modules() {
  MODULE_INIT="${MODULE_INIT:-/gpfs/share/software/module/tools/modules/init/profile.sh}"
  CONDA_INIT="${CONDA_INIT:-/gpfs/share/software/anaconda/3-2023.09-0/etc/profile.d/conda.sh}"
  ANACONDA_MODULE="${ANACONDA_MODULE:-anaconda/3-2023.09-0}"
  CUDA_MODULE="${CUDA_MODULE:-cuda/11.8}"
  LOAD_CUDA="${LOAD_CUDA:-0}"

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
  affinity_configure_user_conda_dirs
  source "${CONDA_INIT}"
  if ! command -v conda >/dev/null 2>&1; then
    echo "[FATAL] conda command is unavailable after sourcing ${CONDA_INIT}" >&2
    exit 6
  fi
}

affinity_activate_conda() {
  local env_name="${AFFINITY_CONDA_ENV:-affitest}"
  if ! conda env list | awk '{print $1}' | grep -qx "${env_name}"; then
    echo "[FATAL] Conda env not found: ${env_name}" >&2
    echo "        Create it first with scripts/slurm/setup_affitest_env.sh or setup_affitest_env.sbatch." >&2
    exit 7
  fi

  conda activate "${env_name}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
  export HF_HOME="${HF_HOME:-${PROJECT_DIR}/cache/huggingface}"
  export TORCH_HOME="${TORCH_HOME:-${PROJECT_DIR}/cache/torch}"
  mkdir -p "${HF_HOME}" "${TORCH_HOME}"
}

affinity_print_header() {
  echo "========================================"
  echo "Job ID:          ${SLURM_JOB_ID:-manual}"
  echo "NodeList:        ${SLURM_NODELIST:-local}"
  echo "Host:            $(hostname)"
  echo "Start:           $(date)"
  echo "Project:         ${PROJECT_DIR:-$(pwd)}"
  echo "Conda env:       ${AFFINITY_CONDA_ENV:-affitest}"
  echo "CUDA module:     ${CUDA_MODULE:-cuda/11.8}"
  echo "Partition:       ${SLURM_JOB_PARTITION:-local}"
  echo "Python:          $(command -v python)"
  echo "CONDA_PREFIX:    ${CONDA_PREFIX:-}"
  echo "CONDA_PKGS_DIRS: ${CONDA_PKGS_DIRS:-}"
  echo "CONDA_ENVS_PATH: ${CONDA_ENVS_PATH:-}"
  echo "========================================"
}

affinity_check_gpu_runtime() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
  fi

  python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is False")
print("gpu:", torch.cuda.get_device_name(0))
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
