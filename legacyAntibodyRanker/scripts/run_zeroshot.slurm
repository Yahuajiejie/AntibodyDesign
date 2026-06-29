#!/bin/bash
#SBATCH --job-name=flab_zeroshot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=GPUA800
#SBATCH --time=12:00:00
#SBATCH --output=logs/zeroshot_%j.log
#SBATCH --error=logs/zeroshot_%j.log
#SBATCH --export=ALL

# run_zeroshot.slurm — FLAb ESM2 zero-shot 打分脚本
#
# 只在 login 节点提交作业，不要直接 bash run_zeroshot.slurm 跑计算：
#   cd /gpfs/share/home/2300012116/Antibody/FLAb
#   mkdir -p logs
#   sbatch run_zeroshot.slurm
#
# 默认只跑 esm2_650M。zero-shot pseudo-perplexity 很慢，3B/15B 更容易超时或 OOM。
# 常用覆盖参数：
#   ZEROSHOT_MODELS="esm2_8M esm2_150M esm2_650M" sbatch run_zeroshot.slurm
#   ZEROSHOT_MODELS="esm2_650M esm2_3B" sbatch run_zeroshot.slurm
#   MAX_ROWS=50 sbatch run_zeroshot.slurm
#   MAX_SIZE=2000 sbatch run_zeroshot.slurm
#   CUDA_MODULE=cuda/11.8 sbatch run_zeroshot.slurm
#
# 如果集群分区或 GPU 资源名不同，修改上面的 --partition / --gres。

set -euo pipefail

trap 'echo "[ERROR] 作业在第 ${LINENO} 行失败，退出码=$?；请查看上方日志定位原因。" >&2' ERR

ENV_NAME="${CONDA_ENV_NAME:-esm2}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.6}"
ZEROSHOT_MODELS="${ZEROSHOT_MODELS:-esm2_650M}"
DATA_DIR="${DATA_DIR:-data/binding}"
MAX_ROWS="${MAX_ROWS:-}"
MAX_SIZE="${MAX_SIZE:-5000}"
MODULE_INIT="${MODULE_INIT:-/gpfs/share/software/module/tools/modules/init/profile.sh}"
CONDA_INIT="${CONDA_INIT:-/gpfs/share/software/anaconda/3-2023.09-0/etc/profile.d/conda.sh}"

echo "========================================"
echo "Job ID:          ${SLURM_JOB_ID:-local}"
echo "Node:            ${SLURM_NODELIST:-local}"
echo "Start:           $(date)"
echo "SubmitDir:       ${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "Conda env:       ${ENV_NAME}"
echo "CUDA module:     ${CUDA_MODULE}"
echo "Zero-shot model: ${ZEROSHOT_MODELS}"
echo "Data dir:        ${DATA_DIR}"
echo "Max rows:        ${MAX_ROWS:-none}"
echo "Max size:        ${MAX_SIZE}"
echo "========================================"

# ── 定位 FLAb 项目目录 ─────────────────────────────────────────────
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

if [[ -n "${FLAB_PROJECT_DIR:-}" ]]; then
    PROJECT_DIR="${FLAB_PROJECT_DIR}"
elif [[ -f "${SUBMIT_DIR}/train.py" && -d "${SUBMIT_DIR}/models" ]]; then
    PROJECT_DIR="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/FLAb/train.py" && -d "${SUBMIT_DIR}/FLAb/models" ]]; then
    PROJECT_DIR="${SUBMIT_DIR}/FLAb"
else
    echo "[FATAL] 找不到 FLAb/train.py。请从 FLAb 目录或项目根目录提交 sbatch。" >&2
    echo "        也可以用 FLAB_PROJECT_DIR=/path/to/FLAb sbatch run_zeroshot.slurm 指定路径。" >&2
    echo "当前 SUBMIT_DIR=${SUBMIT_DIR}" >&2
    exit 2
fi

cd "${PROJECT_DIR}"
mkdir -p logs score_batch cache/huggingface cache/torch

echo "[路径] PROJECT_DIR=$(pwd)"
echo "[路径] ${DATA_DIR} 文件数: $(find "${DATA_DIR}" -maxdepth 1 \( -name '*.csv' -o -name '*.csv.zip' \) | wc -l)"

# ── 加载集群 module 和 CUDA ────────────────────────────────────────
if [[ ! -f "${MODULE_INIT}" ]]; then
    echo "[FATAL] 找不到 module 初始化脚本: ${MODULE_INIT}" >&2
    exit 3
fi

source "${MODULE_INIT}"

if ! command -v module >/dev/null 2>&1; then
    echo "[FATAL] module 命令不可用，请检查 MODULE_INIT。" >&2
    exit 4
fi

module purge || true
if [[ -n "${CUDA_MODULE}" ]]; then
    module load "${CUDA_MODULE}"
fi
module list

# ── 激活 conda 环境 ───────────────────────────────────────────────
if [[ ! -f "${CONDA_INIT}" ]]; then
    echo "[FATAL] 找不到 conda 初始化脚本: ${CONDA_INIT}" >&2
    exit 5
fi

source "${CONDA_INIT}"

if ! command -v conda >/dev/null 2>&1; then
    echo "[FATAL] batch 环境里找不到 conda，请检查 CONDA_INIT。" >&2
    exit 6
fi

conda activate "${ENV_NAME}"

echo "[环境] python=$(command -v python)"
echo "[环境] CONDA_PREFIX=${CONDA_PREFIX}"
conda info --envs

# 让 conda 的 libstdc++ 优先，常见于解决 GLIBCXX_x.y.z not found。
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# HuggingFace/torch 缓存目录放到项目内，避免默认 HOME 配额太小。
export HF_HOME="${PROJECT_DIR}/cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TORCH_HOME="${PROJECT_DIR}/cache/torch"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${TORCH_HOME}"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

# ── 环境自检：失败就立刻退出，不继续假装完成 ─────────────────────
echo ""
echo "[Check] Python / CUDA / 依赖自检..."
python - <<'PY'
import sys
import importlib

required = ["torch", "transformers", "numpy", "pandas", "scipy"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(
        f"缺少 Python 包: {missing}。请在当前 conda 环境中安装 FLAb/requirements.txt。"
    )

import torch
import transformers
import numpy
import pandas
import scipy

print(f"Python:       {sys.version.split()[0]}")
print(f"PyTorch:      {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"NumPy:        {numpy.__version__}")
print(f"pandas:       {pandas.__version__}")
print(f"SciPy:        {scipy.__version__}")
print(f"CUDA:         {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() 是 False。请检查 torch CUDA 版本、GPU 资源和驱动。")

print(f"GPU:          {torch.cuda.get_device_name(0)}")
PY

read -r -a MODEL_ARGS <<< "${ZEROSHOT_MODELS}"

CMD=(
    python models/run_scoring_batch.py
    --model "${MODEL_ARGS[@]}"
    --data_dir "${DATA_DIR}"
    --max_size "${MAX_SIZE}"
)

if [[ -n "${MAX_ROWS}" ]]; then
    CMD+=(--max_rows "${MAX_ROWS}")
fi

echo ""
echo "[Run] 开始执行 zero-shot scoring: $(date)"
printf '[Run] 命令:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
echo "[Run] zero-shot scoring 完成: $(date)"

echo ""
echo "========================================"
echo "全部完成: $(date)"
echo "结果目录: ${PROJECT_DIR}/score_batch"
echo "汇总文件: ${PROJECT_DIR}/score_batch/<model_name>/summary.csv"
echo "逐序列分数: ${PROJECT_DIR}/score_batch/<model_name>/*_perseq.csv"
echo "========================================"
