#!/bin/bash
#SBATCH --job-name=flab_affinity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a800:1
#SBATCH --partition=GPUA800
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.log
#SBATCH --export=ALL

# run.slurm — FLAb 通用亲和力排序模型训练脚本
#
# 只在 login 节点提交作业，不要直接 bash run.slurm 跑计算：
#   cd /gpfs/share/home/2300012116/Antibody/FLAb
#   mkdir -p logs
#   sbatch run.slurm
#
# 常用覆盖参数：
#   TRAIN_MODE=train sbatch run.slurm
#   LOSS_NAMES="ranknet" sbatch run.slurm
#   CUDA_MODULE=cuda/11.8 sbatch run.slurm
#
# 如果集群分区或 GPU 资源名不同，修改上面的 --partition / --gres。

set -euo pipefail

trap 'echo "[ERROR] 作业在第 ${LINENO} 行失败，退出码=$?；请查看上方日志定位原因。" >&2' ERR

ENV_NAME="${CONDA_ENV_NAME:-esm2}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.6}"
TRAIN_MODE="${TRAIN_MODE:-all}"
LOSS_NAMES="${LOSS_NAMES:-mse hinge ranknet}"
MODULE_INIT="${MODULE_INIT:-/gpfs/share/software/module/tools/modules/init/profile.sh}"
CONDA_INIT="${CONDA_INIT:-/gpfs/share/software/anaconda/3-2023.09-0/etc/profile.d/conda.sh}"

echo "========================================"
echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Node:        ${SLURM_NODELIST:-local}"
echo "Start:       $(date)"
echo "SubmitDir:   ${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "Conda env:   ${ENV_NAME}"
echo "CUDA module: ${CUDA_MODULE}"
echo "Train mode:  ${TRAIN_MODE}"
echo "Loss names:  ${LOSS_NAMES}"
echo "========================================"

# ── 定位 FLAb 项目目录 ─────────────────────────────────────────────
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

if [[ -n "${FLAB_PROJECT_DIR:-}" ]]; then
    PROJECT_DIR="${FLAB_PROJECT_DIR}"
elif [[ -f "${SUBMIT_DIR}/train.py" && -d "${SUBMIT_DIR}/affinity_model" ]]; then
    PROJECT_DIR="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/FLAb/train.py" && -d "${SUBMIT_DIR}/FLAb/affinity_model" ]]; then
    PROJECT_DIR="${SUBMIT_DIR}/FLAb"
else
    echo "[FATAL] 找不到 FLAb/train.py。请从 FLAb 目录或项目根目录提交 sbatch。" >&2
    echo "        也可以用 FLAB_PROJECT_DIR=/path/to/FLAb sbatch run.slurm 指定路径。" >&2
    echo "当前 SUBMIT_DIR=${SUBMIT_DIR}" >&2
    exit 2
fi

cd "${PROJECT_DIR}"
mkdir -p logs results/affinity_model cache/embeddings cache/huggingface cache/torch

echo "[路径] PROJECT_DIR=$(pwd)"
echo "[路径] data/binding 文件数: $(find data/binding -maxdepth 1 \( -name '*.csv' -o -name '*.csv.zip' \) | wc -l)"

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

required = ["torch", "transformers", "numpy", "pandas", "scipy", "sklearn", "tqdm"]
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
import sklearn

print(f"Python:       {sys.version.split()[0]}")
print(f"PyTorch:      {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"NumPy:        {numpy.__version__}")
print(f"pandas:       {pandas.__version__}")
print(f"SciPy:        {scipy.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"CUDA:         {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() 是 False。请检查 torch CUDA 版本、GPU 资源和驱动。")

print(f"GPU:          {torch.cuda.get_device_name(0)}")
PY

read -r -a LOSS_ARGS <<< "${LOSS_NAMES}"

case "${TRAIN_MODE}" in
    embed|train|all)
        ;;
    *)
        echo "[FATAL] TRAIN_MODE 必须是 embed / train / all，当前是: ${TRAIN_MODE}" >&2
        exit 7
        ;;
esac

# ── 运行训练入口 ─────────────────────────────────────────────────
echo ""
echo "[Run] 开始执行 train.py: $(date)"
python train.py \
    --mode "${TRAIN_MODE}" \
    --data_dir data/binding \
    --cache_dir cache/embeddings \
    --output_dir results/affinity_model \
    --loss "${LOSS_ARGS[@]}"
echo "[Run] train.py 完成: $(date)"

echo ""
echo "========================================"
echo "全部完成: $(date)"
echo "结果汇总: ${PROJECT_DIR}/results/affinity_model/summary_global_losses.csv"
echo "逐组结果: ${PROJECT_DIR}/results/affinity_model/global_*_test_by_group.csv"
echo "模型权重: ${PROJECT_DIR}/results/affinity_model/global_*.pt"
echo "========================================"
