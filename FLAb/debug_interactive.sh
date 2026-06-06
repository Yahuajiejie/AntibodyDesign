#!/bin/bash

# debug_interactive.sh — 在 srun --pty bash 交互式 GPU 节点中手动调试 FLAb 代码
#
# 使用方式：
#   1. 在 login 节点只申请交互式资源，不直接跑计算：
#        cd /gpfs/share/home/2300012116/Antibody/FLAb
#        mkdir -p logs
#        srun --partition=GPUA800 --gres=gpu:a800:1 --cpus-per-task=8 --mem=64G --time=02:00:00 --pty bash
#
#   2. 进入计算节点 shell 后，再手动运行：
#        cd /gpfs/share/home/2300012116/Antibody/FLAb
#        bash debug_interactive.sh env
#        bash debug_interactive.sh data
#        bash debug_interactive.sh split
#        bash debug_interactive.sh esm-smoke
#
#   3. 确认上面都通过后，再选择性跑完整步骤：
#        bash debug_interactive.sh embed
#        LOSS_NAMES="ranknet" bash debug_interactive.sh train
#
# 常用环境变量：
#   CONDA_ENV_NAME=esm2
#   CUDA_MODULE=cuda/12.6
#   DATA_DIR=data/binding
#   CACHE_DIR=cache/embeddings
#   OUTPUT_DIR=results/affinity_model
#   LOSS_NAMES="ranknet"              # train/all 默认只跑 ranknet，方便 debug
#   ALLOW_NO_SLURM=1                  # 特殊情况下允许非 Slurm allocation 运行

set -euo pipefail

trap 'echo "[ERROR] 第 ${LINENO} 行失败，退出码=$?；请查看上方日志。" >&2' ERR

ACTION="${1:-help}"
ENV_NAME="${CONDA_ENV_NAME:-esm2}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.6}"
DATA_DIR="${DATA_DIR:-data/binding}"
CACHE_DIR="${CACHE_DIR:-cache/embeddings}"
OUTPUT_DIR="${OUTPUT_DIR:-results/affinity_model}"
LOSS_NAMES="${LOSS_NAMES:-ranknet}"
MODULE_INIT="${MODULE_INIT:-/gpfs/share/software/module/tools/modules/init/profile.sh}"
CONDA_INIT="${CONDA_INIT:-/gpfs/share/software/anaconda/3-2023.09-0/etc/profile.d/conda.sh}"

print_usage() {
    cat <<'EOF'
用法：
  bash debug_interactive.sh <action>

action：
  env         只检查 module/conda/python/torch/CUDA
  data        读取并过滤 FLAb binding 数据，检查 data_loader
  split       不提 embedding，用假 embedding 检查 flatten_datasets/split_by_group
  esm-smoke   加载 ESM2-650M，对一条短序列提 embedding，检查模型下载/缓存/GPU
  embed       跑完整 embedding 阶段：python train.py --mode embed
  train       从缓存训练：python train.py --mode train
  all         跑完整流程：python train.py --mode all
  help        显示本帮助

推荐顺序：
  bash debug_interactive.sh env
  bash debug_interactive.sh data
  bash debug_interactive.sh split
  bash debug_interactive.sh esm-smoke
  bash debug_interactive.sh embed
  LOSS_NAMES="ranknet" bash debug_interactive.sh train
EOF
}

print_section() {
    echo ""
    echo "========================================"
    echo "$1"
    echo "========================================"
}

locate_project_dir() {
    local here
    here="$(pwd)"

    if [[ -n "${FLAB_PROJECT_DIR:-}" ]]; then
        PROJECT_DIR="${FLAB_PROJECT_DIR}"
    elif [[ -f "${here}/train.py" && -d "${here}/affinity_model" ]]; then
        PROJECT_DIR="${here}"
    elif [[ -f "${here}/FLAb/train.py" && -d "${here}/FLAb/affinity_model" ]]; then
        PROJECT_DIR="${here}/FLAb"
    else
        echo "[FATAL] 找不到 FLAb/train.py。请 cd 到 FLAb 目录，或设置 FLAB_PROJECT_DIR。" >&2
        exit 2
    fi

    cd "${PROJECT_DIR}"
    mkdir -p logs "${CACHE_DIR}" "${OUTPUT_DIR}" cache/huggingface cache/torch
}

guard_slurm_allocation() {
    if [[ -z "${SLURM_JOB_ID:-}" && "${ALLOW_NO_SLURM:-0}" != "1" ]]; then
        echo "[FATAL] 当前不在 Slurm allocation 中。" >&2
        echo "请先在 login 节点申请交互式 GPU 资源，例如：" >&2
        echo "  srun --partition=GPUA800 --gres=gpu:a800:1 --cpus-per-task=8 --mem=64G --time=02:00:00 --pty bash" >&2
        echo "如果你非常确定不是在 login 节点跑计算，可临时设置 ALLOW_NO_SLURM=1。" >&2
        exit 3
    fi
}

setup_environment() {
    locate_project_dir
    guard_slurm_allocation

    print_section "基本信息"
    echo "Project dir:   $(pwd)"
    echo "Action:        ${ACTION}"
    echo "Job ID:        ${SLURM_JOB_ID:-none}"
    echo "Node:          ${SLURM_NODELIST:-$(hostname)}"
    echo "Conda env:     ${ENV_NAME}"
    echo "CUDA module:   ${CUDA_MODULE}"
    echo "Data dir:      ${DATA_DIR}"
    echo "Cache dir:     ${CACHE_DIR}"
    echo "Output dir:    ${OUTPUT_DIR}"
    echo "Loss names:    ${LOSS_NAMES}"

    if [[ ! -d "${DATA_DIR}" ]]; then
        echo "[FATAL] 数据目录不存在: ${DATA_DIR}" >&2
        exit 4
    fi

    echo "Binding files: $(find "${DATA_DIR}" -maxdepth 1 \( -name '*.csv' -o -name '*.csv.zip' \) | wc -l)"

    print_section "加载 module / CUDA"
    if [[ ! -f "${MODULE_INIT}" ]]; then
        echo "[FATAL] 找不到 module 初始化脚本: ${MODULE_INIT}" >&2
        exit 5
    fi
    source "${MODULE_INIT}"

    if ! command -v module >/dev/null 2>&1; then
        echo "[FATAL] module 命令不可用，请检查 MODULE_INIT。" >&2
        exit 6
    fi

    module purge || true
    if [[ -n "${CUDA_MODULE}" ]]; then
        module load "${CUDA_MODULE}"
    fi
    module list

    print_section "激活 conda"
    if [[ ! -f "${CONDA_INIT}" ]]; then
        echo "[FATAL] 找不到 conda 初始化脚本: ${CONDA_INIT}" >&2
        exit 7
    fi
    source "${CONDA_INIT}"

    if ! command -v conda >/dev/null 2>&1; then
        echo "[FATAL] batch/interactive shell 里找不到 conda，请检查 CONDA_INIT。" >&2
        exit 8
    fi

    conda activate "${ENV_NAME}"

    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONIOENCODING=utf-8
    export TOKENIZERS_PARALLELISM=false
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
    export HF_HOME="${PROJECT_DIR}/cache/huggingface"
    export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
    export TORCH_HOME="${PROJECT_DIR}/cache/torch"
    mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${TORCH_HOME}"

    echo "python:       $(command -v python)"
    echo "CONDA_PREFIX: ${CONDA_PREFIX}"
    conda info --envs

    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi
    fi
}

check_python_env() {
    print_section "Python / CUDA / 依赖自检"
    python - <<'PY'
import sys
import importlib.util

required = ["torch", "transformers", "numpy", "pandas", "scipy", "sklearn", "tqdm"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(
        f"缺少 Python 包: {missing}。请在当前 conda 环境里安装 FLAb/requirements.txt。"
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
    raise SystemExit("torch.cuda.is_available() 是 False。请检查 GPU allocation、CUDA module 和 torch CUDA 版本。")

print(f"GPU:          {torch.cuda.get_device_name(0)}")
PY
}

check_data_loader() {
    print_section "data_loader 数据读取检查"
    python - <<PY
from affinity_model.data_loader import load_all_datasets

datasets = load_all_datasets("${DATA_DIR}")
if not datasets:
    raise SystemExit("没有加载到任何合格数据集")

total_rows = sum(len(df) for df in datasets.values())
print(f"loaded_groups: {len(datasets)}")
print(f"loaded_rows:   {total_rows:,}")

sizes = sorted(((name, len(df)) for name, df in datasets.items()), key=lambda x: x[1])
print("\\nsmallest groups:")
for name, n in sizes[:5]:
    print(f"  {name}: {n}")
print("\\nlargest groups:")
for name, n in sizes[-5:]:
    print(f"  {name}: {n}")

first_name, first_df = next(iter(datasets.items()))
cols = [
    "sequence", "label_raw", "label", "label_z", "label_rank",
    "compatible_group", "assay_family", "assay_units",
]
print(f"\\nfirst group: {first_name}")
print(first_df[[c for c in cols if c in first_df.columns]].head(3).to_string(index=False))
PY
}

check_group_split() {
    print_section "split_by_group 划分检查（不跑 ESM2，用假 embedding）"
    python - <<PY
import numpy as np

from affinity_model.config import cfg
from affinity_model.data_loader import load_all_datasets
from affinity_model.trainer import flatten_datasets, split_by_group

datasets = load_all_datasets("${DATA_DIR}")
embedded = {}
for name, df in datasets.items():
    piece = df.copy()
    piece["embedding"] = [np.zeros(cfg.ESM_EMBEDDING_DIM, dtype=np.float32) for _ in range(len(piece))]
    embedded[name] = piece

all_df = flatten_datasets(embedded)
train_df, val_df, test_df = split_by_group(all_df)

splits = {"train": train_df, "val": val_df, "test": test_df}
group_sets = {name: set(df[cfg.GROUP_COL].astype(str).unique()) for name, df in splits.items()}

print("\\nrows/groups:")
for name, df in splits.items():
    print(f"  {name}: rows={len(df):,}, groups={len(group_sets[name])}")

overlaps = {
    "train&val": group_sets["train"] & group_sets["val"],
    "train&test": group_sets["train"] & group_sets["test"],
    "val&test": group_sets["val"] & group_sets["test"],
}
print("\\noverlap check:")
for name, groups in overlaps.items():
    print(f"  {name}: {len(groups)}")
    if groups:
        raise SystemExit(f"{name} 出现 group 泄漏: {sorted(groups)[:5]}")
PY
}

check_esm_smoke() {
    print_section "ESM2 单序列 smoke test"
    python - <<'PY'
from affinity_model.config import cfg
from affinity_model.embeddings import embed_sequence

seq = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAR"
print(f"device: {cfg.DEVICE}")
print(f"model:  {cfg.ESM_MODEL_NAME}")
emb = embed_sequence(seq)
print(f"embedding_shape: {emb.shape}")
print(f"embedding_mean:  {float(emb.mean()):.6f}")
print(f"embedding_std:   {float(emb.std()):.6f}")
PY
}

run_train_entry() {
    local mode="$1"
    read -r -a LOSS_ARGS <<< "${LOSS_NAMES}"

    print_section "运行 train.py --mode ${mode}"
    python train.py \
        --mode "${mode}" \
        --data_dir "${DATA_DIR}" \
        --cache_dir "${CACHE_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --loss "${LOSS_ARGS[@]}"
}

case "${ACTION}" in
    help|-h|--help)
        print_usage
        ;;
    env)
        setup_environment
        check_python_env
        ;;
    data)
        setup_environment
        check_python_env
        check_data_loader
        ;;
    split)
        setup_environment
        check_python_env
        check_group_split
        ;;
    esm-smoke)
        setup_environment
        check_python_env
        check_esm_smoke
        ;;
    embed)
        setup_environment
        check_python_env
        run_train_entry embed
        ;;
    train)
        setup_environment
        check_python_env
        run_train_entry train
        ;;
    all)
        setup_environment
        check_python_env
        run_train_entry all
        ;;
    *)
        echo "[FATAL] 未知 action: ${ACTION}" >&2
        print_usage >&2
        exit 9
        ;;
esac
