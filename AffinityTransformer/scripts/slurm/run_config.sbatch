#!/usr/bin/env bash
#SBATCH --job-name=aff-train
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --partition=GPUA800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

set -euo pipefail

LOAD_CUDA=1
source scripts/slurm/activate_affitest.sh
affinity_check_gpu_runtime

CONFIG="${CONFIG:-configs/experiments/g01_maxctx_cross_attention.yaml}"
if [[ ! -f "${CONFIG}" ]]; then
  echo "[FATAL] Missing config file: ${CONFIG}" >&2
  exit 8
fi

RUN_NAME="$(basename "$CONFIG" .yaml)"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/slurm/${RUN_NAME}-${SLURM_JOB_ID:-manual}}"

echo "config=${CONFIG}"
echo "output_dir=${OUTPUT_DIR}"

python train.py --config "$CONFIG" --output-dir "$OUTPUT_DIR"

echo "end=$(date)"
