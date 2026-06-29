#!/usr/bin/env bash
#SBATCH --job-name=aff-group
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --partition=GPUA800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00

set -euo pipefail

LOAD_CUDA=1
source scripts/slurm/activate_affitest.sh
affinity_check_gpu_runtime

GROUP_SCRIPT="${GROUP_SCRIPT:-scripts/runs/g01_core_ablation.sh}"
if [[ ! -f "${GROUP_SCRIPT}" ]]; then
  echo "[FATAL] Missing group script: ${GROUP_SCRIPT}" >&2
  exit 8
fi

GROUP_NAME="$(basename "$GROUP_SCRIPT" .sh)"
export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/slurm/${GROUP_NAME}-${SLURM_JOB_ID:-manual}}"

echo "group_script=${GROUP_SCRIPT}"
echo "output_root=${OUTPUT_ROOT}"

bash "$GROUP_SCRIPT"

echo "end=$(date)"
