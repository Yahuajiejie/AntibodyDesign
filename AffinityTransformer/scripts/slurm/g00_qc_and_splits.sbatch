#!/usr/bin/env bash
#SBATCH --job-name=aff-g00-qc
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --partition=C64M256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00

set -euo pipefail

LOAD_CUDA=0
source scripts/slurm/activate_affitest.sh
affinity_check_cpu_runtime

if [[ ! -f "${INPUT_RECORDS:-processed/binding/all_records.parquet}" ]]; then
  echo "[FATAL] Missing input records: ${INPUT_RECORDS:-processed/binding/all_records.parquet}" >&2
  exit 8
fi

bash scripts/runs/g00_qc_and_splits.sh "${INPUT_RECORDS:-processed/binding/all_records.parquet}"

echo "end=$(date)"
