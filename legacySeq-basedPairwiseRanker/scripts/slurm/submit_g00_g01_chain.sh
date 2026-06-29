#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs/slurm

g00_job_id="$(sbatch --parsable scripts/slurm/g00_qc_and_splits.sbatch)"
echo "submitted g00: ${g00_job_id}"

g01_job_id="$(
  sbatch --parsable \
    --dependency="afterok:${g00_job_id}" \
    --job-name=aff-g01-core \
    --export=ALL,GROUP_SCRIPT=scripts/runs/g01_core_ablation.sh \
    scripts/slurm/run_group.sbatch
)"
echo "submitted g01 after g00: ${g01_job_id}"
