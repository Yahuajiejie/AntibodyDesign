#!/usr/bin/env bash
#SBATCH --job-name=aff-setup
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --partition=C64M256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
LOAD_CUDA=1 bash scripts/slurm/setup_affitest_env.sh
