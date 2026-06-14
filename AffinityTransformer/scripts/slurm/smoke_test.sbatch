#!/usr/bin/env bash
#SBATCH --job-name=aff-smoke
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --partition=GPUA800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

set -euo pipefail

LOAD_CUDA=1
source scripts/slurm/activate_affitest.sh
affinity_check_gpu_runtime

python -m pytest -q
