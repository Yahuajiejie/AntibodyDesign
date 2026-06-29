#!/usr/bin/env bash
#SBATCH --job-name=aff-v065-cache
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

SPLIT_DIR="${SPLIT_DIR:-processed/binding/splits/g00_max_antigen_context}"
REVISION_FILE="${REVISION_FILE:-cache/model_revisions_v065.yaml}"
CACHE_ROOT="${CACHE_ROOT:-processed/embeddings/v065}"
python scripts/embeddings/build_v065_cache.py \
  --split-dir "${SPLIT_DIR}" \
  --revision-file "${REVISION_FILE}" \
  --cache-root "${CACHE_ROOT}" \
  --device cuda \
  --dtype "${EMBEDDING_DTYPE:-float16}" \
  --antibody-max-length "${ANTIBODY_MAX_LENGTH:-256}" \
  --antigen-max-length "${ANTIGEN_MAX_LENGTH:-512}" \
  --antibody-shard-size "${ANTIBODY_SHARD_SIZE:-32}" \
  --antigen-shard-size "${ANTIGEN_SHARD_SIZE:-4}"
echo "end=$(date)"
