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
CONFIG_DIR="${CONFIG_DIR:-runs/v065/configs}"

python scripts/embeddings/build_v065_cache_and_configs.py \
  --split-dir "${SPLIT_DIR}" \
  --revision-file "${REVISION_FILE}" \
  --cache-root "${CACHE_ROOT}" \
  --config-dir "${CONFIG_DIR}" \
  --device cuda \
  --dtype "${EMBEDDING_DTYPE:-float16}" \
  --antibody-max-length "${ANTIBODY_MAX_LENGTH:-256}" \
  --antigen-max-length "${ANTIGEN_MAX_LENGTH:-512}" \
  --antibody-shard-size "${ANTIBODY_SHARD_SIZE:-32}" \
  --antigen-shard-size "${ANTIGEN_SHARD_SIZE:-4}" \
  --d-model "${INTERACTION_D_MODEL:-256}" \
  --num-heads "${INTERACTION_NUM_HEADS:-8}" \
  --batch-size "${TRAIN_BATCH_SIZE:-2}" \
  --epochs "${TRAIN_EPOCHS:-10}" \
  --lr "${TRAIN_LR:-1e-4}" \
  --max-pairs-per-group "${MAX_PAIRS_PER_GROUP:-200}" \
  --seed "${TRAIN_SEED:-0}"

echo "generated configs:"
find "${CONFIG_DIR}" -maxdepth 1 -type f -name 'v065_*_ranknet.yaml' -print | sort
echo "end=$(date)"
