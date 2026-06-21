#!/usr/bin/env bash
# Submit the frozen-cache Concat / Deep4 / Deep8 / Deep16 training jobs.
# All four depend only on the shared embedding cache (no weight transfer or
# other real dependency between them) and are submitted in parallel.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
mkdir -p logs/slurm

REVISION_FILE="${REVISION_FILE:-cache/model_revisions_v065.yaml}"
SPLIT_DIR="${SPLIT_DIR:-processed/binding/splits/g00_max_antigen_context}"
CACHE_ROOT="${CACHE_ROOT:-processed/embeddings/v065}"
CONFIG_DIR="${CONFIG_DIR:-configs/v065}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/slurm/v065-${RUN_TAG}}"

if [[ ! -f "${REVISION_FILE}" ]]; then
  echo "[FATAL] Missing ${REVISION_FILE}." >&2
  echo "Run: bash scripts/slurm/download_v065_models_login.sh" >&2
  exit 2
fi

for config in \
  "${CONFIG_DIR}/v065_concat_ranknet.yaml" \
  "${CONFIG_DIR}/v065_deep4_ranknet.yaml" \
  "${CONFIG_DIR}/v065_deep8_ranknet.yaml" \
  "${CONFIG_DIR}/v065_deep16_ranknet.yaml"; do
  if [[ ! -f "${config}" ]]; then
    echo "[FATAL] Missing human-owned training config: ${config}" >&2
    exit 3
  fi
done

if [[ "${SKIP_G00:-0}" == "1" ]]; then
  for split in train valid test; do
    if [[ ! -f "${SPLIT_DIR}/${split}.parquet" ]]; then
      echo "[FATAL] SKIP_G00=1 but missing ${SPLIT_DIR}/${split}.parquet" >&2
      exit 3
    fi
  done
fi

# Run env setup synchronously on the login node (compute nodes have no internet).
echo "[SETUP] Creating conda env on login node..."
bash scripts/slurm/setup_affitest_env.sh
echo "[SETUP] Done."

smoke="$(sbatch --parsable \
  scripts/slurm/smoke_test.sbatch)"
echo "submitted code smoke: ${smoke}"

models="$(sbatch --parsable \
  --export="ALL,REVISION_FILE=${REVISION_FILE}" \
  scripts/slurm/check_v065_models.sbatch)"
echo "submitted model cache check: ${models}"

if [[ "${SKIP_G00:-0}" == "1" ]]; then
  g00_dependency="${smoke}"
  echo "using existing split directory: ${SPLIT_DIR}"
else
  g00="$(sbatch --parsable \
    --dependency="afterok:${smoke}" \
    scripts/slurm/g00_qc_and_splits.sbatch)"
  g00_dependency="${g00}"
  echo "submitted g00: ${g00}"
fi

cache="$(sbatch --parsable \
  --dependency="afterok:${g00_dependency}:${models}" \
  --export="ALL,SPLIT_DIR=${SPLIT_DIR},REVISION_FILE=${REVISION_FILE},CACHE_ROOT=${CACHE_ROOT}" \
  scripts/slurm/build_v065_embedding_cache.sbatch)"
echo "submitted formal embedding cache: ${cache}"

submit_training() {
  local name="$1"
  local config="$2"
  local dependency="$3"
  local output_dir="${OUTPUT_ROOT}/${name}"
  sbatch --parsable \
    --dependency="afterok:${dependency}" \
    --job-name="aff-v065-${name}" \
    --time="${TRAIN_TIME_LIMIT:-72:00:00}" \
    --mem="${TRAIN_MEMORY:-96G}" \
    --export="ALL,CONFIG=${config},OUTPUT_DIR=${output_dir}" \
    scripts/slurm/run_config.sbatch
}

if [[ "${SKIP_CONCAT:-0}" == "1" ]]; then
  concat="skipped"
  echo "skipping concat (SKIP_CONCAT=1)"
else
  concat="$(submit_training concat "${CONFIG_DIR}/v065_concat_ranknet.yaml" "${cache}")"
  echo "submitted concat after cache: ${concat}"
fi

# deep4/deep8/deep16 each depend only on the shared cache, not on each
# other or on concat (no weight transfer between them) -- submitted in
# parallel rather than serialized.
deep4="$(submit_training deep4 "${CONFIG_DIR}/v065_deep4_ranknet.yaml" "${cache}")"
echo "submitted deep4 after cache: ${deep4}"

deep8="$(submit_training deep8 "${CONFIG_DIR}/v065_deep8_ranknet.yaml" "${cache}")"
echo "submitted deep8 after cache: ${deep8}"

deep16="$(submit_training deep16 "${CONFIG_DIR}/v065_deep16_ranknet.yaml" "${cache}")"
echo "submitted deep16 after cache: ${deep16}"

echo ""
echo "v0.65 chain submitted"
echo "  setup=login-node (synchronous)"
echo "  smoke=${smoke}"
echo "  models=${models}"
echo "  cache=${cache}"
echo "  concat=${concat}"
echo "  deep4=${deep4}"
echo "  deep8=${deep8}"
echo "  deep16=${deep16}"
echo "  outputs=${OUTPUT_ROOT}"
