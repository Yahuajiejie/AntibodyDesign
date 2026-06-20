#!/usr/bin/env bash
# Submit the randomized_bst (add_redundancy_edges=false) pair-sampling run,
# frozen-cache deep4 model, on real data.
#
# By default this submits ONLY the new randomized_bst_no_redundancy config --
# set RUN_BASELINE=1 if you ever want the original (capped_proportional)
# config submitted alongside it for a fresh side-by-side run (not needed if
# you already have those numbers from an earlier run).
#
#   randomized_bst_no_redundancy:   configs/v065/v065_deep4_randomized_bst_no_redundancy.yaml
#                                    pair_sample_strategy: randomized_bst
#                                    add_redundancy_edges: false
#   original (optional, RUN_BASELINE=1):
#                                    configs/v065/v065_deep4_ranknet.yaml
#                                    pair_sample_strategy: capped_proportional
#
# Either way this reuses the existing IgBert/ESM2 frozen embedding cache and
# train/valid/test split (SKIP_G00=1) -- the cache and split only depend on
# antibody_encoder/antigen_encoder/data split, not on pair_sample_strategy,
# so nothing needs to be rebuilt for the new config.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
mkdir -p logs/slurm

REVISION_FILE="${REVISION_FILE:-cache/model_revisions_v065.yaml}"
SPLIT_DIR="${SPLIT_DIR:-processed/binding/splits/g00_max_antigen_context}"
CACHE_ROOT="${CACHE_ROOT:-processed/embeddings/v065}"
CONFIG_DIR="${CONFIG_DIR:-configs/v065}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/slurm/randomized-bst-no-redundancy-ablation-${RUN_TAG}}"

RUN_BASELINE="${RUN_BASELINE:-0}"
BASELINE_CONFIG="${CONFIG_DIR}/v065_deep4_ranknet.yaml"
ABLATION_CONFIG="${CONFIG_DIR}/v065_deep4_randomized_bst_no_redundancy.yaml"

if [[ ! -f "${REVISION_FILE}" ]]; then
  echo "[FATAL] Missing ${REVISION_FILE}." >&2
  echo "Run: bash scripts/slurm/download_v065_models_login.sh" >&2
  exit 2
fi

configs_to_check=("${ABLATION_CONFIG}")
if [[ "${RUN_BASELINE}" == "1" ]]; then
  configs_to_check+=("${BASELINE_CONFIG}")
fi
for config in "${configs_to_check[@]}"; do
  if [[ ! -f "${config}" ]]; then
    echo "[FATAL] Missing training config: ${config}" >&2
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
    --job-name="aff-ablation-${name}" \
    --time="${TRAIN_TIME_LIMIT:-72:00:00}" \
    --mem="${TRAIN_MEMORY:-96G}" \
    --export="ALL,CONFIG=${config},OUTPUT_DIR=${output_dir}" \
    scripts/slurm/run_config.sbatch
}

# Independent of any baseline run -- depends only on the shared cache.
randomized_bst_no_redundancy="$(submit_training randomized_bst_no_redundancy "${ABLATION_CONFIG}" "${cache}")"
echo "submitted randomized_bst_no_redundancy after cache: ${randomized_bst_no_redundancy}"

job_ids="${randomized_bst_no_redundancy}"
if [[ "${RUN_BASELINE}" == "1" ]]; then
  original="$(submit_training original "${BASELINE_CONFIG}" "${cache}")"
  echo "submitted original (capped_proportional) after cache: ${original}"
  job_ids="${job_ids},${original}"
fi

echo ""
echo "randomized_bst submitted"
echo "  setup=login-node (synchronous)"
echo "  smoke=${smoke}"
echo "  models=${models}"
echo "  cache=${cache}"
echo "  randomized_bst_no_redundancy=${randomized_bst_no_redundancy}  -> ${OUTPUT_ROOT}/randomized_bst_no_redundancy"
if [[ "${RUN_BASELINE}" == "1" ]]; then
  echo "  original=${original}                       -> ${OUTPUT_ROOT}/original  (RUN_BASELINE=1)"
fi
echo "  outputs=${OUTPUT_ROOT}"
echo ""
echo "Check progress:        squeue -j ${job_ids}"
echo "Once it finishes, collect metrics.json into a CSV (add your existing"
echo "original run's metrics.json under the same output-root, or merge the"
echo "two CSVs afterward, to compare against the original numbers you"
echo "already have):"
echo "  python scripts/experiments/collect_results.py \\"
echo "    --output-root ${OUTPUT_ROOT} \\"
echo "    --output reports/experiments/randomized_bst_no_redundancy-${RUN_TAG}.csv"
