#!/usr/bin/env bash
# Shared helpers for formal group-holdout control experiments.
#
# Source this file from a submit_* script. It centralizes the boring but
# important parts: model-cache checks, split/cache submission, and one-GPU
# training job submission. The experiment scripts remain small enough to audit.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "[FATAL] source this file from a group-holdout submit script; do not run it directly." >&2
  exit 64
fi

affinity_group_holdout_init() {
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  cd "${ROOT}"
  mkdir -p logs/slurm

  REVISION_FILE="${REVISION_FILE:-cache/model_revisions_v065.yaml}"
  SPLIT_DIR="${SPLIT_DIR:-processed/binding/splits/g00_max_antigen_context}"
  CACHE_ROOT="${CACHE_ROOT:-processed/embeddings/v065}"
  CONFIG_DIR="${CONFIG_DIR:-configs/v065}"
  RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
  TRAIN_TIME_LIMIT="${TRAIN_TIME_LIMIT:-72:00:00}"
  TRAIN_MEMORY="${TRAIN_MEMORY:-96G}"
}

affinity_group_holdout_require_files() {
  local path
  for path in "$@"; do
    if [[ ! -f "${path}" ]]; then
      echo "[FATAL] Missing required file: ${path}" >&2
      exit 3
    fi
  done
}

affinity_group_holdout_check_revision_file() {
  if [[ ! -f "${REVISION_FILE}" ]]; then
    echo "[FATAL] Missing ${REVISION_FILE}." >&2
    echo "Run: bash scripts/slurm/download_v065_models_login.sh" >&2
    exit 2
  fi
}

affinity_group_holdout_check_existing_split() {
  local split
  for split in train valid test; do
    if [[ ! -f "${SPLIT_DIR}/${split}.parquet" ]]; then
      echo "[FATAL] SKIP_G00=1 but missing ${SPLIT_DIR}/${split}.parquet" >&2
      exit 3
    fi
  done
}

affinity_group_holdout_submit_prerequisites() {
  affinity_group_holdout_check_revision_file

  if [[ "${SKIP_G00:-0}" == "1" ]]; then
    affinity_group_holdout_check_existing_split
  fi

  if [[ "${SKIP_SETUP:-0}" == "1" ]]; then
    echo "[SETUP] skipping login-node env setup (SKIP_SETUP=1)"
  else
    echo "[SETUP] Creating conda env on login node..."
    bash scripts/slurm/setup_affitest_env.sh
    echo "[SETUP] Done."
  fi

  GROUP_HOLDOUT_SMOKE_JOB="$(sbatch --parsable \
    scripts/slurm/smoke_test.sbatch)"
  echo "submitted code smoke: ${GROUP_HOLDOUT_SMOKE_JOB}"

  GROUP_HOLDOUT_MODELS_JOB="$(sbatch --parsable \
    --export="ALL,REVISION_FILE=${REVISION_FILE}" \
    scripts/slurm/check_v065_models.sbatch)"
  echo "submitted model cache check: ${GROUP_HOLDOUT_MODELS_JOB}"

  if [[ "${SKIP_G00:-0}" == "1" ]]; then
    GROUP_HOLDOUT_G00_DEPENDENCY="${GROUP_HOLDOUT_SMOKE_JOB}"
    echo "using existing split directory: ${SPLIT_DIR}"
  else
    GROUP_HOLDOUT_G00_JOB="$(sbatch --parsable \
      --dependency="afterok:${GROUP_HOLDOUT_SMOKE_JOB}" \
      scripts/slurm/g00_qc_and_splits.sbatch)"
    GROUP_HOLDOUT_G00_DEPENDENCY="${GROUP_HOLDOUT_G00_JOB}"
    echo "submitted g00: ${GROUP_HOLDOUT_G00_JOB}"
  fi

  GROUP_HOLDOUT_CACHE_JOB="$(sbatch --parsable \
    --dependency="afterok:${GROUP_HOLDOUT_G00_DEPENDENCY}:${GROUP_HOLDOUT_MODELS_JOB}" \
    --export="ALL,SPLIT_DIR=${SPLIT_DIR},REVISION_FILE=${REVISION_FILE},CACHE_ROOT=${CACHE_ROOT}" \
    scripts/slurm/build_v065_embedding_cache.sbatch)"
  echo "submitted formal embedding cache: ${GROUP_HOLDOUT_CACHE_JOB}"
}

affinity_group_holdout_submit_training() {
  local job_prefix="$1"
  local name="$2"
  local config="$3"
  local output_dir="$4"
  local dependency="$5"

  sbatch --parsable \
    --dependency="afterok:${dependency}" \
    --job-name="${job_prefix}-${name}" \
    --time="${TRAIN_TIME_LIMIT}" \
    --mem="${TRAIN_MEMORY}" \
    --export="ALL,CONFIG=${config},OUTPUT_DIR=${output_dir}" \
    scripts/slurm/run_config.sbatch
}

affinity_group_holdout_print_footer() {
  local title="$1"
  local output_root="$2"
  local report_path="$3"

  echo ""
  echo "${title} submitted"
  echo "  setup=$([[ "${SKIP_SETUP:-0}" == "1" ]] && echo skipped || echo login-node)"
  echo "  smoke=${GROUP_HOLDOUT_SMOKE_JOB}"
  echo "  models=${GROUP_HOLDOUT_MODELS_JOB}"
  echo "  cache=${GROUP_HOLDOUT_CACHE_JOB}"
  echo "  outputs=${output_root}"
  echo ""
  echo "Check progress:        squeue -u \$USER"
  echo "Once jobs finish, collect metrics.json into one CSV:"
  echo "  python scripts/experiments/collect_results.py \\"
  echo "    --output-root ${output_root} \\"
  echo "    --output ${report_path}"
}
