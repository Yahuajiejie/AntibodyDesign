#!/usr/bin/env bash
# Submit all currently defined group-holdout controls from one shared split/cache.
#
# This is the "press once and walk away" script. It submits setup/smoke/g00/cache
# once, then fans out all training jobs that depend on the shared embedding cache.
# It can use many GPUs at once. Disable families with SKIP_RANKNET=1,
# SKIP_RANDOMIZED_BST=1, SKIP_NOISE_AWARE=1, or SKIP_BALANCED_TREE=1.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/group_holdout_submit_common.sh"
affinity_group_holdout_init

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/slurm/group-holdout-controls-${RUN_TAG}}"
REPORT_PATH="${REPORT_PATH:-reports/experiments/group_holdout_controls-${RUN_TAG}.csv}"

ranknet_configs=(
  "${CONFIG_DIR}/v065_concat_ranknet.yaml"
  "${CONFIG_DIR}/v065_deep4_ranknet.yaml"
  "${CONFIG_DIR}/v065_deep8_ranknet.yaml"
  "${CONFIG_DIR}/v065_deep16_ranknet.yaml"
)
randomized_bst_configs=(
  "${CONFIG_DIR}/v065_concat_randomized_bst_no_redundancy.yaml"
  "${CONFIG_DIR}/v065_deep4_randomized_bst_no_redundancy.yaml"
  "${CONFIG_DIR}/v065_deep8_randomized_bst_no_redundancy.yaml"
  "${CONFIG_DIR}/v065_deep16_randomized_bst_no_redundancy.yaml"
)
noise_aware_configs=(
  "${CONFIG_DIR}/v065_concat_noise_aware_multiscale.yaml"
  "${CONFIG_DIR}/v065_deep4_noise_aware_multiscale.yaml"
  "${CONFIG_DIR}/v065_deep8_noise_aware_multiscale.yaml"
  "${CONFIG_DIR}/v065_deep16_noise_aware_multiscale.yaml"
)
balanced_tree_config="${CONFIG_DIR}/v065_deep4_balanced_tree.yaml"

affinity_group_holdout_require_files \
  "${ranknet_configs[@]}" \
  "${randomized_bst_configs[@]}" \
  "${noise_aware_configs[@]}" \
  "${balanced_tree_config}"

affinity_group_holdout_submit_prerequisites

submit_depth_family() {
  local family="$1"
  local job_prefix="$2"
  local skip_value="$3"
  shift 3
  local configs=("$@")
  local names=(concat deep4 deep8 deep16)
  local index
  local job

  if [[ "${skip_value}" == "1" ]]; then
    echo "skipping ${family}"
    return
  fi

  for index in "${!names[@]}"; do
    job="$(affinity_group_holdout_submit_training \
      "${job_prefix}" \
      "${names[index]}" \
      "${configs[index]}" \
      "${OUTPUT_ROOT}/${family}/${names[index]}" \
      "${GROUP_HOLDOUT_CACHE_JOB}")"
    echo "submitted ${family}/${names[index]}: ${job}"
  done
}

submit_depth_family ranknet aff-gh-ranknet "${SKIP_RANKNET:-0}" "${ranknet_configs[@]}"
submit_depth_family randomized-bst aff-gh-rbst "${SKIP_RANDOMIZED_BST:-0}" "${randomized_bst_configs[@]}"
submit_depth_family noise-aware-multiscale aff-gh-nam "${SKIP_NOISE_AWARE:-0}" "${noise_aware_configs[@]}"

if [[ "${SKIP_BALANCED_TREE:-0}" == "1" ]]; then
  echo "skipping balanced-tree"
else
  balanced_tree_job="$(affinity_group_holdout_submit_training \
    aff-gh-btree \
    deep4 \
    "${balanced_tree_config}" \
    "${OUTPUT_ROOT}/balanced-tree/deep4" \
    "${GROUP_HOLDOUT_CACHE_JOB}")"
  echo "submitted balanced-tree/deep4: ${balanced_tree_job}"
fi

affinity_group_holdout_print_footer "all group-holdout controls" "${OUTPUT_ROOT}" "${REPORT_PATH}"
