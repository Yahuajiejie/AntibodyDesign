#!/usr/bin/env bash
# Submit randomized BST no-redundancy group-holdout controls.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/group_holdout_submit_common.sh"
affinity_group_holdout_init

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/slurm/group-holdout-randomized-bst-${RUN_TAG}}"
REPORT_PATH="${REPORT_PATH:-reports/experiments/group_holdout_randomized_bst-${RUN_TAG}.csv}"

configs=(
  "${CONFIG_DIR}/v065_concat_randomized_bst_no_redundancy.yaml"
  "${CONFIG_DIR}/v065_deep4_randomized_bst_no_redundancy.yaml"
  "${CONFIG_DIR}/v065_deep8_randomized_bst_no_redundancy.yaml"
  "${CONFIG_DIR}/v065_deep16_randomized_bst_no_redundancy.yaml"
)
affinity_group_holdout_require_files "${configs[@]}"
affinity_group_holdout_submit_prerequisites

concat="$(affinity_group_holdout_submit_training aff-gh-rbst concat "${configs[0]}" "${OUTPUT_ROOT}/concat" "${GROUP_HOLDOUT_CACHE_JOB}")"
deep4="$(affinity_group_holdout_submit_training aff-gh-rbst deep4 "${configs[1]}" "${OUTPUT_ROOT}/deep4" "${GROUP_HOLDOUT_CACHE_JOB}")"
deep8="$(affinity_group_holdout_submit_training aff-gh-rbst deep8 "${configs[2]}" "${OUTPUT_ROOT}/deep8" "${GROUP_HOLDOUT_CACHE_JOB}")"
deep16="$(affinity_group_holdout_submit_training aff-gh-rbst deep16 "${configs[3]}" "${OUTPUT_ROOT}/deep16" "${GROUP_HOLDOUT_CACHE_JOB}")"

echo "submitted concat: ${concat}"
echo "submitted deep4: ${deep4}"
echo "submitted deep8: ${deep8}"
echo "submitted deep16: ${deep16}"

affinity_group_holdout_print_footer "group-holdout randomized BST controls" "${OUTPUT_ROOT}" "${REPORT_PATH}"
