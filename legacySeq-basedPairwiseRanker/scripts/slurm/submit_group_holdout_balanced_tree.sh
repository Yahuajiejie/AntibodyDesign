#!/usr/bin/env bash
# Submit the currently available balanced-tree group-holdout control.
#
# At the moment this family has only a deep4 config. Keep it separate so it is
# obvious in result tables that it is a sampler ablation, not a full depth sweep.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/group_holdout_submit_common.sh"
affinity_group_holdout_init

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/slurm/group-holdout-balanced-tree-${RUN_TAG}}"
REPORT_PATH="${REPORT_PATH:-reports/experiments/group_holdout_balanced_tree-${RUN_TAG}.csv}"
config="${CONFIG_DIR}/v065_deep4_balanced_tree.yaml"

affinity_group_holdout_require_files "${config}"
affinity_group_holdout_submit_prerequisites

deep4="$(affinity_group_holdout_submit_training aff-gh-btree deep4 "${config}" "${OUTPUT_ROOT}/deep4" "${GROUP_HOLDOUT_CACHE_JOB}")"
echo "submitted deep4: ${deep4}"

affinity_group_holdout_print_footer "group-holdout balanced-tree control" "${OUTPUT_ROOT}" "${REPORT_PATH}"
