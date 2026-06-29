#!/usr/bin/env bash
# Compatibility wrapper. Prefer:
#   bash scripts/slurm/submit_group_holdout_randomized_bst.sh
set -euo pipefail

echo "[WARN] submit_randomized_bst_no_redundancy.sh is deprecated; forwarding to submit_group_holdout_randomized_bst.sh" >&2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/submit_group_holdout_randomized_bst.sh" "$@"
