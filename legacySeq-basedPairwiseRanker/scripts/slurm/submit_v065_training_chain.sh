#!/usr/bin/env bash
# Compatibility wrapper. Prefer:
#   bash scripts/slurm/submit_group_holdout_ranknet.sh
set -euo pipefail

echo "[WARN] submit_v065_training_chain.sh is deprecated; forwarding to submit_group_holdout_ranknet.sh" >&2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/submit_group_holdout_ranknet.sh" "$@"
