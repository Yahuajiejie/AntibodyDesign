#!/usr/bin/env bash
# Compatibility wrapper. Prefer:
#   bash scripts/slurm/submit_group_holdout_noise_aware_multiscale.sh
set -euo pipefail

echo "[WARN] submit_noise_aware_multiscale.sh is deprecated; forwarding to submit_group_holdout_noise_aware_multiscale.sh" >&2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/submit_group_holdout_noise_aware_multiscale.sh" "$@"
