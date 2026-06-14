#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/slurm_common.sh"

affinity_enable_error_trap
affinity_resolve_project_dir
affinity_load_modules
affinity_activate_conda
affinity_print_header
