#!/usr/bin/env bash
# Run this on the LOGIN NODE (has internet) before submitting the SLURM job chain.
# Downloads ESM2 weights into the project's local HuggingFace cache.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/slurm_common.sh"

affinity_resolve_project_dir
LOAD_CUDA=0 affinity_load_modules
affinity_activate_conda

# Override offline mode for this login-node download script
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=0

echo "HF_HOME=${HF_HOME}"

python - <<'PY'
from huggingface_hub import snapshot_download

repos = ["facebook/esm2_t12_35M_UR50D"]
for repo in repos:
    print(f"downloading {repo} ...")
    snapshot_path = snapshot_download(repo_id=repo)
    print(f"  snapshot={snapshot_path}")
    print(f"  ok: {repo}")
print("ESM2 download complete. You can now submit the SLURM job chain.")
PY
