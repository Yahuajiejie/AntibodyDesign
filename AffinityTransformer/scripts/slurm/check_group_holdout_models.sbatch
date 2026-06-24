#!/usr/bin/env bash
#SBATCH --job-name=aff-gh-models
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --partition=C64M256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail

LOAD_CUDA=0
source scripts/slurm/activate_affitest.sh
affinity_check_cpu_runtime

REVISION_FILE="${REVISION_FILE:-cache/model_revisions_group_holdout.yaml}"
if [[ ! -f "${REVISION_FILE}" ]]; then
  echo "[FATAL] Missing ${REVISION_FILE}. Run scripts/slurm/download_group_holdout_models_login.sh first." >&2
  exit 8
fi

REVISION_FILE="${REVISION_FILE}" python - <<'PY'
import gc
import os
import yaml
from transformers import AutoModel, AutoTokenizer

with open(os.environ["REVISION_FILE"], encoding="utf-8") as handle:
    revisions = yaml.safe_load(handle)
for role in ("antibody", "antigen"):
    item = revisions[role]
    repo = item["model_name"]
    model_revision = item["model_revision"]
    tokenizer_revision = item["tokenizer_revision"]
    print(f"checking {role}: {repo}@{model_revision}")
    model = AutoModel.from_pretrained(
        repo, revision=model_revision, local_files_only=True
    )
    AutoTokenizer.from_pretrained(
        repo, revision=tokenizer_revision, local_files_only=True
    )
    del model
    gc.collect()
print("group-holdout model cache check complete")
PY
