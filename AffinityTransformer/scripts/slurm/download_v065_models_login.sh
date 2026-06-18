#!/usr/bin/env bash
# Run on the login node before submit_v065_training_chain.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/slurm_common.sh"

affinity_resolve_project_dir
LOAD_CUDA=0 affinity_load_modules
affinity_activate_conda

export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=0
REVISION_FILE="${REVISION_FILE:-cache/model_revisions_v065.yaml}"
mkdir -p "$(dirname "${REVISION_FILE}")"

echo "HF_HOME=${HF_HOME}"
echo "revision_file=${REVISION_FILE}"

REVISION_FILE="${REVISION_FILE}" python - <<'PY'
import gc
import os
from pathlib import Path

import torch
import yaml
from huggingface_hub import HfApi
from transformers import AutoModel, AutoTokenizer

models = {
    "antibody": os.environ.get("IGBERT_MODEL", "Exscientia/IgBert"),
    "antigen": os.environ.get("ESM2_MODEL", "facebook/esm2_t12_35M_UR50D"),
}
revision_overrides = {
    "antibody": os.environ.get("IGBERT_REVISION"),
    "antigen": os.environ.get("ESM2_REVISION"),
}
api = HfApi()
result = {}
for role, repo in models.items():
    revision = revision_overrides[role] or api.model_info(repo_id=repo).sha
    if not revision or revision.lower() in {"main", "master", "latest"}:
        raise SystemExit(f"failed to resolve immutable revision for {repo}: {revision!r}")
    print(f"downloading {role}: {repo}@{revision}")
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    model = AutoModel.from_pretrained(repo, revision=revision)
    tokenizer_revision = tokenizer.init_kwargs.get("_commit_hash") or revision
    result[role] = {
        "model_name": repo,
        "model_revision": revision,
        "tokenizer_revision": tokenizer_revision,
    }
    del tokenizer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

path = Path(os.environ["REVISION_FILE"])
path.write_text(yaml.safe_dump(result, sort_keys=False), encoding="utf-8")
print(path.read_text(encoding="utf-8"))
print("v0.65 model download complete")
PY
