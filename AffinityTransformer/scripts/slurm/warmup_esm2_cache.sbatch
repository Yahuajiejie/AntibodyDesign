#!/usr/bin/env bash
#SBATCH --job-name=aff-cache-esm2
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

python - <<'PY'
# Models must be pre-downloaded on the login node before submitting this job.
# Run scripts/slurm/download_esm2_login.sh on the login node first.
from transformers import AutoModel, AutoTokenizer

repos = ["facebook/esm2_t12_35M_UR50D"]
for repo in repos:
    print(f"loading {repo} from local cache")
    AutoTokenizer.from_pretrained(repo, local_files_only=True)
    AutoModel.from_pretrained(repo, local_files_only=True)
    print(f"  ok: {repo}")
print("ESM2 cache warmup complete")
PY
