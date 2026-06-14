#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/g01_core_ablation}"

"$PYTHON_BIN" scripts/experiments/run_many.py \
  --configs \
    configs/experiments/g01_maxctx_antibody_only.yaml \
    configs/experiments/g01_maxctx_concat_antigen.yaml \
    configs/experiments/g01_maxctx_cross_attention.yaml \
  --output-root "$OUTPUT_ROOT" \
  --python "$PYTHON_BIN"

"$PYTHON_BIN" scripts/experiments/collect_results.py \
  --output-root "$OUTPUT_ROOT" \
  --output reports/experiments/g01_core_ablation_metrics.csv

echo "g01 complete: ${OUTPUT_ROOT}"
