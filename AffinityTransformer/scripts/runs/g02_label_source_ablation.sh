#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/g02_label_source_ablation}"

"$PYTHON_BIN" scripts/experiments/run_many.py \
  --configs \
    configs/experiments/g02_experimental_only_antibody_only.yaml \
    configs/experiments/g02_experimental_only_concat_antigen.yaml \
    configs/experiments/g02_experimental_only_cross_attention.yaml \
    configs/experiments/g02_all_label_kinds_cross_attention.yaml \
    configs/experiments/g02_no_predicted_cross_attention.yaml \
  --output-root "$OUTPUT_ROOT" \
  --python "$PYTHON_BIN"

"$PYTHON_BIN" scripts/experiments/collect_results.py \
  --output-root "$OUTPUT_ROOT" \
  --output reports/experiments/g02_label_source_ablation_metrics.csv

echo "g02 complete: ${OUTPUT_ROOT}"
