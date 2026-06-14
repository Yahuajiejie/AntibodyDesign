#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/g03_pair_sampling_ablation}"

"$PYTHON_BIN" scripts/experiments/run_many.py \
  --configs \
    configs/experiments/g03_cross_attention_pairs_abs50.yaml \
    configs/experiments/g03_cross_attention_pairs_abs200.yaml \
    configs/experiments/g03_cross_attention_pairs_abs500.yaml \
    configs/experiments/g03_cross_attention_pairs_prop1_cap500.yaml \
    configs/experiments/g03_cross_attention_pairs_prop5_cap500.yaml \
  --output-root "$OUTPUT_ROOT" \
  --python "$PYTHON_BIN"

"$PYTHON_BIN" scripts/experiments/collect_results.py \
  --output-root "$OUTPUT_ROOT" \
  --output reports/experiments/g03_pair_sampling_ablation_metrics.csv

echo "g03 complete: ${OUTPUT_ROOT}"
