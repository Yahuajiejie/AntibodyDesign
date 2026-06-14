#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/g04_antigen_subset_ablation}"

CONFIGS=()
for name in g04_cov2_rbd g04_influenza_ha g04_lysozyme g04_no_abrank g04_vegf; do
  if [[ -f "processed/binding/splits/${name}/train.parquet" ]]; then
    CONFIGS+=("configs/experiments/${name}_cross_attention.yaml")
  else
    echo "skip ${name}: missing processed/binding/splits/${name}/train.parquet"
  fi
done

if [[ "${#CONFIGS[@]}" -eq 0 ]]; then
  echo "no g04 split is available; run scripts/runs/g00_qc_and_splits.sh first"
  exit 1
fi

"$PYTHON_BIN" scripts/experiments/run_many.py \
  --configs "${CONFIGS[@]}" \
  --output-root "$OUTPUT_ROOT" \
  --python "$PYTHON_BIN"

"$PYTHON_BIN" scripts/experiments/collect_results.py \
  --output-root "$OUTPUT_ROOT" \
  --output reports/experiments/g04_antigen_subset_ablation_metrics.csv

echo "g04 complete: ${OUTPUT_ROOT}"
