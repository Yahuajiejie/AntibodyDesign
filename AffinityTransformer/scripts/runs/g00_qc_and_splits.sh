#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT="${1:-processed/binding/all_records.parquet}"
VALID_FRACTION="${VALID_FRACTION:-0.1}"
TEST_FRACTION="${TEST_FRACTION:-0.1}"
SEED="${SEED:-0}"

mkdir -p reports/data processed/binding/filtered processed/binding/splits

"$PYTHON_BIN" scripts/data/inspect_records.py \
  --input "$INPUT" \
  --output reports/data/g00_all_records_full_qc.csv \
  --by-dataset reports/data/g00_all_records_by_dataset.csv

filter_and_split_required() {
  local name="$1"
  local filter_config="$2"
  local filtered="processed/binding/filtered/${name}/all_records.parquet"
  local summary="reports/data/${name}_filter_qc.csv"
  local split_dir="processed/binding/splits/${name}"

  "$PYTHON_BIN" scripts/data/filter_records.py \
    --input "$INPUT" \
    --filter-config "$filter_config" \
    --output "$filtered" \
    --summary "$summary"

  "$PYTHON_BIN" scripts/data/build_splits.py \
    --input "$filtered" \
    --strategy group_holdout_split \
    --output-dir "$split_dir" \
    --valid-fraction "$VALID_FRACTION" \
    --test-fraction "$TEST_FRACTION" \
    --seed "$SEED"
}

filter_and_split_optional() {
  local name="$1"
  local filter_config="$2"
  local filtered="processed/binding/filtered/${name}/all_records.parquet"
  local summary="reports/data/${name}_filter_qc.csv"
  local split_dir="processed/binding/splits/${name}"

  if ! "$PYTHON_BIN" scripts/data/filter_records.py \
    --input "$INPUT" \
    --filter-config "$filter_config" \
    --output "$filtered" \
    --summary "$summary"; then
    echo "optional split skipped at filter stage: ${name}"
    return 0
  fi

  if ! "$PYTHON_BIN" scripts/data/build_splits.py \
    --input "$filtered" \
    --strategy group_holdout_split \
    --output-dir "$split_dir" \
    --valid-fraction "$VALID_FRACTION" \
    --test-fraction "$TEST_FRACTION" \
    --seed "$SEED"; then
    echo "optional split skipped at split stage: ${name}"
    return 0
  fi
}

filter_and_split_required g00_max_antigen_context configs/filters/g00_max_antigen_context.yaml
filter_and_split_required g02_experimental_only configs/filters/g02_experimental_only.yaml
filter_and_split_required g02_no_predicted configs/filters/g02_no_predicted.yaml

filter_and_split_optional g04_cov2_rbd configs/filters/g04_cov2_rbd.yaml
filter_and_split_optional g04_influenza_ha configs/filters/g04_influenza_ha.yaml
filter_and_split_optional g04_lysozyme configs/filters/g04_lysozyme.yaml
filter_and_split_optional g04_no_abrank configs/filters/g04_no_abrank.yaml
filter_and_split_optional g04_vegf configs/filters/g04_vegf.yaml

echo "g00 complete: QC reports in reports/data, splits in processed/binding/splits"
