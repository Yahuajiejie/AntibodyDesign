#!/usr/bin/env bash
# Prepare engelhart2022dataset/scFv-SARS-CoV-2_affinity
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT"
SRC="data/binding/engelhart2022dataset_scFv-SARS-CoV-2_affinity.csv.zip"
OUT="processed/binding/engelhart2022dataset/scFv-SARS-CoV-2_affinity"
python3 "scripts/prepare/binding/engelhart2022dataset/scFv-SARS-CoV-2_affinity/convert.py" "$SRC" "$OUT"
python3 scripts/prepare/validate_processed_table.py "$OUT/records.parquet"
echo "Done: $OUT"
