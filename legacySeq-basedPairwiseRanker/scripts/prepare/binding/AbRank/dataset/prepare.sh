#!/usr/bin/env bash
# Prepare AbRank/dataset
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT"
SRC="data/binding/AbRank_dataset.csv.zip"
OUT="processed/binding/AbRank/dataset"
python3 "scripts/prepare/binding/AbRank/dataset/convert.py" "$SRC" "$OUT"
python3 scripts/prepare/validate_processed_table.py "$OUT/records.parquet"
echo "Done: $OUT"
