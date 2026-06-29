#!/usr/bin/env bash
# Prepare shanehsazzadeh2023unlocking/adcc_ec50
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT"
SRC="data/binding/shanehsazzadeh2023unlocking_adcc_ec50.csv"
OUT="processed/binding/shanehsazzadeh2023unlocking/adcc_ec50"
python3 "scripts/prepare/binding/shanehsazzadeh2023unlocking/adcc_ec50/convert.py" "$SRC" "$OUT"
python3 scripts/prepare/validate_processed_table.py "$OUT/records.parquet"
echo "Done: $OUT"
