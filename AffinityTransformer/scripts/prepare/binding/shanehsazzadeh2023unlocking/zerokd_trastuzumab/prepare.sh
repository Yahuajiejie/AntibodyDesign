#!/usr/bin/env bash
# shanehsazzadeh2023unlocking/zerokd_trastuzumab
# Run from AffinityTransformer repo root.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT"

SRC="data/binding/shanehsazzadeh2023unlocking_zerokd_trastuzumab.csv"
OUT="processed/binding/shanehsazzadeh2023unlocking/zerokd_trastuzumab"

echo "[$(date -u +%H:%M:%S)] Converting $SRC ..."
python3 "scripts/prepare/binding/shanehsazzadeh2023unlocking/zerokd_trastuzumab/convert.py" "$SRC" "$OUT"

echo "[$(date -u +%H:%M:%S)] Validating schema ..."
python3 scripts/prepare/validate_processed_table.py "$OUT/records.parquet"

echo "Done: $OUT"
