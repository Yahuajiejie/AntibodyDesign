#!/usr/bin/env bash
# shanehsazzadeh2023unlocking/kd_hher2_mab
# Run from AffinityTransformer repo root.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT"

SRC="data/binding/shanehsazzadeh2023unlocking_kd_hher2_mab.csv"
OUT="processed/binding/shanehsazzadeh2023unlocking/kd_hher2_mab"

echo "[$(date -u +%H:%M:%S)] Converting $SRC ..."
python3 "scripts/prepare/binding/shanehsazzadeh2023unlocking/kd_hher2_mab/convert.py" "$SRC" "$OUT"

echo "[$(date -u +%H:%M:%S)] Validating schema ..."
python3 scripts/prepare/validate_processed_table.py "$OUT/records.parquet"

echo "Done: $OUT"
