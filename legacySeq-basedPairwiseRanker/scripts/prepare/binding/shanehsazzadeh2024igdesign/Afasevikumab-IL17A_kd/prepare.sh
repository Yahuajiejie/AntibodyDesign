#!/usr/bin/env bash
# shanehsazzadeh2024igdesign/Afasevikumab-IL17A_kd
# Run from AffinityTransformer repo root.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT"

SRC="data/binding/shanehsazzadeh2024igdesign_Afasevikumab-IL17A_kd.csv"
OUT="processed/binding/shanehsazzadeh2024igdesign/Afasevikumab-IL17A_kd"

echo "[$(date -u +%H:%M:%S)] Converting $SRC ..."
python3 "scripts/prepare/binding/shanehsazzadeh2024igdesign/Afasevikumab-IL17A_kd/convert.py" "$SRC" "$OUT"

echo "[$(date -u +%H:%M:%S)] Validating schema ..."
python3 scripts/prepare/validate_processed_table.py "$OUT/records.parquet"

echo "Done: $OUT"
