#!/usr/bin/env bash
# hie2023efficient/CoV2Beta_REGN10987_Kd
# Run from AffinityTransformer repo root.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT"

SRC="data/binding/hie2023efficient_CoV2Beta_REGN10987_Kd.csv"
OUT="processed/binding/hie2023efficient/CoV2Beta_REGN10987_Kd"

echo "[$(date -u +%H:%M:%S)] Converting $SRC ..."
python3 "scripts/prepare/binding/hie2023efficient/CoV2Beta_REGN10987_Kd/convert.py" "$SRC" "$OUT"

echo "[$(date -u +%H:%M:%S)] Validating schema ..."
python3 scripts/prepare/validate_processed_table.py "$OUT/records.parquet"

echo "Done: $OUT"
