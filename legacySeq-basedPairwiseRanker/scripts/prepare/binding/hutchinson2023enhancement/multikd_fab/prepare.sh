#!/usr/bin/env bash
# hutchinson2023enhancement/multikd_fab
# Run from AffinityTransformer repo root.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT"

SRC="data/binding/hutchinson2023enhancement_multikd_fab.csv"
OUT="processed/binding/hutchinson2023enhancement/multikd_fab"

echo "[$(date -u +%H:%M:%S)] Converting $SRC ..."
python3 "scripts/prepare/binding/hutchinson2023enhancement/multikd_fab/convert.py" "$SRC" "$OUT"

echo "[$(date -u +%H:%M:%S)] Validating schema ..."
python3 scripts/prepare/validate_processed_table.py "$OUT/records.parquet"

echo "Done: $OUT"
