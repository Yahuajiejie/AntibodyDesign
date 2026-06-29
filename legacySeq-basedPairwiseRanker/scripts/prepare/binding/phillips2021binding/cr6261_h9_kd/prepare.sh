#!/usr/bin/env bash
# phillips2021binding/cr6261_h9_kd
# Run from AffinityTransformer repo root.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT"

SRC="data/binding/phillips2021binding_cr6261_h9_kd.csv"
OUT="processed/binding/phillips2021binding/cr6261_h9_kd"

echo "[$(date -u +%H:%M:%S)] Converting $SRC ..."
python3 "scripts/prepare/binding/phillips2021binding/cr6261_h9_kd/convert.py" "$SRC" "$OUT"

echo "[$(date -u +%H:%M:%S)] Validating schema ..."
python3 scripts/prepare/validate_processed_table.py "$OUT/records.parquet"

echo "Done: $OUT"
