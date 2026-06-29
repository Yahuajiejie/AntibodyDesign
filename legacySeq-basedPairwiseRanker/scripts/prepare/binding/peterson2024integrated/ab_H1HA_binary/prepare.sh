#!/usr/bin/env bash
# Prepare peterson2024integrated/ab_H1HA_binary
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT"
SRC="data/binding/peterson2024integrated_ab_H1HA_binary.csv"
OUT="processed/binding/peterson2024integrated/ab_H1HA_binary"
python3 "scripts/prepare/binding/peterson2024integrated/ab_H1HA_binary/convert.py" "$SRC" "$OUT"
python3 scripts/prepare/validate_processed_table.py "$OUT/records.parquet"
echo "Done: $OUT"
