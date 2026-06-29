#!/usr/bin/env bash
# Prepare tsuruta2024sarscov2/binary
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT"
SRC="data/binding/tsuruta2024sarscov2_binary.csv.zip"
OUT="processed/binding/tsuruta2024sarscov2/binary"
python3 "scripts/prepare/binding/tsuruta2024sarscov2/binary/convert.py" "$SRC" "$OUT"
python3 scripts/prepare/validate_processed_table.py "$OUT/records.parquet"
echo "Done: $OUT"
