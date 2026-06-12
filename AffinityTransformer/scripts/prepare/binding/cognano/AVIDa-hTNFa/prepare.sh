#!/usr/bin/env bash
# Prepare cognano/AVIDa-hTNFa
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT"
SRC="data/binding/cognano_AVIDa-hTNFa.csv"
OUT="processed/binding/cognano/AVIDa-hTNFa"
python3 "scripts/prepare/binding/cognano/AVIDa-hTNFa/convert.py" "$SRC" "$OUT"
python3 scripts/prepare/validate_processed_table.py "$OUT/records.parquet"
echo "Done: $OUT"
