#!/usr/bin/env bash
# Prepare shanker2024unsupervised/SA58-BA.1_IC50
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT"
SRC="data/binding/shanker2024unsupervised_SA58-BA.1_IC50.csv"
OUT="processed/binding/shanker2024unsupervised/SA58-BA.1_IC50"
python3 "scripts/prepare/binding/shanker2024unsupervised/SA58-BA.1_IC50/convert.py" "$SRC" "$OUT"
python3 scripts/prepare/validate_processed_table.py "$OUT/records.parquet"
echo "Done: $OUT"
