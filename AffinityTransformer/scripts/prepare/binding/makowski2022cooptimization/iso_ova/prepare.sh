#!/usr/bin/env bash
# Prepare makowski2022cooptimization/iso_ova
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT"
SRC="data/binding/makowski2022cooptimization_iso_ova.csv"
OUT="processed/binding/makowski2022cooptimization/iso_ova"
python3 "scripts/prepare/binding/makowski2022cooptimization/iso_ova/convert.py" "$SRC" "$OUT"
python3 scripts/prepare/validate_processed_table.py "$OUT/records.parquet"
echo "Done: $OUT"
