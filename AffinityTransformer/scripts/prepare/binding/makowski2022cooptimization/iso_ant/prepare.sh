#!/usr/bin/env bash
# Prepare makowski2022cooptimization/iso_ant
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT"
SRC="data/binding/makowksi2022cooptimization_iso_ant.csv"
OUT="processed/binding/makowski2022cooptimization/iso_ant"
python3 "scripts/prepare/binding/makowski2022cooptimization/iso_ant/convert.py" "$SRC" "$OUT"
python3 scripts/prepare/validate_processed_table.py "$OUT/records.parquet"
echo "Done: $OUT"
