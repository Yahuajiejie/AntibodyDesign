#!/usr/bin/env bash
# Prepare makowski2022cooptimization/igg_ant
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT"
SRC="data/binding/makowski2022cooptimization_igg_ant.csv"
OUT="processed/binding/makowski2022cooptimization/igg_ant"
python3 "scripts/prepare/binding/makowski2022cooptimization/igg_ant/convert.py" "$SRC" "$OUT"
python3 scripts/prepare/validate_processed_table.py "$OUT/records.parquet"
echo "Done: $OUT"
