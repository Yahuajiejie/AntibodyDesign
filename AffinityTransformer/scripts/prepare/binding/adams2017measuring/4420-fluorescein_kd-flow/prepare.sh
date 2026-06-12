#!/usr/bin/env bash
# Prepare adams2017measuring/4420-fluorescein_kd-flow
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT"
SRC="data/binding/adams2017measuring_4420-fluorescein_kd-flow.csv"
OUT="processed/binding/adams2017measuring/4420-fluorescein_kd-flow"
python3 "scripts/prepare/binding/adams2017measuring/4420-fluorescein_kd-flow/convert.py" "$SRC" "$OUT"
python3 scripts/prepare/validate_processed_table.py "$OUT/records.parquet"
echo "Done: $OUT"
