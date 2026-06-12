#!/usr/bin/env bash
# Prepare kirby2024retrospective/ab-SARSCoV2_kd
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT"
SRC="data/binding/kirby2024retrospective_ab-SARSCoV2_kd.csv"
OUT="processed/binding/kirby2024retrospective/ab-SARSCoV2_kd"
python3 "scripts/prepare/binding/kirby2024retrospective/ab-SARSCoV2_kd/convert.py" "$SRC" "$OUT"
python3 scripts/prepare/validate_processed_table.py "$OUT/records.parquet"
echo "Done: $OUT"
