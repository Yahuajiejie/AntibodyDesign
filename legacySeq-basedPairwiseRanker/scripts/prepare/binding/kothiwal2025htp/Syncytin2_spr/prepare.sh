#!/usr/bin/env bash
# Prepare kothiwal2025htp/Syncytin2_spr
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT"
SRC="data/binding/kothiwal2025htp_Syncytin2_spr.csv"
OUT="processed/binding/kothiwal2025htp/Syncytin2_spr"
python3 "scripts/prepare/binding/kothiwal2025htp/Syncytin2_spr/convert.py" "$SRC" "$OUT"
python3 scripts/prepare/validate_processed_table.py "$OUT/records.parquet"
echo "Done: $OUT"
