#!/usr/bin/env bash
# Prepare kothiwal2025htp/PDL2_ec50
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT"
SRC="data/binding/kothiwal2025htp_PDL2_ec50.csv"
OUT="processed/binding/kothiwal2025htp/PDL2_ec50"
python3 "scripts/prepare/binding/kothiwal2025htp/PDL2_ec50/convert.py" "$SRC" "$OUT"
python3 scripts/prepare/validate_processed_table.py "$OUT/records.parquet"
echo "Done: $OUT"
