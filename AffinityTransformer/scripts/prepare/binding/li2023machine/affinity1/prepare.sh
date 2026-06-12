#!/usr/bin/env bash
# Prepare li2023machine/affinity1
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT"
SRC="data/binding/li2023machine_scFv-SARS-CoV-2_affinity1.csv.zip"
OUT="processed/binding/li2023machine/affinity1"
python3 "scripts/prepare/binding/li2023machine/affinity1/convert.py" "$SRC" "$OUT"
python3 scripts/prepare/validate_processed_table.py "$OUT/records.parquet"
echo "Done: $OUT"
