#!/usr/bin/env python3
"""Filter a standard processed records table into a reproducible subset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from affinity_transformer.dataset import load_records
from affinity_transformer.record_filter import (
    filter_records,
    load_record_filter_config,
    write_filter_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input records.parquet/csv")
    parser.add_argument("--filter-config", required=True, type=Path, help="YAML filter config")
    parser.add_argument("--output", required=True, type=Path, help="Filtered .parquet or .csv output")
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Filter summary CSV. Defaults to <output stem>_summary.csv.",
    )
    args = parser.parse_args()

    records = load_records(args.input)
    config = load_record_filter_config(args.filter_config)
    filtered = filter_records(records, config)
    summary = args.summary or args.output.with_name(f"{args.output.stem}_summary.csv")
    write_filter_outputs(records, filtered, config, args.output, summary)

    print(f"input rows={len(records)}")
    print(f"filtered rows={len(filtered)} -> {args.output}")
    print(f"summary -> {summary}")


if __name__ == "__main__":
    main()
