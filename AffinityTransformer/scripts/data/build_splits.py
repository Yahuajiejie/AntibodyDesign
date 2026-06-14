#!/usr/bin/env python3
"""Build fixed train/valid/test splits from a processed records table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from affinity_transformer.dataset import load_records
from affinity_transformer.record_filter import (
    filter_records,
    load_record_filter_config,
    write_filter_outputs,
)
from affinity_transformer.splits import VALID_STRATEGIES, build_splits, write_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input records.parquet/csv")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for split files")
    parser.add_argument("--strategy", required=True, choices=sorted(VALID_STRATEGIES))
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--filter-config",
        type=Path,
        default=None,
        help="Optional YAML filter applied before splitting.",
    )
    args = parser.parse_args()

    records = load_records(args.input)
    if args.filter_config is not None:
        config = load_record_filter_config(args.filter_config)
        filtered = filter_records(records, config)
        if filtered.empty:
            raise ValueError(f"filter produced no rows: {args.filter_config}")
        write_filter_outputs(
            records,
            filtered,
            config,
            args.output_dir / "filtered_records.parquet",
            args.output_dir / "filter_summary.csv",
        )
        records = filtered

    result = build_splits(
        records,
        strategy=args.strategy,
        valid_fraction=args.valid_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    write_splits(result, args.output_dir)

    print(f"strategy={args.strategy}")
    print(f"train rows={len(result.train)}")
    print(f"valid rows={len(result.valid)}")
    print(f"test rows={len(result.test)}")
    print(f"split outputs -> {args.output_dir}")


if __name__ == "__main__":
    main()
