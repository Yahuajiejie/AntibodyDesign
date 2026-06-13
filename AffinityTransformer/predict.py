#!/usr/bin/env python3
"""Competition/user prediction entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from affinity_transformer.user_entry import rank_antibody_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="best", help="Model name from configs/model_registry.yaml")
    parser.add_argument("--input", required=True, type=Path, help="Input CSV/TSV table")
    parser.add_argument("--output", required=True, type=Path, help="Output rankings CSV")
    parser.add_argument(
        "--format",
        choices=("auto", "csv", "tsv"),
        default="auto",
        help="Input format. 'auto' infers from file suffix.",
    )
    args = parser.parse_args()

    input_table = _read_table(args.input, args.format)
    rankings = rank_antibody_table(input_table, model_name=args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rankings.to_csv(args.output, index=False)
    print(f"wrote {len(rankings)} ranked row(s) -> {args.output}")


def _read_table(path: Path, file_format: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input table not found: {path}")
    if file_format == "auto":
        if path.suffix.lower() == ".tsv":
            file_format = "tsv"
        else:
            file_format = "csv"
    sep = "\t" if file_format == "tsv" else ","
    return pd.read_csv(path, sep=sep, keep_default_na=True)


if __name__ == "__main__":
    main()
