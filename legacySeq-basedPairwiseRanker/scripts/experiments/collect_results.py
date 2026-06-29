#!/usr/bin/env python3
"""Collect metrics.json files from a group output directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for metrics_path in sorted(args.output_root.glob("*/metrics.json")):
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        row: dict[str, object] = {
            "run_name": metrics_path.parent.name,
            "output_dir": str(metrics_path.parent),
        }
        row.update(metrics)
        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No metrics.json files found under {args.output_root}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"collected {len(rows)} run(s) -> {args.output}")


if __name__ == "__main__":
    main()
