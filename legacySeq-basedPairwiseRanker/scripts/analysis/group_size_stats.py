#!/usr/bin/env python3
"""One-off diagnostic: group-size / candidate-pair distribution for a
processed split.

Used to set `pair_fraction` / `min_pairs_per_group` / `max_pairs_per_group`
for `capped_proportional` pair sampling (see `dataset/pair_sampling/common.py`
`_pair_sample_count`) based on the *actual* shape of the data, instead of
guessing. Pure pandas over the processed record table -- no model, no
embeddings, no GPU -- safe to run on the login node.

Usage:
    python scripts/analysis/group_size_stats.py \
        processed/binding/splits/g00_max_antigen_context/train.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from affinity_transformer.dataset.pair_sampling.common import _candidate_pair_count
from affinity_transformer.dataset.records import filter_trainable_records, load_records


def _group_sizes(path: Path) -> pd.DataFrame:
    records = load_records(path)
    trainable = filter_trainable_records(records)

    rows: list[dict[str, object]] = []
    for group_id, group in trainable.groupby("group_id", sort=False):
        if group["rank_label"].astype(float).nunique() < 2:
            continue  # not rankable -- same exclusion build_groups/build_pairs apply
        rows.append({
            "group_id": group_id,
            "n_records": len(group),
            "n_candidates": _candidate_pair_count(group),
        })
    return pd.DataFrame(rows)


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        raise SystemExit(f"usage: {argv[0]} <records.parquet|records.csv>")

    sizes = _group_sizes(Path(argv[1]))
    percentiles = [0.5, 0.75, 0.9, 0.95, 0.99]

    print(f"rankable groups: {len(sizes)}\n")
    print("--- n_records per group ---")
    print(sizes["n_records"].describe(percentiles=percentiles))
    print("\n--- n_candidates (raw, pre-cap) per group ---")
    print(sizes["n_candidates"].describe(percentiles=percentiles))
    print("\n--- 15 largest groups by n_records ---")
    print(
        sizes.sort_values("n_records", ascending=False)
        .head(15)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main(sys.argv)
