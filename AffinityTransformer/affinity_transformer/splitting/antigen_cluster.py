"""antigen_cluster_holdout_split strategy."""
from __future__ import annotations

import pandas as pd

from .common import _partition_weighted_units, _rows_for_values


def _split_by_antigen_cluster(
    records: pd.DataFrame,
    valid_fraction: float,
    test_fraction: float,
    seed: int,
    antigen_clusters: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Like `_split_by_group`, but partitions by `antigen_cluster_id`
    instead of the exact `group_id` -- so near-duplicate antigens under
    different `antigen_key` names (e.g. point-mutant variants) stay on the
    same side of the split (see `antigen_clustering.py`).
    """
    required = ("antigen_key", "antigen_cluster_id")
    missing = [column for column in required if column not in antigen_clusters.columns]
    if missing:
        raise ValueError(f"antigen_clusters is missing required column(s): {missing}")

    record_keys = set(records["antigen_key"].astype(str))
    cluster_keys = set(antigen_clusters["antigen_key"].astype(str))
    unmapped = record_keys - cluster_keys
    if unmapped:
        raise ValueError(
            f"records contains antigen_key values missing from antigen_clusters: "
            f"{sorted(unmapped)[:10]}"
        )

    cluster_map = dict(zip(
        antigen_clusters["antigen_key"].astype(str),
        antigen_clusters["antigen_cluster_id"].astype(str),
    ))
    working = records.assign(
        _antigen_cluster_id=records["antigen_key"].astype(str).map(cluster_map)
    )
    cluster_sizes = (
        working.groupby("_antigen_cluster_id", sort=True).size().astype(int).to_dict()
    )
    train_clusters, valid_clusters, test_clusters = _partition_weighted_units(
        cluster_sizes, valid_fraction, test_fraction, seed
    )
    # Deliberately keep `_antigen_cluster_id` on the returned frames (unlike
    # most other split helpers' temp columns) so `_build_leakage_report` can
    # run its own independent overlap check on it, the same way
    # `group_id_overlap` double-checks `_split_by_group`'s partition.
    # `build_splits` drops it after that check passes.
    return (
        _rows_for_values(working, "_antigen_cluster_id", train_clusters),
        _rows_for_values(working, "_antigen_cluster_id", valid_clusters),
        _rows_for_values(working, "_antigen_cluster_id", test_clusters),
    )
