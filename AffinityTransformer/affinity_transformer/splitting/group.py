"""Group holdout strategy split and group K-fold."""
from __future__ import annotations

import random

import pandas as pd

from .common import _partition_weighted_units, _rows_for_values
from .results import GroupFold


def build_group_kfolds(
    records: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> list[GroupFold]:
    """Partition records into deterministic, size-balanced group folds.

    Every ``group_id`` is assigned to exactly one validation fold.  The
    greedy assignment balances record counts while the seeded tie ordering
    prevents input row order from affecting the result.
    """
    required = ("record_id", "group_id", "dataset_id")
    missing = [column for column in required if column not in records.columns]
    if missing:
        raise ValueError(f"records is missing required column(s): {missing}")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if records.empty:
        raise ValueError("records must be non-empty")
    if records["record_id"].isna().any() or records["group_id"].isna().any():
        raise ValueError("records contains null record_id or group_id values")
    record_ids = records["record_id"].astype(str)
    if record_ids.duplicated().any():
        duplicated = record_ids[record_ids.duplicated()].tolist()
        raise ValueError(f"records contains duplicate record_id values: {duplicated[:10]}")

    group_sizes = records.groupby(records["group_id"].astype(str), sort=True).size().to_dict()
    if len(group_sizes) < n_splits:
        raise ValueError(
            f"n_splits={n_splits} exceeds the number of groups={len(group_sizes)}"
        )

    groups = list(group_sizes)
    random.Random(seed).shuffle(groups)
    tie_order = {group_id: index for index, group_id in enumerate(groups)}
    groups.sort(key=lambda group_id: (-group_sizes[group_id], tie_order[group_id]))

    fold_groups: list[set[str]] = [set() for _ in range(n_splits)]
    fold_sizes = [0] * n_splits
    for group_id in groups:
        fold_index = min(range(n_splits), key=lambda index: (fold_sizes[index], index))
        fold_groups[fold_index].add(group_id)
        fold_sizes[fold_index] += int(group_sizes[group_id])

    all_groups = set(group_sizes)
    folds: list[GroupFold] = []
    for index, valid_groups in enumerate(fold_groups):
        train_groups = all_groups - valid_groups
        train = _rows_for_values(records, "group_id", train_groups)
        valid = _rows_for_values(records, "group_id", valid_groups)
        if train.empty or valid.empty:
            raise ValueError(f"fold {index} produced an empty train or validation split")
        if set(train["group_id"].astype(str)) & set(valid["group_id"].astype(str)):
            raise ValueError(f"group leakage detected in fold {index}")
        folds.append(GroupFold(index=index, train=train, valid=valid))
    return folds


def _split_by_group(
    records: pd.DataFrame,
    valid_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_sizes = (
        records.assign(_group_id_str=records["group_id"].astype(str))
        .groupby("_group_id_str", sort=True)
        .size()
        .astype(int)
        .to_dict()
    )
    train_groups, valid_groups, test_groups = _partition_weighted_units(
        group_sizes, valid_fraction, test_fraction, seed
    )
    return (
        _rows_for_values(records, "group_id", train_groups),
        _rows_for_values(records, "group_id", valid_groups),
        _rows_for_values(records, "group_id", test_groups),
    )
