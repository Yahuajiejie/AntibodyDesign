"""Label-kind helpers for pair sampling."""

from __future__ import annotations

import pandas as pd

from ..schema import _BINARY_LABEL_KIND


def _is_discrete_label_group(
    group: pd.DataFrame,
    discrete_label_unique_threshold: int,
    discrete_label_ratio_threshold: float,
) -> bool:
    label_kind = group["label_kind"].astype(str).str.lower()
    if (label_kind == _BINARY_LABEL_KIND).any():
        return True
    labels = group["rank_label"].astype(float)
    n_records = len(labels)
    if n_records == 0:
        return False
    n_unique = int(labels.nunique(dropna=False))
    return (
        n_unique <= discrete_label_unique_threshold
        or n_unique / n_records <= discrete_label_ratio_threshold
    )


def _is_two_label_group(group: pd.DataFrame) -> bool:
    label_kind = group["label_kind"].astype(str).str.lower()
    n_unique = group["rank_label"].astype(float).nunique(dropna=False)
    if (label_kind == _BINARY_LABEL_KIND).any():
        if n_unique > 2:
            raise ValueError("binary label_kind groups must have at most two unique rank_label values")
        return n_unique == 2
    return n_unique == 2


def _label_to_record_ids(group: pd.DataFrame) -> dict[float, tuple[str, ...]]:
    label_to_ids: dict[float, list[str]] = {}
    for record_id, label in zip(group["record_id"].astype(str), group["rank_label"].astype(float)):
        label_to_ids.setdefault(float(label), []).append(record_id)
    return {
        label: tuple(sorted(record_ids))
        for label, record_ids in sorted(label_to_ids.items(), key=lambda item: item[0])
    }
