"""Common pair sampler primitives."""

from __future__ import annotations

import math
import random

import pandas as pd


def _pair_row(
    group_id: object,
    record_id_i: str,
    record_id_j: str,
    label_i: float,
    label_j: float,
    y_ij: float,
) -> dict[str, object]:
    return dict(
        pair_id=f"{record_id_i}::{record_id_j}",
        group_id=group_id,
        record_id_i=record_id_i,
        record_id_j=record_id_j,
        label_i=label_i,
        label_j=label_j,
        y_ij=y_ij,
    )


def _candidate_pair_count(group: pd.DataFrame) -> int:
    labels = group["rank_label"].astype(float)
    counts = labels.value_counts(dropna=False)
    n_records = int(counts.sum())
    total = n_records * (n_records - 1) // 2
    same_label = sum(int(count) * (int(count) - 1) // 2 for count in counts)
    return int(total - same_label)


def _should_enumerate_pairs(
    group: pd.DataFrame,
    n_candidates: int,
    large_group_threshold: int,
    pair_enumeration_limit: int,
) -> bool:
    return len(group) < large_group_threshold and n_candidates <= pair_enumeration_limit


def _canonical_pair(
    record_id_a: str,
    label_a: float,
    record_id_b: str,
    label_b: float,
) -> tuple[str, float, str, float]:
    if record_id_a <= record_id_b:
        return record_id_a, label_a, record_id_b, label_b
    return record_id_b, label_b, record_id_a, label_a


def _emit_pair(
    record_id_a: str,
    label_a: float,
    record_id_b: str,
    label_b: float,
    group_id: object,
    rows: list[dict[str, object]],
    seen: set[tuple[str, str]],
) -> None:
    """Append one canonicalized, deduplicated pair row, skipping ties.

    Shared by every tree-shaped pair builder (`tree.py`, `heap_tree.py`,
    `randomized_tree.py`): each decides which two records form an edge in
    its own structure, then hands them here to apply the same `y_ij`
    convention, `pair_id` canonicalization, and `seen`-based dedup that
    `build_pairs`'s other strategies already use.
    """
    if label_a == label_b:
        return
    record_id_i, label_i, record_id_j, label_j = _canonical_pair(
        record_id_a, label_a, record_id_b, label_b
    )
    key = (record_id_i, record_id_j)
    if key in seen:
        return
    seen.add(key)
    y_ij = 1.0 if label_i > label_j else 0.0
    rows.append(_pair_row(group_id, record_id_i, record_id_j, label_i, label_j, y_ij))


def _weighted_choice(items, rng: random.Random):
    total = sum(weight for _, weight in items)
    if total <= 0:
        raise ValueError("weighted choice requires at least one positive weight")
    threshold = rng.uniform(0, total)
    cumulative = 0.0
    for value, weight in items:
        cumulative += weight
        if threshold <= cumulative:
            return value
    return items[-1][0]


def _pair_sample_count(
    n_candidates: int,
    max_pairs_per_group: int,
    pair_sample_strategy: str,
    pair_fraction: float | None,
    min_pairs_per_group: int,
) -> int:
    if pair_sample_strategy == "absolute_cap":
        return min(n_candidates, max_pairs_per_group)

    assert pair_fraction is not None
    target = max(min_pairs_per_group, math.ceil(n_candidates * pair_fraction))
    return min(n_candidates, max_pairs_per_group, target)
