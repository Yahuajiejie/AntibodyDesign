"""Dedicated sampler for binary or exactly two-label groups."""

from __future__ import annotations

import random

import pandas as pd

from .common import _canonical_pair, _pair_row, _pair_sample_count
from .labels import _label_to_record_ids


def _sample_two_label_group_pairs(
    group_id: str,
    group: pd.DataFrame,
    n_candidates: int,
    max_pairs_per_group: int,
    seed: int,
    pair_sample_strategy: str,
    pair_fraction: float | None,
    min_pairs_per_group: int,
) -> list[dict[str, object]]:
    label_to_ids = _label_to_record_ids(group)
    if len(label_to_ids) != 2:
        return []

    labels = sorted(label_to_ids)
    left_label, right_label = labels[0], labels[1]
    left_ids = label_to_ids[left_label]
    right_ids = label_to_ids[right_label]
    target = _pair_sample_count(
        n_candidates,
        max_pairs_per_group=max_pairs_per_group,
        pair_sample_strategy=pair_sample_strategy,
        pair_fraction=pair_fraction,
        min_pairs_per_group=min_pairs_per_group,
    )

    rng = random.Random(f"{seed}:{group_id}:two_label")
    seen: set[tuple[str, str]] = set()
    rows: list[dict[str, object]] = []
    max_attempts = max(1000, target * 100)
    attempts = 0
    while len(rows) < target and attempts < max_attempts:
        attempts += 1
        record_id_a = rng.choice(left_ids)
        record_id_b = rng.choice(right_ids)
        record_id_i, label_i, record_id_j, label_j = _canonical_pair(
            record_id_a, left_label, record_id_b, right_label
        )
        key = (record_id_i, record_id_j)
        if key in seen:
            continue
        seen.add(key)
        y_ij = 1.0 if label_i > label_j else 0.0
        rows.append(_pair_row(group_id, record_id_i, record_id_j, label_i, label_j, y_ij))

    rows.sort(key=lambda row: str(row["pair_id"]))
    return rows
