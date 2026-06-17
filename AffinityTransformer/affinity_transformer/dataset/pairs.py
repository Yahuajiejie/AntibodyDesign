"""Public pairwise ranking pair construction API."""

from __future__ import annotations

import itertools
import random

import pandas as pd

from .pair_sampling import (
    _candidate_pair_count,
    _pair_row,
    _pair_sample_count,
    _sample_large_group_pairs,
    _should_enumerate_pairs,
    _validate_pair_sampling,
)
from .records import filter_trainable_records
from .schema import (
    PAIR_COLUMNS,
    _DEFAULT_DISCRETE_LABEL_RATIO_THRESHOLD,
    _DEFAULT_DISCRETE_LABEL_UNIQUE_THRESHOLD,
    _DEFAULT_INTRA_BLOCK_PAIRS_PER_LARGE_GROUP,
    _DEFAULT_LABEL_BLOCK_COUNT,
    _DEFAULT_LARGE_GROUP_THRESHOLD,
    _DEFAULT_PAIR_ENUMERATION_LIMIT,
)


def _candidate_pairs(group: pd.DataFrame) -> list[tuple[str, str, float, float, float]]:
    """Enumerate valid unordered candidate pairs within one group."""
    items = sorted(
        zip(group["record_id"].astype(str), group["rank_label"].astype(float)),
        key=lambda item: item[0],
    )

    pairs: list[tuple[str, str, float, float, float]] = []
    for (record_id_i, label_i), (record_id_j, label_j) in itertools.combinations(items, 2):
        if label_i == label_j:
            continue
        y_ij = 1.0 if label_i > label_j else 0.0
        pairs.append((record_id_i, record_id_j, label_i, label_j, y_ij))
    return pairs


def build_pairs(
    records: pd.DataFrame,
    max_pairs_per_group: int,
    seed: int,
    pair_sample_strategy: str = "absolute_cap",
    pair_fraction: float | None = None,
    min_pairs_per_group: int = 1,
    large_group_threshold: int = _DEFAULT_LARGE_GROUP_THRESHOLD,
    pair_enumeration_limit: int = _DEFAULT_PAIR_ENUMERATION_LIMIT,
    label_block_count: int = _DEFAULT_LABEL_BLOCK_COUNT,
    intra_block_pairs_per_large_group: int = _DEFAULT_INTRA_BLOCK_PAIRS_PER_LARGE_GROUP,
    discrete_label_unique_threshold: int = _DEFAULT_DISCRETE_LABEL_UNIQUE_THRESHOLD,
    discrete_label_ratio_threshold: float = _DEFAULT_DISCRETE_LABEL_RATIO_THRESHOLD,
) -> pd.DataFrame:
    """Build pairwise ranking examples within each group."""
    required = ("record_id", "group_id", "rank_label", "label_kind", "keep_for_training")
    missing = [c for c in required if c not in records.columns]
    if missing:
        raise ValueError(f"records is missing required column(s): {missing}")
    _validate_pair_sampling(
        max_pairs_per_group,
        pair_sample_strategy,
        pair_fraction,
        min_pairs_per_group,
        large_group_threshold,
        pair_enumeration_limit,
        label_block_count,
        intra_block_pairs_per_large_group,
        discrete_label_unique_threshold,
        discrete_label_ratio_threshold,
    )

    trainable = filter_trainable_records(records)

    rows: list[dict[str, object]] = []
    for group_id, group in trainable.groupby("group_id", sort=True):
        n_candidates = _candidate_pair_count(group)
        if n_candidates == 0:
            continue

        if _should_enumerate_pairs(group, n_candidates, large_group_threshold, pair_enumeration_limit):
            candidates = _candidate_pairs(group)
            if not candidates:
                continue

            n_sample = _pair_sample_count(
                len(candidates),
                max_pairs_per_group=max_pairs_per_group,
                pair_sample_strategy=pair_sample_strategy,
                pair_fraction=pair_fraction,
                min_pairs_per_group=min_pairs_per_group,
            )
            if len(candidates) > n_sample:
                rng = random.Random(f"{seed}:{group_id}")
                candidates = rng.sample(candidates, n_sample)
                candidates.sort(key=lambda c: (c[0], c[1]))

            for record_id_i, record_id_j, label_i, label_j, y_ij in candidates:
                rows.append(_pair_row(group_id, record_id_i, record_id_j, label_i, label_j, y_ij))
            continue

        rows.extend(
            _sample_large_group_pairs(
                str(group_id),
                group,
                n_candidates=n_candidates,
                max_pairs_per_group=max_pairs_per_group,
                seed=seed,
                pair_sample_strategy=pair_sample_strategy,
                pair_fraction=pair_fraction,
                min_pairs_per_group=min_pairs_per_group,
                label_block_count=label_block_count,
                intra_block_pairs_per_large_group=intra_block_pairs_per_large_group,
                discrete_label_unique_threshold=discrete_label_unique_threshold,
                discrete_label_ratio_threshold=discrete_label_ratio_threshold,
            )
        )

    if not rows:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)
