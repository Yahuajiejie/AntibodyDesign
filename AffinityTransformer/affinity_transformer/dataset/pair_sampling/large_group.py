"""Top-level large-group pair sampler."""

from __future__ import annotations

import pandas as pd

from .blocks import _build_label_blocks, _sample_from_block_pairs, _sample_within_blocks
from .common import _pair_sample_count
from .labels import _is_discrete_label_group, _is_two_label_group
from .two_label import _sample_two_label_group_pairs


def _sample_large_group_pairs(
    group_id: str,
    group: pd.DataFrame,
    n_candidates: int,
    max_pairs_per_group: int,
    seed: int,
    pair_sample_strategy: str,
    pair_fraction: float | None,
    min_pairs_per_group: int,
    label_block_count: int,
    intra_block_pairs_per_large_group: int,
    discrete_label_unique_threshold: int,
    discrete_label_ratio_threshold: float,
) -> list[dict[str, object]]:
    if _is_two_label_group(group):
        return _sample_two_label_group_pairs(
            group_id,
            group,
            n_candidates=n_candidates,
            max_pairs_per_group=max_pairs_per_group,
            seed=seed,
            pair_sample_strategy=pair_sample_strategy,
            pair_fraction=pair_fraction,
            min_pairs_per_group=min_pairs_per_group,
        )

    blocks = _build_label_blocks(group, label_block_count)
    use_label_buckets = _is_discrete_label_group(
        group,
        discrete_label_unique_threshold=discrete_label_unique_threshold,
        discrete_label_ratio_threshold=discrete_label_ratio_threshold,
    )
    seen: set[tuple[str, str]] = set()
    rows: list[dict[str, object]] = []

    inter_target = _pair_sample_count(
        n_candidates,
        max_pairs_per_group=max_pairs_per_group,
        pair_sample_strategy=pair_sample_strategy,
        pair_fraction=pair_fraction,
        min_pairs_per_group=min_pairs_per_group,
    )
    _sample_from_block_pairs(
        blocks,
        target=min(inter_target, n_candidates),
        seed=f"{seed}:{group_id}:inter",
        seen=seen,
        rows=rows,
        group_id=group_id,
        use_label_buckets=use_label_buckets,
    )

    remaining = max(0, n_candidates - len(seen))
    intra_target = min(intra_block_pairs_per_large_group, remaining)
    if intra_target > 0:
        _sample_within_blocks(
            blocks,
            target=intra_target,
            seed=f"{seed}:{group_id}:intra",
            seen=seen,
            rows=rows,
            group_id=group_id,
            use_label_buckets=use_label_buckets,
        )

    rows.sort(key=lambda row: str(row["pair_id"]))
    return rows
