"""Rank-label block sampling for large continuous or multi-label groups."""

from __future__ import annotations

import itertools
import random

import pandas as pd

from ..examples import _LabelBlock
from .common import _canonical_pair, _pair_row, _weighted_choice


def _build_label_blocks(group: pd.DataFrame, label_block_count: int) -> list[_LabelBlock]:
    items = sorted(
        zip(group["record_id"].astype(str), group["rank_label"].astype(float)),
        key=lambda item: (item[1], item[0]),
    )
    n_blocks = min(label_block_count, len(items))
    blocks: list[_LabelBlock] = []
    for block_index in range(n_blocks):
        start = block_index * len(items) // n_blocks
        end = (block_index + 1) * len(items) // n_blocks
        block_items = items[start:end]
        label_to_ids: dict[float, list[str]] = {}
        for record_id, label in block_items:
            label_to_ids.setdefault(label, []).append(record_id)
        blocks.append(
            _LabelBlock(
                index=block_index,
                items=tuple(block_items),
                label_to_ids={
                    label: tuple(sorted(record_ids))
                    for label, record_ids in sorted(label_to_ids.items(), key=lambda item: item[0])
                },
                n_records=end - start,
            )
        )
    return blocks


def _sample_from_block_pairs(
    blocks: list[_LabelBlock],
    target: int,
    seed: str,
    seen: set[tuple[str, str]],
    rows: list[dict[str, object]],
    group_id: str,
    use_label_buckets: bool,
) -> None:
    candidates: list[tuple[tuple[_LabelBlock, _LabelBlock], int]] = []
    for left, right in itertools.combinations(blocks, 2):
        valid_count = _cross_block_candidate_count(left, right)
        if valid_count > 0:
            candidates.append(((left, right), valid_count * abs(right.index - left.index)))

    if not candidates:
        return

    rng = random.Random(seed)
    _sample_until_target(
        target=target,
        rng=rng,
        seen=seen,
        rows=rows,
        group_id=group_id,
        draw=lambda: _draw_between_blocks(candidates, rng, use_label_buckets),
    )


def _sample_within_blocks(
    blocks: list[_LabelBlock],
    target: int,
    seed: str,
    seen: set[tuple[str, str]],
    rows: list[dict[str, object]],
    group_id: str,
    use_label_buckets: bool,
) -> None:
    candidates = [
        (block, _within_block_candidate_count(block))
        for block in blocks
        if _within_block_candidate_count(block) > 0
    ]
    if not candidates:
        return

    rng = random.Random(seed)
    _sample_until_target(
        target=target,
        rng=rng,
        seen=seen,
        rows=rows,
        group_id=group_id,
        draw=lambda: _draw_within_block(candidates, rng, use_label_buckets),
    )


def _sample_until_target(
    target: int,
    rng: random.Random,
    seen: set[tuple[str, str]],
    rows: list[dict[str, object]],
    group_id: str,
    draw,
) -> None:
    start_count = len(rows)
    max_attempts = max(1000, target * 100)
    attempts = 0
    while len(rows) - start_count < target and attempts < max_attempts:
        attempts += 1
        drawn = draw()
        if drawn is None:
            continue
        record_id_a, label_a, record_id_b, label_b = drawn
        if label_a == label_b:
            continue
        record_id_i, label_i, record_id_j, label_j = _canonical_pair(
            record_id_a, label_a, record_id_b, label_b
        )
        key = (record_id_i, record_id_j)
        if key in seen:
            continue
        seen.add(key)
        y_ij = 1.0 if label_i > label_j else 0.0
        rows.append(_pair_row(group_id, record_id_i, record_id_j, label_i, label_j, y_ij))


def _draw_between_blocks(
    candidates: list[tuple[tuple[_LabelBlock, _LabelBlock], int]],
    rng: random.Random,
    use_label_buckets: bool,
) -> tuple[str, float, str, float] | None:
    left, right = _weighted_choice(candidates, rng)
    if not use_label_buckets:
        record_id_left, label_left = rng.choice(left.items)
        record_id_right, label_right = rng.choice(right.items)
        return record_id_left, label_left, record_id_right, label_right

    label_left = _weighted_label_excluding(left, right, rng)
    if label_left is None:
        return None
    label_right = _weighted_choice(
        [
            (label, len(record_ids))
            for label, record_ids in right.label_to_ids.items()
            if label != label_left
        ],
        rng,
    )
    return (
        rng.choice(left.label_to_ids[label_left]),
        label_left,
        rng.choice(right.label_to_ids[label_right]),
        label_right,
    )


def _draw_within_block(
    candidates: list[tuple[_LabelBlock, int]],
    rng: random.Random,
    use_label_buckets: bool,
) -> tuple[str, float, str, float] | None:
    block = _weighted_choice(candidates, rng)
    if not use_label_buckets:
        record_id_a, label_a = rng.choice(block.items)
        record_id_b, label_b = rng.choice(block.items)
        return record_id_a, label_a, record_id_b, label_b

    label_a = _weighted_label_excluding(block, block, rng)
    if label_a is None:
        return None
    label_b = _weighted_choice(
        [
            (label, len(record_ids))
            for label, record_ids in block.label_to_ids.items()
            if label != label_a
        ],
        rng,
    )
    return (
        rng.choice(block.label_to_ids[label_a]),
        label_a,
        rng.choice(block.label_to_ids[label_b]),
        label_b,
    )


def _weighted_label_excluding(
    source: _LabelBlock,
    other: _LabelBlock,
    rng: random.Random,
) -> float | None:
    weighted_labels: list[tuple[float, int]] = []
    for label, record_ids in source.label_to_ids.items():
        other_same = len(other.label_to_ids.get(label, ()))
        weight = len(record_ids) * (other.n_records - other_same)
        if weight > 0:
            weighted_labels.append((label, weight))
    if not weighted_labels:
        return None
    return _weighted_choice(weighted_labels, rng)


def _cross_block_candidate_count(left: _LabelBlock, right: _LabelBlock) -> int:
    same_label = sum(
        len(left_ids) * len(right.label_to_ids.get(label, ()))
        for label, left_ids in left.label_to_ids.items()
    )
    return left.n_records * right.n_records - same_label


def _within_block_candidate_count(block: _LabelBlock) -> int:
    total = block.n_records * (block.n_records - 1) // 2
    same_label = sum(
        len(record_ids) * (len(record_ids) - 1) // 2
        for record_ids in block.label_to_ids.values()
    )
    return total - same_label
