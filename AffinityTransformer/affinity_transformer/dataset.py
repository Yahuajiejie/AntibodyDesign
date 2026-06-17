"""Standard table loading and ranking dataset construction.

(spec docs/programming_spec.md §5.2)

This module only consumes processed tables that already conform to the
standard schema (spec §3) and have already passed
`scripts/prepare/validate_processed_table.py`. It is not responsible for raw
data cleaning, metric-direction conversion, or `group_id` construction --
those are data-prep concerns (spec §1.1, §4).

Pipeline implemented here::

    processed table
    -> schema validation        (load_records)
    -> filter trainable records (filter_trainable_records)
    -> build AffinityExample     (AffinityRecordDataset)
    -> build pairs (pairwise)    (build_pairs)   -> PairwiseAffinityDataset
    -> build groups (listwise)   (build_groups)  -> ListwiseAffinityDataset
    -> collate_fn / DataLoader / Trainer   (dataloader.py, trainer.py)

`AffinityRecordDataset` / `AffinityExample` is the shared substrate for every
upstream training task. Pairwise (RankNet-style) and listwise (ListMLE /
LambdaRank-style / differentiable-Spearman) tasks are independent *views* on
top of that substrate -- `build_pairs`/`PairwiseAffinityDataset` for the
former, `build_groups`/`ListwiseAffinityDataset` for the latter -- so that
controlled comparisons between upstream tasks can reuse the same filtered
records without re-deriving them. A pointwise view needs no extra class:
`AffinityRecordDataset` already exposes one `rank_label` per record. Which
view a given training run consumes is expected to become a config-level
switch once `dataloader.py`/`trainer.py` are written (not yet wired up here).

If this file grows too large, the plan (spec §5.2) is to split it into
`schema.py`, `records.py`, `pairs.py`, `groups.py`, `dataset.py`, and
`collate.py`.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset

# ── standard schema (spec §3) ────────────────────────────────────────────────

REQUIRED_COLUMNS: tuple[str, ...] = (
    "record_id", "dataset_id", "study_id", "table_id",
    "source_file", "source_row",
    "antibody_id", "antibody_type",
    "heavy_chain", "light_chain", "single_chain_sequence",
    "antigen_key", "antigen_name", "antigen_sequence", "antigen_source",
    "assay_name", "assay_type",
    "metric_name", "metric_value_raw", "metric_value_numeric",
    "metric_unit", "metric_direction", "transform_rule",
    "rank_label", "label_kind",
    "group_id", "keep_for_training", "drop_reason",
)

# Columns needed to build one AffinityExample (spec §5.2).
_EXAMPLE_COLUMNS: tuple[str, ...] = (
    "record_id", "dataset_id",
    "heavy_chain", "light_chain", "single_chain_sequence", "antibody_type",
    "antigen_sequence", "antigen_key",
    "rank_label", "label_kind", "group_id",
)

# Output columns of build_pairs.
PAIR_COLUMNS: tuple[str, ...] = (
    "pair_id", "group_id", "record_id_i", "record_id_j",
    "label_i", "label_j", "y_ij",
)

# Output columns of build_groups.
GROUP_COLUMNS: tuple[str, ...] = (
    "group_id", "record_id", "rank_label", "label_kind",
)

_BINARY_LABEL_KIND = "binary"
_DEFAULT_LARGE_GROUP_THRESHOLD = 10_000
_DEFAULT_PAIR_ENUMERATION_LIMIT = 100_000
_DEFAULT_LABEL_BLOCK_COUNT = 5
_DEFAULT_INTRA_BLOCK_PAIRS_PER_LARGE_GROUP = 50
_DEFAULT_DISCRETE_LABEL_UNIQUE_THRESHOLD = 32
_DEFAULT_DISCRETE_LABEL_RATIO_THRESHOLD = 0.05


# ── example / pair containers (spec §5.2) ───────────────────────────────────


@dataclass(frozen=True)
class AffinityExample:
    """One trainable antibody-antigen record.

    Attributes:
        record_id: Unique identifier of the source record.
        dataset_id: `{study_id}/{table_id}` identifier of the source table.
        heavy_chain: Heavy-chain (or VHH) sequence, or None if absent.
        light_chain: Light-chain sequence, or None if absent/not applicable.
        single_chain_sequence: Single-chain sequence (e.g. scFv), or None.
        antibody_type: One of "Fv", "scFv", "VHH", "Fab", "IgG", "unknown".
        antigen_sequence: Antigen sequence, or None if missing.
        antigen_key: Identifier used to group records by antigen, or None.
        rank_label: Direction-normalized label; larger is better.
        label_kind: One of "experimental", "predicted", "binary", "unknown".
        group_id: Identifier of the homogeneous comparison group.
    """

    record_id: str
    dataset_id: str
    heavy_chain: str | None
    light_chain: str | None
    single_chain_sequence: str | None
    antibody_type: str
    antigen_sequence: str | None
    antigen_key: str | None
    rank_label: float
    label_kind: str
    group_id: str


@dataclass(frozen=True)
class AffinityPairExample:
    """One pairwise ranking example.

    Attributes:
        pair_id: Unique identifier of this pair.
        group_id: Identifier of the homogeneous comparison group both
            records belong to.
        left: The first ("i") record of the pair.
        right: The second ("j") record of the pair.
        y_ij: 1.0 if `left.rank_label > right.rank_label`, else 0.0.
    """

    pair_id: str
    group_id: str
    left: AffinityExample
    right: AffinityExample
    y_ij: float


@dataclass(frozen=True)
class AffinityGroupExample:
    """One listwise ranking example: every surviving record of one group.

    Attributes:
        group_id: Identifier of the homogeneous comparison group.
        label_kind: Shared `label_kind` of every record in the group (the
            standard `group_id` format, spec §3 rule 3, already encodes
            `label_kind`, so this is consistent across `examples`).
        examples: `AffinityExample` for each surviving member, ordered by
            `record_id`. Has at least two elements.
    """

    group_id: str
    label_kind: str
    examples: tuple[AffinityExample, ...]


@dataclass(frozen=True)
class _LabelBlock:
    """Rank-label quantile block used by the large-group pair sampler."""

    index: int
    items: tuple[tuple[str, float], ...]
    label_to_ids: dict[float, tuple[str, ...]]
    n_records: int


# ── loading and filtering ────────────────────────────────────────────────────


def load_records(path: Path) -> pd.DataFrame:
    """Load a standard processed table and check it has the required columns.

    Args:
        path: Path to a `records.parquet` or `records.csv` file produced by
            a data-prep script (spec §3 / §4).

    Returns:
        The table exactly as stored on disk, guaranteed to contain every
        column in `REQUIRED_COLUMNS`.

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValueError: If `path` has an extension other than `.parquet` or
            `.csv`, or the table is missing one or more required columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Processed table not found: {path}")

    if path.suffix == ".parquet":
        records = pd.read_parquet(path)
    elif path.suffix == ".csv":
        records = pd.read_csv(path, low_memory=False)
    else:
        raise ValueError(f"Unsupported processed table extension: {path.suffix!r} ({path})")

    missing = [c for c in REQUIRED_COLUMNS if c not in records.columns]
    if missing:
        raise ValueError(f"Processed table {path} is missing required column(s): {missing}")

    return records


def _is_finite_number(value: object) -> bool:
    """Return True if `value` can be interpreted as a finite float.

    Args:
        value: Any scalar, typically from a `rank_label` cell.

    Returns:
        True if `value` is not None/NaN/+-inf and is castable to float;
        False otherwise (including for non-numeric strings).
    """
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _parse_bool(value: object) -> bool:
    """Parse a standard-table boolean cell strictly.

    Args:
        value: Cell value from `keep_for_training`.

    Returns:
        Boolean interpretation of `value`.

    Raises:
        ValueError: If `value` is not a recognized boolean representation.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, float) and value in (0.0, 1.0):
        return bool(int(value))
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "t", "1", "yes", "y"}:
            return True
        if normalized in {"false", "f", "0", "no", "n"}:
            return False
    raise ValueError(f"Invalid boolean value for keep_for_training: {value!r}")


def filter_trainable_records(records: pd.DataFrame) -> pd.DataFrame:
    """Keep only records usable for training.

    Args:
        records: Standard processed table (e.g. from `load_records`). Must
            contain `keep_for_training` and `rank_label`.

    Returns:
        A new DataFrame (independent copy, index reset) containing only rows
        where `keep_for_training` is True and `rank_label` is a finite
        number. Input `records` is not modified.

    Raises:
        ValueError: If `keep_for_training` or `rank_label` columns are
            missing from `records`.
    """
    required = ("keep_for_training", "rank_label")
    missing = [c for c in required if c not in records.columns]
    if missing:
        raise ValueError(f"records is missing required column(s): {missing}")

    keep_mask = records["keep_for_training"].apply(_parse_bool)
    finite_mask = records["rank_label"].apply(_is_finite_number)
    return records[keep_mask & finite_mask].reset_index(drop=True)


# ── pair construction (spec §5.2 "Pair 构造规则") ────────────────────────────


def _candidate_pairs(group: pd.DataFrame) -> list[tuple[str, str, float, float, float]]:
    """Enumerate valid (i, j) candidate pairs within a single group.

    Implements rules 2, 4, 5, and 8 of spec §5.2: pairs are local to one
    group (caller guarantees this), records with equal `rank_label` are
    skipped, `y_ij` is derived from the label ordering, and each unordered
    pair is produced at most once (no reverse duplicates).

    Rule 7 (binary `label_kind` only pairs across classes) follows
    automatically here: for `label_kind = "binary"`, equal `rank_label`
    values represent the same class and are already excluded by the
    equal-label check.

    Args:
        group: Rows of a `filter_trainable_records` output that all share
            the same `group_id`. Must contain `record_id` and `rank_label`.

    Returns:
        A list of `(record_id_i, record_id_j, label_i, label_j, y_ij)`
        tuples, sorted by `(record_id_i, record_id_j)`. Empty if the group
        has fewer than two records or all records share the same label.
    """
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
    """Build pairwise ranking examples within each group.

    Args:
        records: Standard processed table. Must contain record_id, group_id,
            rank_label, label_kind, and keep_for_training.
        max_pairs_per_group: Maximum sampled pairs per group.
        seed: Random seed for reproducible sampling.
        pair_sample_strategy: `"absolute_cap"` keeps the legacy behavior:
            sample at most `max_pairs_per_group` pairs per group.
            `"capped_proportional"` samples
            `min(max_pairs_per_group, max(min_pairs_per_group,
            ceil(n_candidate_pairs * pair_fraction)))`.
        pair_fraction: Fraction of candidate pairs used only by
            `"capped_proportional"`.
        min_pairs_per_group: Lower target for `"capped_proportional"` before
            applying the upper cap. Never creates more pairs than a group has.
        large_group_threshold: Groups with at least this many trainable
            records use memory-safe block sampling instead of full pair
            enumeration.
        pair_enumeration_limit: Groups with more candidate pairs than this
            use memory-safe block sampling even if they are below
            `large_group_threshold`.
        label_block_count: Number of rank-label quantile blocks used by the
            large-group sampler.
        intra_block_pairs_per_large_group: Extra fine-grained pairs sampled
            from within rank-label blocks for large groups.
        discrete_label_unique_threshold: Groups with at most this many unique
            labels use label-aware block sampling.
        discrete_label_ratio_threshold: Groups whose unique-label ratio is at
            most this value use label-aware block sampling.

    Returns:
        DataFrame with pair_id, group_id, record_id_i, record_id_j, label_i,
        label_j, and y_ij. Has zero rows (but the columns above) if no group
        produces a valid pair. Groups with fewer than two trainable records,
        or where every record shares the same `rank_label`, contribute no
        rows and do not raise an error.

    Raises:
        ValueError: If required columns are missing, or if
            `max_pairs_per_group` is less than 1.
    """
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

    # Rules 1 and 3: only keep_for_training=True records with a finite label.
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

            # Rule 6: cap/sample pairs per group, deterministically from `seed`.
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


def _canonical_pair(
    record_id_a: str,
    label_a: float,
    record_id_b: str,
    label_b: float,
) -> tuple[str, float, str, float]:
    if record_id_a <= record_id_b:
        return record_id_a, label_a, record_id_b, label_b
    return record_id_b, label_b, record_id_a, label_a


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


def _validate_pair_sampling(
    max_pairs_per_group: int,
    pair_sample_strategy: str,
    pair_fraction: float | None,
    min_pairs_per_group: int,
    large_group_threshold: int,
    pair_enumeration_limit: int,
    label_block_count: int,
    intra_block_pairs_per_large_group: int,
    discrete_label_unique_threshold: int,
    discrete_label_ratio_threshold: float,
) -> None:
    if max_pairs_per_group < 1:
        raise ValueError(f"max_pairs_per_group must be >= 1, got {max_pairs_per_group}")
    if min_pairs_per_group < 1:
        raise ValueError(f"min_pairs_per_group must be >= 1, got {min_pairs_per_group}")
    if large_group_threshold < 2:
        raise ValueError(f"large_group_threshold must be >= 2, got {large_group_threshold}")
    if pair_enumeration_limit < 1:
        raise ValueError(f"pair_enumeration_limit must be >= 1, got {pair_enumeration_limit}")
    if label_block_count < 2:
        raise ValueError(f"label_block_count must be >= 2, got {label_block_count}")
    if intra_block_pairs_per_large_group < 0:
        raise ValueError(
            "intra_block_pairs_per_large_group must be >= 0, "
            f"got {intra_block_pairs_per_large_group}"
        )
    if discrete_label_unique_threshold < 2:
        raise ValueError(
            "discrete_label_unique_threshold must be >= 2, "
            f"got {discrete_label_unique_threshold}"
        )
    if not (0.0 < discrete_label_ratio_threshold <= 1.0):
        raise ValueError(
            "discrete_label_ratio_threshold must be in (0, 1], "
            f"got {discrete_label_ratio_threshold}"
        )
    if pair_sample_strategy not in {"absolute_cap", "capped_proportional"}:
        raise ValueError(
            "pair_sample_strategy must be 'absolute_cap' or 'capped_proportional', "
            f"got {pair_sample_strategy!r}"
        )
    if pair_sample_strategy == "capped_proportional":
        if pair_fraction is None or not (0.0 < pair_fraction <= 1.0):
            raise ValueError(
                "pair_fraction must be in (0, 1] when pair_sample_strategy='capped_proportional'"
            )


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


# ── group construction (listwise analogue of build_pairs) ───────────────────


def _group_member_ids(group: pd.DataFrame) -> list[str]:
    """Return the trainable record_ids of one group, or `[]` if not rankable.

    A group is "rankable" for listwise training only if it has at least two
    distinct `rank_label` values. This mirrors `_candidate_pairs`'s
    equal-label skip (rule 4) and `compute_group_spearman`'s
    `n_unique_labels < 2` skip (spec §5.6 rule 1): a group where every record
    shares one label carries no ranking signal, regardless of upstream task.

    Args:
        group: Rows of a `filter_trainable_records` output that all share
            the same `group_id`. Must contain `record_id` and `rank_label`.

    Returns:
        Sorted `record_id` values for the group, or `[]` if the group has
        fewer than two distinct `rank_label` values.
    """
    labels = group["rank_label"].astype(float)
    if labels.nunique() < 2:
        return []
    return sorted(group["record_id"].astype(str))


def build_groups(
    records: pd.DataFrame,
    max_group_size: int | None,
    seed: int,
) -> pd.DataFrame:
    """Build listwise ranking groups (the listwise analogue of `build_pairs`).

    Args:
        records: Standard processed table. Must contain record_id, group_id,
            rank_label, label_kind, and keep_for_training.
        max_group_size: Maximum number of records retained per group, or
            `None` to keep every trainable record. If not `None`, must be
            >= 2 (a group of size 1 carries no ranking signal).
        seed: Random seed for reproducible sampling.

    Returns:
        DataFrame with GROUP_COLUMNS (group_id, record_id, rank_label,
        label_kind), one row per surviving (group, record) pair, sorted by
        (group_id, record_id). Has zero rows (but the columns above) if no
        group qualifies. Groups with fewer than two distinct `rank_label`
        values contribute no rows and do not raise an error (mirrors
        `build_pairs`'s equal-label skip and `compute_group_spearman`'s
        `n_unique_labels < 2` skip).

    Raises:
        ValueError: If required columns are missing, or if
            `max_group_size` is not `None` and is less than 2.
    """
    required = ("record_id", "group_id", "rank_label", "label_kind", "keep_for_training")
    missing = [c for c in required if c not in records.columns]
    if missing:
        raise ValueError(f"records is missing required column(s): {missing}")
    if max_group_size is not None and max_group_size < 2:
        raise ValueError(f"max_group_size must be None or >= 2, got {max_group_size}")

    # Rules 1 and 3 (spec §5.2): only keep_for_training=True records with a
    # finite label.
    trainable = filter_trainable_records(records)

    rows: list[dict[str, object]] = []
    for group_id, group in trainable.groupby("group_id", sort=True):
        member_ids = _group_member_ids(group)
        if not member_ids:
            continue

        # Cap group size, sampled deterministically from `seed` (mirrors
        # build_pairs's per-group pair cap, rule 6). Note: a capped group can
        # in principle end up with a single distinct `rank_label`; callers
        # picking a small `max_group_size` for binary-style groups should
        # keep this in mind.
        if max_group_size is not None and len(member_ids) > max_group_size:
            rng = random.Random(f"{seed}:{group_id}")
            member_ids = sorted(rng.sample(member_ids, max_group_size))

        group_by_id = group.set_index(group["record_id"].astype(str))
        for record_id in member_ids:
            row = group_by_id.loc[record_id]
            rows.append(
                dict(
                    group_id=group_id,
                    record_id=record_id,
                    rank_label=float(row["rank_label"]),
                    label_kind=str(row["label_kind"]),
                )
            )

    if not rows:
        return pd.DataFrame(columns=GROUP_COLUMNS)
    return pd.DataFrame(rows, columns=GROUP_COLUMNS)


# ── row -> AffinityExample ───────────────────────────────────────────────────


def _optional_str(value: object) -> str | None:
    """Convert a table cell to `str`, mapping missing values to `None`.

    Args:
        value: A cell value, possibly `None` or `float('nan')` (pandas'
            representation of a missing value in an object column).

    Returns:
        `None` if `value` is `None` or NaN, otherwise `str(value)`.
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value)


def _row_to_example(row: pd.Series) -> AffinityExample:
    """Convert one processed-table row into an `AffinityExample`.

    Args:
        row: A row containing at least the columns in `_EXAMPLE_COLUMNS`,
            typically from `filter_trainable_records` output (i.e. with a
            finite `rank_label`).

    Returns:
        The corresponding `AffinityExample`.
    """
    return AffinityExample(
        record_id=str(row["record_id"]),
        dataset_id=str(row["dataset_id"]),
        heavy_chain=_optional_str(row["heavy_chain"]),
        light_chain=_optional_str(row["light_chain"]),
        single_chain_sequence=_optional_str(row["single_chain_sequence"]),
        antibody_type=str(row["antibody_type"]),
        antigen_sequence=_optional_str(row["antigen_sequence"]),
        antigen_key=_optional_str(row["antigen_key"]),
        rank_label=float(row["rank_label"]),
        label_kind=str(row["label_kind"]),
        group_id=str(row["group_id"]),
    )


# ── torch datasets (spec §5.2 core classes) ─────────────────────────────────


class AffinityRecordDataset(Dataset):
    """Exposes processed-table rows as `AffinityExample` items.

    `records` is expected to be the output of `filter_trainable_records`
    (i.e. only `keep_for_training = True` rows with a finite `rank_label`).
    This class does not filter or validate label finiteness itself.
    """

    def __init__(self, records: pd.DataFrame) -> None:
        """Wrap a processed table for indexed `AffinityExample` access.

        Args:
            records: Table containing at least `_EXAMPLE_COLUMNS`, typically
                the output of `filter_trainable_records`.

        Raises:
            ValueError: If `records` is missing any column required to build
                an `AffinityExample`.
        """
        missing = [c for c in _EXAMPLE_COLUMNS if c not in records.columns]
        if missing:
            raise ValueError(f"records is missing required column(s): {missing}")
        self._records = records.reset_index(drop=True)

    def __len__(self) -> int:
        """Return the number of records."""
        return len(self._records)

    def __getitem__(self, index: int) -> AffinityExample:
        """Return the `AffinityExample` at `index`.

        Args:
            index: Zero-based row position.

        Returns:
            The corresponding `AffinityExample`.
        """
        return _row_to_example(self._records.iloc[index])


class PairwiseAffinityDataset(Dataset):
    """Exposes a `build_pairs` table as `AffinityPairExample` items."""

    def __init__(self, records: pd.DataFrame, pairs: pd.DataFrame) -> None:
        """Wrap a processed table and a pairs table for pairwise access.

        Args:
            records: Table containing at least `_EXAMPLE_COLUMNS`, with a
                unique `record_id` per row (typically the output of
                `filter_trainable_records`).
            pairs: Table containing at least `PAIR_COLUMNS`, typically the
                output of `build_pairs`. Every `record_id_i`/`record_id_j`
                must be present in `records["record_id"]`.

        Raises:
            ValueError: If `records` or `pairs` is missing required columns.
        """
        missing_records = [c for c in _EXAMPLE_COLUMNS if c not in records.columns]
        if missing_records:
            raise ValueError(f"records is missing required column(s): {missing_records}")
        missing_pairs = [c for c in PAIR_COLUMNS if c not in pairs.columns]
        if missing_pairs:
            raise ValueError(f"pairs is missing required column(s): {missing_pairs}")

        self._records = records.set_index("record_id", drop=False)
        self._pairs = pairs.reset_index(drop=True)

    def __len__(self) -> int:
        """Return the number of pairs."""
        return len(self._pairs)

    def __getitem__(self, index: int) -> AffinityPairExample:
        """Return the `AffinityPairExample` at `index`.

        Args:
            index: Zero-based row position into the pairs table.

        Returns:
            The corresponding `AffinityPairExample`, with `left`/`right`
            built from the matching rows of `records`.
        """
        pair = self._pairs.iloc[index]
        left = _row_to_example(self._records.loc[pair["record_id_i"]])
        right = _row_to_example(self._records.loc[pair["record_id_j"]])
        return AffinityPairExample(
            pair_id=str(pair["pair_id"]),
            group_id=str(pair["group_id"]),
            left=left,
            right=right,
            y_ij=float(pair["y_ij"]),
        )


class ListwiseAffinityDataset(Dataset):
    """Exposes a `build_groups` table as `AffinityGroupExample` items.

    Each item is one homogeneous comparison group with every surviving
    member record, for listwise upstream tasks (e.g. ListMLE, LambdaRank-
    style dynamic weighting, differentiable-Spearman losses) -- as opposed to
    `PairwiseAffinityDataset`, where each item is a single (i, j) pair.
    """

    def __init__(self, records: pd.DataFrame, groups: pd.DataFrame) -> None:
        """Wrap a processed table and a groups table for listwise access.

        Args:
            records: Table containing at least `_EXAMPLE_COLUMNS`, with a
                unique `record_id` per row (typically the output of
                `filter_trainable_records`).
            groups: Table containing at least `GROUP_COLUMNS`, typically the
                output of `build_groups`. Every `record_id` must be present
                in `records["record_id"]`.

        Raises:
            ValueError: If `records` or `groups` is missing required
                columns.
        """
        missing_records = [c for c in _EXAMPLE_COLUMNS if c not in records.columns]
        if missing_records:
            raise ValueError(f"records is missing required column(s): {missing_records}")
        missing_groups = [c for c in GROUP_COLUMNS if c not in groups.columns]
        if missing_groups:
            raise ValueError(f"groups is missing required column(s): {missing_groups}")

        self._records = records.set_index("record_id", drop=False)
        groups = groups.reset_index(drop=True)
        self._groups = groups

        self._group_ids: list[str] = []
        self._group_row_indices: dict[str, list[int]] = {}
        for group_id, index in groups.groupby("group_id", sort=False).groups.items():
            group_id = str(group_id)
            self._group_ids.append(group_id)
            self._group_row_indices[group_id] = list(index)

    def __len__(self) -> int:
        """Return the number of groups."""
        return len(self._group_ids)

    def __getitem__(self, index: int) -> AffinityGroupExample:
        """Return the `AffinityGroupExample` at `index`.

        Args:
            index: Zero-based position into the distinct `group_id` values
                of `groups`, in order of first appearance.

        Returns:
            The corresponding `AffinityGroupExample`, with `examples` built
            from the matching rows of `records`, in the order they appear in
            `groups`.
        """
        group_id = self._group_ids[index]
        rows = self._groups.iloc[self._group_row_indices[group_id]]
        examples = tuple(
            _row_to_example(self._records.loc[record_id]) for record_id in rows["record_id"]
        )
        label_kind = str(rows.iloc[0]["label_kind"])
        return AffinityGroupExample(group_id=group_id, label_kind=label_kind, examples=examples)
