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
    _validate_pair_sampling(max_pairs_per_group, pair_sample_strategy, pair_fraction, min_pairs_per_group)

    # Rules 1 and 3: only keep_for_training=True records with a finite label.
    trainable = filter_trainable_records(records)

    rows: list[dict[str, object]] = []
    for group_id, group in trainable.groupby("group_id", sort=True):
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
            rows.append(
                dict(
                    pair_id=f"{record_id_i}::{record_id_j}",
                    group_id=group_id,
                    record_id_i=record_id_i,
                    record_id_j=record_id_j,
                    label_i=label_i,
                    label_j=label_j,
                    y_ij=y_ij,
                )
            )

    if not rows:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def _validate_pair_sampling(
    max_pairs_per_group: int,
    pair_sample_strategy: str,
    pair_fraction: float | None,
    min_pairs_per_group: int,
) -> None:
    if max_pairs_per_group < 1:
        raise ValueError(f"max_pairs_per_group must be >= 1, got {max_pairs_per_group}")
    if min_pairs_per_group < 1:
        raise ValueError(f"min_pairs_per_group must be >= 1, got {min_pairs_per_group}")
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
