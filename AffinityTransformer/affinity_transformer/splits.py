"""Train/validation/test splitting for standard processed records.

This module consumes only processed tables that already follow the standard
schema. It does not read raw CSVs, does not derive labels, and does not build
pairs. Its job is to create split files and explicit leakage reports before
training starts.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import pandas as pd

from .record_filter import antibody_sequence_hashes
from .utils import ensure_dir

SPLIT_COLUMNS = ("split", "strategy", "n_records", "n_trainable_records", "n_groups",
                 "n_trainable_groups", "n_spearman_eligible_groups",
                 "label_kind_counts", "antigen_source_counts")
LEAKAGE_COLUMNS = ("check_name", "status", "n_violations", "details")
PINNED_GROUPS_COLUMNS = ("group_id", "n_records", "n_antibody_units", "reason")
ENTITY_UNIT_COLUMNS = (
    "component_id", "assigned_split", "validation_fold", "n_records",
    "n_entity_clusters", "n_measurement_families",
)
ELIGIBILITY_COLUMNS = (
    "split", "group_id", "n_assigned_records", "n_candidate_records",
    "n_eligible_records", "n_unique_inputs", "n_unique_labels", "status", "reason",
)
COLD_START_IDENTITY_COLUMNS = (
    "measurement_family_id",
    "antibody_sequence_key",
    "antibody_cluster_id",
    "antigen_sequence_key",
    "antigen_cluster_id",
    "interaction_key",
    "effective_antigen_input_hash",
)

VALID_STRATEGIES = {"debug_record_split", "group_holdout_split", "antigen_cluster_holdout_split"}

# `within_antigen_split` is intentionally NOT in VALID_STRATEGIES / build_splits.
# It answers a different question (programming_spec_v1.0.md section 3.2:
# "known-antigen, new-antibody") and deliberately allows the same group_id
# to appear in more than one split -- mixing it into build_splits's dispatch
# would make it too easy to point a real training config at it by mistake
# and report the result as unseen-antigen generalization, which it is not.
AUXILIARY_STRATEGIES = {"within_antigen_split"}
ENTITY_COLD_START_STRATEGIES = {
    "antibody_cold_start_split",
    "antigen_cold_start_split",
}


@dataclass
class SplitResult:
    """DataFrames and QC reports produced by `build_splits`."""

    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    summary: pd.DataFrame
    leakage_report: pd.DataFrame


@dataclass
class WithinAntigenSplitResult:
    """Output of `build_within_antigen_split` (programming_spec_v1.0.md 3.2,
    "known-antigen, new-antibody").

    Unlike `SplitResult` from `group_holdout_split`, the same `group_id` (and
    the same antibody-sequence identity, as long as it's via a *different*
    group) may legitimately appear in more than one split here -- that is
    the point of this auxiliary protocol. What this guarantees, per group,
    is the only thing that actually matters for `dataset.pairs.build_pairs`
    (which only ever constructs pairs *within* one group_id): no
    `record_id` crosses a split, and within any single group, an antibody
    assigned to train never also shows up in that same group's valid/test
    rows. Whether that exact antibody sequence also appears in some other,
    unrelated group's training data is allowed and is not leakage -- the
    relationship actually being predicted (this antibody vs THIS antigen's
    other candidates) was never trained on regardless.

    `pinned_groups` lists every group that was too small to split reliably
    (fewer than 3 distinct antibody-sequence units, or splitting would leave
    fewer than `min_eval_records` records in valid or test) -- these are
    routed entirely to train rather than forced into an unstable 1-2-point
    split (spec section 3.4).

    Always report results from this split as "within-antigen
    generalization", never as evidence of generalization to unseen
    antigens.
    """

    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    summary: pd.DataFrame
    leakage_report: pd.DataFrame
    pinned_groups: pd.DataFrame


@dataclass(frozen=True)
class GroupFold:
    """One group-isolated cross-validation fold."""

    index: int
    train: pd.DataFrame
    valid: pd.DataFrame


@dataclass
class EntityColdStartSplitResult:
    """Fixed train/valid/test artifacts for one strict entity protocol."""

    protocol: str
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    summary: pd.DataFrame
    leakage_report: pd.DataFrame
    eligibility_report: pd.DataFrame
    excluded_records: pd.DataFrame
    unit_assignments: pd.DataFrame


@dataclass
class EntityColdStartFold:
    """One protocol-aware development fold; final test never rotates here."""

    protocol: str
    index: int
    train: pd.DataFrame
    valid: pd.DataFrame
    leakage_report: pd.DataFrame
    eligibility_report: pd.DataFrame
    excluded_records: pd.DataFrame
    unit_assignments: pd.DataFrame


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


def build_antibody_cold_start_split(
    records: pd.DataFrame,
    valid_fraction: float,
    test_fraction: float,
    seed: int,
    *,
    min_eval_records: int = 2,
    require_train_group: bool = True,
) -> EntityColdStartSplitResult:
    """Split globally by antibody components for known-antigen evaluation.

    Antibody clusters and measurement families are indivisible across the
    whole table. Validation/test records are retained only when their exact
    antigen (and, by default, their experimental group) occurs in train.
    Records excluded from the primary protocol remain auditable in
    ``excluded_records``.
    """
    return _build_entity_cold_start_split(
        records,
        protocol="antibody_cold_start",
        valid_fraction=valid_fraction,
        test_fraction=test_fraction,
        seed=seed,
        min_eval_records=min_eval_records,
        require_train_group=require_train_group,
    )


def build_antigen_cold_start_split(
    records: pd.DataFrame,
    valid_fraction: float,
    test_fraction: float,
    seed: int,
    *,
    min_eval_records: int = 2,
) -> EntityColdStartSplitResult:
    """Split by antigen components and evaluate only train-seen antibodies."""
    return _build_entity_cold_start_split(
        records,
        protocol="antigen_cold_start",
        valid_fraction=valid_fraction,
        test_fraction=test_fraction,
        seed=seed,
        min_eval_records=min_eval_records,
        require_train_group=False,
    )


def build_antibody_cold_start_kfolds(
    records: pd.DataFrame,
    n_splits: int,
    seed: int,
    *,
    min_eval_records: int = 2,
    require_train_group: bool = True,
) -> list[EntityColdStartFold]:
    """Build antibody-component-isolated development folds."""
    return _build_entity_cold_start_kfolds(
        records,
        protocol="antibody_cold_start",
        n_splits=n_splits,
        seed=seed,
        min_eval_records=min_eval_records,
        require_train_group=require_train_group,
    )


def build_antigen_cold_start_kfolds(
    records: pd.DataFrame,
    n_splits: int,
    seed: int,
    *,
    min_eval_records: int = 2,
) -> list[EntityColdStartFold]:
    """Build antigen-component-isolated development folds."""
    return _build_entity_cold_start_kfolds(
        records,
        protocol="antigen_cold_start",
        n_splits=n_splits,
        seed=seed,
        min_eval_records=min_eval_records,
        require_train_group=False,
    )


def build_splits(
    records: pd.DataFrame,
    strategy: str,
    valid_fraction: float,
    test_fraction: float,
    seed: int,
    antigen_clusters: pd.DataFrame | None = None,
) -> SplitResult:
    """Build train/valid/test splits from one merged processed table.

    Args:
        records: Standard processed records.
        strategy: "debug_record_split", "group_holdout_split", or
            "antigen_cluster_holdout_split".
        valid_fraction: Fraction of units reserved for validation.
        test_fraction: Fraction of units reserved for test.
        seed: Random seed controlling unit shuffling.
        antigen_clusters: Required when `strategy ==
            "antigen_cluster_holdout_split"` -- the output of
            `antigen_clustering.compute_antigen_clusters` (a DataFrame
            mapping every `antigen_key` in `records` to an
            `antigen_cluster_id`). Ignored for other strategies.

    Returns:
        `SplitResult` containing train/valid/test records and two QC reports.

    Raises:
        ValueError: If required columns are missing, the strategy/fractions
            are invalid, too few units exist, `antigen_clusters` is missing
            or incomplete for `antigen_cluster_holdout_split`, or a leakage
            check fails.
    """
    _validate_inputs(records, strategy, valid_fraction, test_fraction)

    if strategy == "debug_record_split":
        train, valid, test = _split_by_record(records, valid_fraction, test_fraction, seed)
    elif strategy == "group_holdout_split":
        train, valid, test = _split_by_group(records, valid_fraction, test_fraction, seed)
    elif strategy == "antigen_cluster_holdout_split":
        if antigen_clusters is None:
            raise ValueError(
                "antigen_clusters is required for strategy='antigen_cluster_holdout_split' "
                "-- compute it with antigen_clustering.compute_antigen_clusters first"
            )
        train, valid, test = _split_by_antigen_cluster(
            records, valid_fraction, test_fraction, seed, antigen_clusters
        )
    else:  # guarded by _validate_inputs
        raise ValueError(f"Unsupported split strategy: {strategy!r}")

    summary = _build_summary(strategy, train=train, valid=valid, test=test)
    leakage_report = _build_leakage_report(strategy, train=train, valid=valid, test=test)
    failed = leakage_report[leakage_report["status"] != "PASS"]
    if not failed.empty:
        raise ValueError(f"Split leakage check failed: {failed.to_dict(orient='records')}")

    # _split_by_antigen_cluster keeps a `_antigen_cluster_id` helper column
    # so the leakage check above could verify it directly; strip it from
    # the user-facing output now that the check has passed.
    if "_antigen_cluster_id" in train.columns:
        train = train.drop(columns=["_antigen_cluster_id"])
        valid = valid.drop(columns=["_antigen_cluster_id"])
        test = test.drop(columns=["_antigen_cluster_id"])

    return SplitResult(train=train, valid=valid, test=test, summary=summary,
                       leakage_report=leakage_report)


def write_splits(result: SplitResult, output_dir: Path) -> None:
    """Write split records and QC reports to `output_dir`."""
    output_dir = ensure_dir(Path(output_dir))
    result.train.to_parquet(output_dir / "train.parquet", index=False)
    result.valid.to_parquet(output_dir / "valid.parquet", index=False)
    result.test.to_parquet(output_dir / "test.parquet", index=False)
    result.summary.to_csv(output_dir / "split_summary.csv", index=False)
    result.leakage_report.to_csv(output_dir / "leakage_report.csv", index=False)


def build_within_antigen_split(
    records: pd.DataFrame,
    valid_fraction: float,
    test_fraction: float,
    seed: int,
    min_eval_records: int = 5,
) -> WithinAntigenSplitResult:
    """Auxiliary "known-antigen, new-antibody" split (programming_spec_v1.0.md
    section 3.2). See `WithinAntigenSplitResult` for what is and isn't
    guaranteed.

    Each `group_id` is split independently: its own records are partitioned
    by antibody-sequence identity (`record_filter.antibody_sequence_hashes`)
    into train/valid/test, weighted by record count, reusing the same
    `_partition_weighted_units` helper `group_holdout_split` uses (just with
    "antibody identity within this group" as the unit instead of "group_id
    across the whole table"). This is sufficient because
    `dataset.pairs.build_pairs` only ever constructs comparison pairs within
    a single group_id -- so independently deciding each group's own
    train/valid/test split already guarantees no comparison pair repeats
    across a split boundary; nothing is gained by additionally forcing
    antibody identity to be consistent *across* different groups (see the
    discussion this design is based on: the same antibody legitimately
    being "known" via an unrelated antigen does not leak anything about how
    it ranks against THIS antigen).

    Args:
        records: Standard processed records (must include heavy_chain,
            light_chain, single_chain_sequence in addition to the columns
            `build_splits` requires).
        valid_fraction: Fraction of each group's own antibody-sequence units
            reserved for validation.
        test_fraction: Fraction of each group's own antibody-sequence units
            reserved for test.
        seed: Base random seed. Each group derives its own seed from this
            plus its `group_id` (`_derive_group_seed`, hashlib-based so it's
            stable across process restarts, unlike Python's randomized
            `hash()`), so groups don't all shuffle identically while the
            whole split stays a deterministic function of `(records, seed)`.
        min_eval_records: Minimum number of records a group must contribute
            to BOTH valid and test for it to be split at all; smaller
            groups are routed entirely to train (see `pinned_groups`).

    Raises:
        ValueError: If required columns are missing, fractions are invalid,
            no group could be split, or the record_id leakage check fails.
    """
    required = ("record_id", "group_id", "keep_for_training", "rank_label",
                "heavy_chain", "light_chain", "single_chain_sequence")
    missing = [column for column in required if column not in records.columns]
    if missing:
        raise ValueError(f"records is missing required column(s): {missing}")
    if valid_fraction < 0 or test_fraction < 0:
        raise ValueError("valid_fraction and test_fraction must be non-negative")
    if not (0.0 < valid_fraction + test_fraction < 1.0):
        raise ValueError("valid_fraction + test_fraction must be > 0 and < 1")
    if min_eval_records < 1:
        raise ValueError("min_eval_records must be >= 1")
    if records.empty:
        raise ValueError("records must be non-empty")
    if records["record_id"].isna().any():
        raise ValueError("records contains null record_id values")
    record_ids = records["record_id"].astype(str)
    if record_ids.duplicated().any():
        duplicated = record_ids[record_ids.duplicated()].tolist()
        raise ValueError(f"records contains duplicate record_id values: {duplicated[:10]}")

    working = records.copy()
    working["_antibody_unit"] = antibody_sequence_hashes(working)

    train_parts: list[pd.DataFrame] = []
    valid_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    pinned_rows: list[dict[str, object]] = []

    grouped = working.groupby(working["group_id"].astype(str), sort=True)
    for group_id, group_df in grouped:
        weights = group_df.groupby("_antibody_unit").size().astype(int).to_dict()

        if len(weights) < 3:
            train_parts.append(group_df)
            pinned_rows.append({
                "group_id": group_id, "n_records": len(group_df),
                "n_antibody_units": len(weights),
                "reason": "fewer than 3 distinct antibody-sequence units",
            })
            continue

        group_seed = _derive_group_seed(seed, group_id)
        try:
            train_units, valid_units, test_units = _partition_weighted_units(
                weights, valid_fraction, test_fraction, group_seed
            )
        except ValueError as error:
            train_parts.append(group_df)
            pinned_rows.append({
                "group_id": group_id, "n_records": len(group_df),
                "n_antibody_units": len(weights),
                "reason": f"partitioning failed: {error}",
            })
            continue

        n_valid_records = sum(weights[unit] for unit in valid_units)
        n_test_records = sum(weights[unit] for unit in test_units)
        if n_valid_records < min_eval_records or n_test_records < min_eval_records:
            train_parts.append(group_df)
            pinned_rows.append({
                "group_id": group_id, "n_records": len(group_df),
                "n_antibody_units": len(weights),
                "reason": (
                    f"valid/test would have fewer than min_eval_records={min_eval_records} "
                    f"records (got valid={n_valid_records}, test={n_test_records})"
                ),
            })
            continue

        unit = group_df["_antibody_unit"]
        train_parts.append(group_df.loc[unit.isin(train_units)])
        valid_parts.append(group_df.loc[unit.isin(valid_units)])
        test_parts.append(group_df.loc[unit.isin(test_units)])

    train = _concat_sorted(train_parts, working.columns)
    valid = _concat_sorted(valid_parts, working.columns)
    test = _concat_sorted(test_parts, working.columns)
    if valid.empty or test.empty:
        raise ValueError(
            "No group had enough antibody-sequence units to populate both valid "
            "and test under the within-antigen protocol; lower min_eval_records "
            "or valid_fraction/test_fraction, or use group_holdout_split instead."
        )

    leakage_report = _build_within_antigen_leakage_report(train=train, valid=valid, test=test)
    failed = leakage_report[leakage_report["status"] != "PASS"]
    if not failed.empty:
        raise ValueError(f"Split leakage check failed: {failed.to_dict(orient='records')}")

    summary = _build_summary("within_antigen_split", train=train, valid=valid, test=test)
    pinned_groups = pd.DataFrame(pinned_rows, columns=PINNED_GROUPS_COLUMNS)

    return WithinAntigenSplitResult(
        train=train.drop(columns=["_antibody_unit"]),
        valid=valid.drop(columns=["_antibody_unit"]),
        test=test.drop(columns=["_antibody_unit"]),
        summary=summary,
        leakage_report=leakage_report,
        pinned_groups=pinned_groups,
    )


def write_within_antigen_split(result: WithinAntigenSplitResult, output_dir: Path) -> None:
    """Write within-antigen split records and QC reports to `output_dir`."""
    output_dir = ensure_dir(Path(output_dir))
    result.train.to_parquet(output_dir / "train.parquet", index=False)
    result.valid.to_parquet(output_dir / "valid.parquet", index=False)
    result.test.to_parquet(output_dir / "test.parquet", index=False)
    result.summary.to_csv(output_dir / "split_summary.csv", index=False)
    result.leakage_report.to_csv(output_dir / "leakage_report.csv", index=False)
    result.pinned_groups.to_csv(output_dir / "pinned_groups.csv", index=False)


def write_entity_cold_start_split(
    result: EntityColdStartSplitResult,
    output_dir: Path,
) -> None:
    """Write one strict entity-protocol split and all audit artifacts."""
    output_dir = ensure_dir(Path(output_dir))
    result.train.to_parquet(output_dir / "train.parquet", index=False)
    result.valid.to_parquet(output_dir / "valid.parquet", index=False)
    result.test.to_parquet(output_dir / "test.parquet", index=False)
    result.summary.to_csv(output_dir / "split_summary.csv", index=False)
    result.leakage_report.to_csv(output_dir / "leakage_report.csv", index=False)
    result.eligibility_report.to_csv(output_dir / "eligibility_report.csv", index=False)
    result.excluded_records.to_parquet(output_dir / "excluded_records.parquet", index=False)
    result.unit_assignments.to_parquet(output_dir / "unit_assignments.parquet", index=False)


def _build_entity_cold_start_split(
    records: pd.DataFrame,
    *,
    protocol: str,
    valid_fraction: float,
    test_fraction: float,
    seed: int,
    min_eval_records: int,
    require_train_group: bool,
) -> EntityColdStartSplitResult:
    _validate_fraction_and_eval_size(valid_fraction, test_fraction, min_eval_records)
    working, pre_split_excluded = _prepare_cold_start_records(records, protocol)
    entity_column = _protocol_entity_column(protocol)
    working["_component_id"] = _derive_entity_components(working, protocol)
    weights = working.groupby("_component_id", sort=True).size().astype(int).to_dict()
    train_units, valid_units, test_units = _partition_weighted_units(
        weights, valid_fraction, test_fraction, seed
    )
    assigned = _assign_component_splits(
        working,
        train_units=train_units,
        valid_units=valid_units,
        test_units=test_units,
    )
    raw_train = assigned.loc[assigned["_assigned_split"] == "train"].copy()
    raw_valid = assigned.loc[assigned["_assigned_split"] == "valid"].copy()
    raw_test = assigned.loc[assigned["_assigned_split"] == "test"].copy()

    valid, valid_report, valid_excluded = _select_protocol_eligible_records(
        raw_train,
        raw_valid,
        protocol=protocol,
        split_name="valid",
        min_eval_records=min_eval_records,
        require_train_group=require_train_group,
    )
    test, test_report, test_excluded = _select_protocol_eligible_records(
        raw_train,
        raw_test,
        protocol=protocol,
        split_name="test",
        min_eval_records=min_eval_records,
        require_train_group=require_train_group,
    )
    if valid.empty or test.empty:
        raise ValueError(
            f"{protocol} produced an empty protocol-eligible valid or test set; "
            "inspect entity frequencies and group eligibility"
        )

    leakage_report = _build_entity_protocol_leakage_report(
        protocol,
        assigned={"train": raw_train, "valid": raw_valid, "test": raw_test},
        eligible={"valid": valid, "test": test},
        require_train_group=require_train_group,
    )
    failed = leakage_report.loc[leakage_report["status"] != "PASS"]
    if not failed.empty:
        raise ValueError(
            f"{protocol} leakage check failed: {failed.to_dict(orient='records')}"
        )

    eligibility_report = pd.concat(
        [valid_report, test_report], ignore_index=True
    )[list(ELIGIBILITY_COLUMNS)]
    excluded_records = _combine_excluded_records(
        pre_split_excluded, valid_excluded, test_excluded, columns=records.columns
    )
    unit_assignments = _build_unit_assignments(
        assigned,
        entity_column=entity_column,
        validation_folds=None,
    )
    train = _drop_split_helpers(raw_train)
    valid = _drop_split_helpers(valid)
    test = _drop_split_helpers(test)
    summary = _build_summary(protocol, train=train, valid=valid, test=test)
    return EntityColdStartSplitResult(
        protocol=protocol,
        train=train,
        valid=valid,
        test=test,
        summary=summary,
        leakage_report=leakage_report,
        eligibility_report=eligibility_report,
        excluded_records=excluded_records,
        unit_assignments=unit_assignments,
    )


def _build_entity_cold_start_kfolds(
    records: pd.DataFrame,
    *,
    protocol: str,
    n_splits: int,
    seed: int,
    min_eval_records: int,
    require_train_group: bool,
) -> list[EntityColdStartFold]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if min_eval_records < 2:
        raise ValueError("min_eval_records must be at least 2")
    working, pre_split_excluded = _prepare_cold_start_records(records, protocol)
    entity_column = _protocol_entity_column(protocol)
    working["_component_id"] = _derive_entity_components(working, protocol)
    weights = working.groupby("_component_id", sort=True).size().astype(int).to_dict()
    fold_units = _assign_weighted_units_to_folds(weights, n_splits, seed)
    validation_folds = {
        component_id: fold_index
        for fold_index, components in enumerate(fold_units)
        for component_id in components
    }
    base_assignments = working.copy()
    base_assignments["_validation_fold"] = base_assignments["_component_id"].map(
        validation_folds
    )
    unit_assignments = _build_unit_assignments(
        base_assignments,
        entity_column=entity_column,
        validation_folds=validation_folds,
    )

    folds: list[EntityColdStartFold] = []
    for fold_index in range(n_splits):
        raw_valid = base_assignments.loc[
            base_assignments["_validation_fold"] == fold_index
        ].copy()
        raw_train = base_assignments.loc[
            base_assignments["_validation_fold"] != fold_index
        ].copy()
        raw_train["_assigned_split"] = "train"
        raw_valid["_assigned_split"] = "valid"
        valid, eligibility, fold_excluded = _select_protocol_eligible_records(
            raw_train,
            raw_valid,
            protocol=protocol,
            split_name="valid",
            min_eval_records=min_eval_records,
            require_train_group=require_train_group,
        )
        if valid.empty:
            raise ValueError(
                f"{protocol} fold {fold_index} has no protocol-eligible validation records"
            )
        leakage_report = _build_entity_protocol_leakage_report(
            protocol,
            assigned={"train": raw_train, "valid": raw_valid},
            eligible={"valid": valid},
            require_train_group=require_train_group,
        )
        failed = leakage_report.loc[leakage_report["status"] != "PASS"]
        if not failed.empty:
            raise ValueError(
                f"{protocol} fold {fold_index} leakage check failed: "
                f"{failed.to_dict(orient='records')}"
            )
        excluded_records = _combine_excluded_records(
            pre_split_excluded, fold_excluded, columns=records.columns
        )
        folds.append(EntityColdStartFold(
            protocol=protocol,
            index=fold_index,
            train=_drop_split_helpers(raw_train),
            valid=_drop_split_helpers(valid),
            leakage_report=leakage_report,
            eligibility_report=eligibility,
            excluded_records=excluded_records,
            unit_assignments=unit_assignments.copy(),
        ))
    return folds


def _prepare_cold_start_records(
    records: pd.DataFrame,
    protocol: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _protocol_entity_column(protocol)  # validates the protocol name
    required = (
        "record_id", "group_id", "dataset_id", "keep_for_training", "rank_label",
        *COLD_START_IDENTITY_COLUMNS,
    )
    missing = [column for column in required if column not in records.columns]
    if missing:
        raise ValueError(f"records is missing cold-start identity column(s): {missing}")
    if records.empty:
        raise ValueError("records must be non-empty")
    record_ids = records["record_id"].astype(str)
    if records["record_id"].isna().any() or record_ids.duplicated().any():
        duplicates = record_ids[record_ids.duplicated()].tolist()
        raise ValueError(
            "cold-start records require non-null unique record_id values; "
            f"duplicates={duplicates[:10]}"
        )

    working = _trainable_records(records)
    if working.empty:
        raise ValueError("records contains no trainable rows")
    null_identity = working[list(COLD_START_IDENTITY_COLUMNS)].isna().any(axis=1)
    empty_identity = working[list(COLD_START_IDENTITY_COLUMNS)].apply(
        lambda column: column.astype(str).str.strip().eq("")
    ).any(axis=1)
    if (null_identity | empty_identity).any():
        bad = working.loc[null_identity | empty_identity, "record_id"].astype(str).tolist()
        raise ValueError(
            "cold-start identity fields must be non-null and non-empty; "
            f"record_id={bad[:10]}"
        )
    _validate_identity_mapping(
        working, "antibody_sequence_key", "antibody_cluster_id"
    )
    _validate_identity_mapping(
        working, "antigen_sequence_key", "antigen_cluster_id"
    )
    label_counts = (
        working.assign(_numeric_label=pd.to_numeric(working["rank_label"], errors="coerce"))
        .groupby(["group_id", "interaction_key"], sort=False)["_numeric_label"]
        .nunique()
    )
    conflicts = label_counts[label_counts > 1]
    if not conflicts.empty:
        raise ValueError(
            "identical group/interaction inputs have conflicting rank labels; "
            f"first={list(conflicts.index[:10])}"
        )

    trainable_ids = set(working["record_id"].astype(str))
    pre_split_excluded = records.loc[
        ~records["record_id"].astype(str).isin(trainable_ids)
    ].copy()
    if not pre_split_excluded.empty:
        pre_split_excluded["_assigned_split"] = "not_assigned"
        pre_split_excluded["protocol_exclusion_reason"] = "not_trainable"
    return working.copy(), pre_split_excluded


def _validate_identity_mapping(
    records: pd.DataFrame,
    exact_column: str,
    cluster_column: str,
) -> None:
    counts = records.groupby(exact_column, sort=False)[cluster_column].nunique()
    conflicts = counts[counts != 1]
    if not conflicts.empty:
        raise ValueError(
            f"{exact_column} must map to exactly one {cluster_column}; "
            f"conflicts={conflicts.index.astype(str).tolist()[:10]}"
        )


def _protocol_entity_column(protocol: str) -> str:
    if protocol == "antibody_cold_start":
        return "antibody_cluster_id"
    if protocol == "antigen_cold_start":
        return "antigen_cluster_id"
    raise ValueError(f"unsupported entity cold-start protocol: {protocol!r}")


def _derive_entity_components(records: pd.DataFrame, protocol: str) -> pd.Series:
    if protocol == "antibody_cold_start":
        link_columns = (
            "antibody_cluster_id", "antibody_sequence_key", "measurement_family_id"
        )
    elif protocol == "antigen_cold_start":
        link_columns = (
            "antigen_cluster_id", "antigen_sequence_key",
            "effective_antigen_input_hash", "measurement_family_id",
        )
    else:
        raise ValueError(f"unsupported entity cold-start protocol: {protocol!r}")

    n_records = len(records)
    parent = list(range(n_records))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for column in link_columns:
        for positions in records.groupby(column, sort=False).indices.values():
            positions = list(positions)
            for position in positions[1:]:
                union(int(positions[0]), int(position))

    root_to_record_ids: dict[int, list[str]] = {}
    for position, record_id in enumerate(records["record_id"].astype(str)):
        root_to_record_ids.setdefault(find(position), []).append(record_id)
    root_to_component = {
        root: "component_" + hashlib.sha256(
            "\n".join(sorted(record_ids)).encode("utf-8")
        ).hexdigest()[:16]
        for root, record_ids in root_to_record_ids.items()
    }
    return pd.Series(
        [root_to_component[find(position)] for position in range(n_records)],
        index=records.index,
        dtype="object",
    )


def _assign_component_splits(
    records: pd.DataFrame,
    *,
    train_units: set[str],
    valid_units: set[str],
    test_units: set[str],
) -> pd.DataFrame:
    assigned = records.copy()
    split_by_component = {
        **{unit: "train" for unit in train_units},
        **{unit: "valid" for unit in valid_units},
        **{unit: "test" for unit in test_units},
    }
    assigned["_assigned_split"] = assigned["_component_id"].map(split_by_component)
    if assigned["_assigned_split"].isna().any():
        raise RuntimeError("some entity components were not assigned to a split")
    return assigned


def _assign_weighted_units_to_folds(
    weights: dict[str, int],
    n_splits: int,
    seed: int,
) -> list[set[str]]:
    if len(weights) < n_splits:
        raise ValueError(
            f"n_splits={n_splits} exceeds the number of components={len(weights)}"
        )
    units = sorted(weights)
    random.Random(seed).shuffle(units)
    tie_order = {unit: index for index, unit in enumerate(units)}
    units.sort(key=lambda unit: (-weights[unit], tie_order[unit]))
    fold_units: list[set[str]] = [set() for _ in range(n_splits)]
    fold_weights = [0] * n_splits
    for unit in units:
        fold_index = min(range(n_splits), key=lambda index: (fold_weights[index], index))
        fold_units[fold_index].add(unit)
        fold_weights[fold_index] += int(weights[unit])
    return fold_units


def _select_protocol_eligible_records(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    *,
    protocol: str,
    split_name: str,
    min_eval_records: int,
    require_train_group: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reasons: dict[str, str] = {}
    if protocol == "antibody_cold_start":
        known_antigens = set(train["antigen_sequence_key"].astype(str))
        known_groups = set(train["group_id"].astype(str))
        for row in holdout[["record_id", "antigen_sequence_key", "group_id"]].itertuples(index=False):
            row_reasons = []
            if str(row.antigen_sequence_key) not in known_antigens:
                row_reasons.append("antigen_sequence_not_seen_in_train")
            if require_train_group and str(row.group_id) not in known_groups:
                row_reasons.append("group_not_seen_in_train")
            if row_reasons:
                reasons[str(row.record_id)] = ";".join(row_reasons)
    elif protocol == "antigen_cold_start":
        known_antibodies = set(train["antibody_cluster_id"].astype(str))
        for row in holdout[["record_id", "antibody_cluster_id"]].itertuples(index=False):
            if str(row.antibody_cluster_id) not in known_antibodies:
                reasons[str(row.record_id)] = "antibody_cluster_not_seen_in_train"
    else:
        raise ValueError(f"unsupported entity cold-start protocol: {protocol!r}")

    candidate_mask = ~holdout["record_id"].astype(str).isin(reasons)
    candidates = holdout.loc[candidate_mask].copy()
    eligible_groups: set[str] = set()
    group_failure_reasons: dict[str, str] = {}
    report_rows: list[dict[str, object]] = []
    for group_id, assigned_group in holdout.groupby(
        holdout["group_id"].astype(str), sort=True
    ):
        candidate_group = candidates.loc[
            candidates["group_id"].astype(str) == group_id
        ]
        labels = pd.to_numeric(candidate_group["rank_label"], errors="coerce")
        n_unique_inputs = int(candidate_group["antibody_sequence_key"].astype(str).nunique())
        n_unique_labels = int(labels.nunique())
        failures = []
        if len(candidate_group) < min_eval_records:
            failures.append(f"fewer_than_{min_eval_records}_protocol_records")
        if n_unique_inputs < 2:
            failures.append("fewer_than_2_unique_antibody_inputs")
        if n_unique_labels < 2:
            failures.append("fewer_than_2_unique_labels")
        status = "PASS" if not failures else "EXCLUDED"
        if not failures:
            eligible_groups.add(str(group_id))
        else:
            group_failure_reasons[str(group_id)] = ";".join(failures)
        report_rows.append({
            "split": split_name,
            "group_id": str(group_id),
            "n_assigned_records": int(len(assigned_group)),
            "n_candidate_records": int(len(candidate_group)),
            "n_eligible_records": int(len(candidate_group)) if not failures else 0,
            "n_unique_inputs": n_unique_inputs,
            "n_unique_labels": n_unique_labels,
            "status": status,
            "reason": ";".join(failures),
        })

    eligible = candidates.loc[
        candidates["group_id"].astype(str).isin(eligible_groups)
    ].copy()
    eligible_ids = set(eligible["record_id"].astype(str))
    excluded = holdout.loc[
        ~holdout["record_id"].astype(str).isin(eligible_ids)
    ].copy()
    if not excluded.empty:
        exclusion_reasons = []
        for row in excluded[["record_id", "group_id"]].itertuples(index=False):
            record_id, group_id = str(row.record_id), str(row.group_id)
            exclusion_reasons.append(
                reasons.get(
                    record_id,
                    "group_not_evaluable:" + group_failure_reasons.get(
                        group_id, "no_protocol_eligible_records"
                    ),
                )
            )
        excluded["protocol_exclusion_reason"] = exclusion_reasons
    report = pd.DataFrame(report_rows, columns=ELIGIBILITY_COLUMNS)
    return eligible, report, excluded


def _build_entity_protocol_leakage_report(
    protocol: str,
    *,
    assigned: dict[str, pd.DataFrame],
    eligible: dict[str, pd.DataFrame],
    require_train_group: bool,
) -> pd.DataFrame:
    rows = [
        _overlap_report("record_id_overlap", assigned, "record_id"),
        _overlap_report("measurement_family_overlap", assigned, "measurement_family_id"),
        _overlap_report("interaction_overlap", assigned, "interaction_key"),
    ]
    train = assigned["train"]
    if protocol == "antibody_cold_start":
        rows.extend([
            _overlap_report(
                "antibody_sequence_overlap", assigned, "antibody_sequence_key"
            ),
            _overlap_report(
                "antibody_cluster_overlap", assigned, "antibody_cluster_id"
            ),
        ])
        for split_name, split_records in eligible.items():
            rows.append(_coverage_report(
                f"{split_name}_antigen_seen_in_train",
                train,
                split_records,
                "antigen_sequence_key",
            ))
            if require_train_group:
                rows.append(_coverage_report(
                    f"{split_name}_group_seen_in_train",
                    train,
                    split_records,
                    "group_id",
                ))
    elif protocol == "antigen_cold_start":
        rows.extend([
            _overlap_report("antigen_sequence_overlap", assigned, "antigen_sequence_key"),
            _overlap_report("antigen_cluster_overlap", assigned, "antigen_cluster_id"),
            _overlap_report(
                "effective_antigen_input_overlap",
                assigned,
                "effective_antigen_input_hash",
            ),
        ])
        for split_name, split_records in eligible.items():
            rows.append(_coverage_report(
                f"{split_name}_antibody_seen_in_train",
                train,
                split_records,
                "antibody_cluster_id",
            ))
    else:
        raise ValueError(f"unsupported entity cold-start protocol: {protocol!r}")
    return pd.DataFrame(rows, columns=LEAKAGE_COLUMNS)


def _coverage_report(
    check_name: str,
    reference: pd.DataFrame,
    target: pd.DataFrame,
    column: str,
) -> dict[str, object]:
    missing = sorted(
        set(target[column].astype(str)) - set(reference[column].astype(str))
    )
    return {
        "check_name": check_name,
        "status": "PASS" if not missing else "FAIL",
        "n_violations": len(missing),
        "details": ", ".join(missing[:10]),
    }


def _build_unit_assignments(
    assigned: pd.DataFrame,
    *,
    entity_column: str,
    validation_folds: dict[str, int] | None,
) -> pd.DataFrame:
    rows = []
    for component_id, component in assigned.groupby("_component_id", sort=True):
        assigned_split = (
            str(component["_assigned_split"].iloc[0])
            if "_assigned_split" in component.columns
            else "development"
        )
        rows.append({
            "component_id": str(component_id),
            "assigned_split": assigned_split,
            "validation_fold": (
                None if validation_folds is None else validation_folds[str(component_id)]
            ),
            "n_records": int(len(component)),
            "n_entity_clusters": int(component[entity_column].astype(str).nunique()),
            "n_measurement_families": int(
                component["measurement_family_id"].astype(str).nunique()
            ),
        })
    return pd.DataFrame(rows, columns=ENTITY_UNIT_COLUMNS)


def _combine_excluded_records(
    *tables: pd.DataFrame,
    columns: pd.Index,
) -> pd.DataFrame:
    output_columns = list(columns) + ["_assigned_split", "protocol_exclusion_reason"]
    present = [table for table in tables if table is not None and not table.empty]
    if not present:
        return pd.DataFrame(columns=output_columns)
    combined = pd.concat(present, ignore_index=True, sort=False)
    for column in output_columns:
        if column not in combined.columns:
            combined[column] = None
    return combined[output_columns].sort_values(
        ["_assigned_split", "dataset_id", "record_id"], kind="stable"
    ).reset_index(drop=True)


def _drop_split_helpers(records: pd.DataFrame) -> pd.DataFrame:
    helper_columns = [
        column
        for column in ("_component_id", "_assigned_split", "_validation_fold")
        if column in records.columns
    ]
    return records.drop(columns=helper_columns).sort_values(
        ["dataset_id", "record_id"], kind="stable"
    ).reset_index(drop=True)


def _validate_fraction_and_eval_size(
    valid_fraction: float,
    test_fraction: float,
    min_eval_records: int,
) -> None:
    if valid_fraction < 0 or test_fraction < 0:
        raise ValueError("valid_fraction and test_fraction must be non-negative")
    if not (0.0 < valid_fraction + test_fraction < 1.0):
        raise ValueError("valid_fraction + test_fraction must be > 0 and < 1")
    if min_eval_records < 2:
        raise ValueError("min_eval_records must be at least 2")


def _derive_group_seed(seed: int, group_id: str) -> int:
    """Stable per-group seed derived from a base seed and `group_id`.

    Plain `hash()` is per-process-randomized (PYTHONHASHSEED), so it would
    make this split non-reproducible across runs/restarts. hashlib is not.
    """
    digest = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _concat_sorted(parts: list[pd.DataFrame], columns: pd.Index) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=columns)
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["dataset_id", "record_id"])
        .reset_index(drop=True)
    )


def _build_within_antigen_leakage_report(**splits: pd.DataFrame) -> pd.DataFrame:
    # Deliberately ONLY record_id_overlap: group_id and antibody-identity
    # crossing splits (via a DIFFERENT group) are both allowed by design
    # under this protocol (see build_within_antigen_split's docstring).
    return pd.DataFrame(
        [_overlap_report("record_id_overlap", splits, "record_id")],
        columns=LEAKAGE_COLUMNS,
    )


def _validate_inputs(
    records: pd.DataFrame,
    strategy: str,
    valid_fraction: float,
    test_fraction: float,
) -> None:
    required = ("record_id", "group_id", "keep_for_training", "rank_label")
    if strategy == "antigen_cluster_holdout_split":
        required = required + ("antigen_key",)
    missing = [column for column in required if column not in records.columns]
    if missing:
        raise ValueError(f"records is missing required column(s): {missing}")
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"strategy must be one of {sorted(VALID_STRATEGIES)}, got {strategy!r}")
    if valid_fraction < 0 or test_fraction < 0:
        raise ValueError("valid_fraction and test_fraction must be non-negative")
    if not (0.0 < valid_fraction + test_fraction < 1.0):
        raise ValueError("valid_fraction + test_fraction must be > 0 and < 1")
    if records["record_id"].isna().any():
        raise ValueError("records contains null record_id values")
    if records["record_id"].astype(str).duplicated().any():
        duplicated = records.loc[records["record_id"].astype(str).duplicated(), "record_id"].tolist()
        raise ValueError(f"records contains duplicate record_id values: {duplicated[:10]}")


def _split_by_record(
    records: pd.DataFrame,
    valid_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    record_ids = sorted(records["record_id"].astype(str).tolist())
    train_ids, valid_ids, test_ids = _partition_units(record_ids, valid_fraction, test_fraction, seed)
    return (
        _rows_for_values(records, "record_id", train_ids),
        _rows_for_values(records, "record_id", valid_ids),
        _rows_for_values(records, "record_id", test_ids),
    )


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


def _partition_units(
    units: list[str],
    valid_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[set[str], set[str], set[str]]:
    if len(units) < 3:
        raise ValueError(
            f"At least 3 split units are required to create train/valid/test, got {len(units)}"
        )

    shuffled = list(units)
    random.Random(seed).shuffle(shuffled)

    n_units = len(shuffled)
    n_valid = _fraction_count(n_units, valid_fraction)
    n_test = _fraction_count(n_units, test_fraction)
    while n_valid + n_test >= n_units:
        if n_test >= n_valid and n_test > 0:
            n_test -= 1
        elif n_valid > 0:
            n_valid -= 1
        else:
            break

    if n_valid == 0 and valid_fraction > 0:
        n_valid = 1
    if n_test == 0 and test_fraction > 0 and n_valid + n_test < n_units - 1:
        n_test = 1
    while n_valid + n_test >= n_units:
        if n_test >= n_valid and n_test > 0:
            n_test -= 1
        else:
            n_valid -= 1

    test_units = set(shuffled[:n_test])
    valid_units = set(shuffled[n_test:n_test + n_valid])
    train_units = set(shuffled[n_test + n_valid:])
    if not train_units or not valid_units or not test_units:
        raise ValueError(
            "Split fractions produced an empty train, valid, or test split; "
            f"n_units={n_units}, n_valid={n_valid}, n_test={n_test}"
        )
    return train_units, valid_units, test_units


def _fraction_count(n_units: int, fraction: float) -> int:
    if fraction <= 0:
        return 0
    return max(1, int(round(n_units * fraction)))


def _partition_weighted_units(
    weights: dict[str, int],
    valid_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[set[str], set[str], set[str]]:
    """Partition group ids while keeping over-target groups in train."""
    if len(weights) < 3:
        raise ValueError(
            f"At least 3 split units are required to create train/valid/test, got {len(weights)}"
        )
    if any(weight < 1 for weight in weights.values()):
        raise ValueError("split unit weights must be positive")

    total_weight = sum(weights.values())
    holdout_limit = _fraction_count(total_weight, max(valid_fraction, test_fraction))
    pinned_train = {unit for unit, weight in weights.items() if weight > holdout_limit}
    eligible = sorted(set(weights) - pinned_train)
    if len(eligible) < 2:
        return _partition_units(sorted(weights), valid_fraction, test_fraction, seed)

    n_valid = _fraction_count(len(weights), valid_fraction)
    n_test = _fraction_count(len(weights), test_fraction)
    reserve_for_train = 0 if pinned_train else 1
    while n_valid + n_test > len(eligible) - reserve_for_train:
        if n_test >= n_valid and n_test > 1:
            n_test -= 1
        elif n_valid > 1:
            n_valid -= 1
        else:
            break

    if n_valid < 1 or n_test < 1 or n_valid + n_test > len(eligible):
        return _partition_units(sorted(weights), valid_fraction, test_fraction, seed)

    units = list(eligible)
    rng = random.Random(seed)
    rng.shuffle(units)
    holdout_units = units[:n_test + n_valid]
    test_units, valid_units = _split_holdout_by_weight(
        holdout_units, weights, n_test=n_test, n_valid=n_valid
    )
    train_units = (set(units[n_test + n_valid:]) | pinned_train)

    if not train_units or not valid_units or not test_units:
        raise ValueError(
            "Split fractions produced an empty train, valid, or test split; "
            f"n_units={len(weights)}, n_valid={len(valid_units)}, n_test={len(test_units)}"
        )
    return train_units, valid_units, test_units


def _split_holdout_by_weight(
    holdout_units: list[str],
    weights: dict[str, int],
    n_test: int,
    n_valid: int,
) -> tuple[set[str], set[str]]:
    test_units: set[str] = set()
    valid_units: set[str] = set()
    test_weight = 0
    valid_weight = 0

    for unit in sorted(holdout_units, key=lambda item: weights[item], reverse=True):
        if len(test_units) >= n_test:
            valid_units.add(unit)
            valid_weight += weights[unit]
        elif len(valid_units) >= n_valid:
            test_units.add(unit)
            test_weight += weights[unit]
        elif test_weight <= valid_weight:
            test_units.add(unit)
            test_weight += weights[unit]
        else:
            valid_units.add(unit)
            valid_weight += weights[unit]

    return test_units, valid_units


def _rows_for_values(records: pd.DataFrame, column: str, values: set[str]) -> pd.DataFrame:
    mask = records[column].astype(str).isin(values)
    return records.loc[mask].sort_values(["dataset_id", "record_id"]).reset_index(drop=True)


def _build_summary(strategy: str, **splits: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split_name, split_records in splits.items():
        trainable = _trainable_records(split_records)
        rows.append({
            "split": split_name,
            "strategy": strategy,
            "n_records": len(split_records),
            "n_trainable_records": len(trainable),
            "n_groups": int(split_records["group_id"].nunique()) if "group_id" in split_records else 0,
            "n_trainable_groups": int(trainable["group_id"].nunique()) if not trainable.empty else 0,
            "n_spearman_eligible_groups": _count_spearman_eligible_groups(trainable),
            "label_kind_counts": _value_counts_json(split_records, "label_kind"),
            "antigen_source_counts": _value_counts_json(split_records, "antigen_source"),
        })
    return pd.DataFrame(rows, columns=SPLIT_COLUMNS)


def _build_leakage_report(strategy: str, **splits: pd.DataFrame) -> pd.DataFrame:
    rows = [_overlap_report("record_id_overlap", splits, "record_id")]
    if strategy == "group_holdout_split":
        rows.append(_overlap_report("group_id_overlap", splits, "group_id"))
    elif strategy == "antigen_cluster_holdout_split":
        # group_id_overlap is implied (every group_id maps to exactly one
        # antigen_key, which maps to exactly one antigen_cluster_id) but we
        # check it explicitly anyway, the same defense-in-depth reasoning
        # group_holdout_split uses for its own structurally-guaranteed
        # partition. antigen_cluster_overlap is the check that actually
        # matters here -- it's the whole reason this strategy exists.
        rows.append(_overlap_report("group_id_overlap", splits, "group_id"))
        rows.append(_overlap_report("antigen_cluster_overlap", splits, "_antigen_cluster_id"))
    return pd.DataFrame(rows, columns=LEAKAGE_COLUMNS)


def _overlap_report(check_name: str, splits: dict[str, pd.DataFrame], column: str) -> dict[str, object]:
    violations: list[str] = []
    for left_name, right_name in combinations(splits, 2):
        left = set(splits[left_name][column].astype(str))
        right = set(splits[right_name][column].astype(str))
        overlap = sorted(left & right)
        if overlap:
            violations.append(f"{left_name}/{right_name}: {overlap[:10]}")

    return {
        "check_name": check_name,
        "status": "PASS" if not violations else "FAIL",
        "n_violations": len(violations),
        "details": "; ".join(violations),
    }


def _trainable_records(records: pd.DataFrame) -> pd.DataFrame:
    keep = records["keep_for_training"].apply(_parse_bool)
    labels = pd.to_numeric(records["rank_label"], errors="coerce")
    finite = labels.apply(lambda value: pd.notna(value) and value not in (float("inf"), float("-inf")))
    return records.loc[keep & finite].copy()


def _parse_bool(value: object) -> bool:
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


def _count_spearman_eligible_groups(records: pd.DataFrame) -> int:
    if records.empty:
        return 0
    count = 0
    for _, group in records.groupby("group_id"):
        if pd.to_numeric(group["rank_label"], errors="coerce").nunique() >= 2:
            count += 1
    return count


def _value_counts_json(records: pd.DataFrame, column: str) -> str:
    if column not in records:
        return "{}"
    counts = records[column].fillna("<NA>").astype(str).value_counts().sort_index().to_dict()
    return json.dumps(counts, sort_keys=True)
