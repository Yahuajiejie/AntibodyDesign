"""Within-antigen antibody holdout (auxiliary, group-local) split."""
from __future__ import annotations

import pandas as pd

from ..record_filter import antibody_sequence_hashes
from .common import _concat_sorted, _derive_group_seed, _partition_weighted_units
from .audits import (
    PINNED_GROUPS_COLUMNS,
    _build_summary,
    _build_within_antigen_leakage_report,
)
from .results import WithinAntigenSplitResult


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
