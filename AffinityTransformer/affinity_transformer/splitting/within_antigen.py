"""Within-antigen antibody holdout split.

The split is antigen-context local, not group-local: records are first grouped
by an antigen context (prefer antigen_cluster_id, then antigen_sequence_key),
and antibody candidates are held out inside each context.  group_id is
deliberately not used as the antigen context because
multiple groups can describe the same or similar antigen.
"""
from __future__ import annotations

import pandas as pd

from ..record_filter import antibody_sequence_hashes
from .common import (
    _concat_sorted,
    _derive_group_seed,
    _partition_weighted_units,
    derive_link_components,
)
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
    """Build a known-antigen, new-antibody split.

    The protocol is scoped by antigen context, not by raw ``group_id``.  Records
    with the same or similar antigen are pooled first; within each antigen
    context, antibody candidates/components are partitioned into train, valid
    and test.  This answers the intended question: for an antigen context that
    is still present in train, how well does the model rank held-out antibody
    candidates for that context?

    Antigen context priority is:

    1. ``antigen_cluster_id`` (preferred; similarity-aware);
    2. ``antigen_sequence_key`` (exact-sequence context).

    ``group_id`` and legacy ``antigen_key`` are not used as fallback antigen
    contexts, because that would silently recreate an exact/group-local split
    and miss antigen-similarity leakage.

    Within an antigen context, antibody split units are derived from
    ``antibody_cluster_id`` if present, then ``antibody_sequence_key`` if
    present, then exact antibody-sequence hashes.  ``measurement_family_id`` and
    ``interaction_key`` do not define antigen contexts; when present, they only
    link records inside the current antigen context so technical duplicates or
    repeated interactions cannot cross split boundaries.

    Args:
        records: Standard processed records. Must include one antigen-context
            column (``antigen_cluster_id`` or ``antigen_sequence_key``). If
            antibody cluster/key columns are absent, the exact sequence columns
            ``heavy_chain``, ``light_chain`` and ``single_chain_sequence`` are
            required.
        valid_fraction: Fraction of each antigen context's antibody components
            reserved for validation.
        test_fraction: Fraction of each antigen context's antibody components
            reserved for test.
        seed: Base random seed. Each antigen context derives its own stable
            seed from this value and the context id.
        min_eval_records: Minimum number of records an antigen context must
            contribute to BOTH valid and test to be split; smaller contexts are
            routed entirely to train and recorded in ``pinned_groups``.

    Raises:
        ValueError: If required columns are missing, fractions are invalid,
            no antigen context could be split, or leakage checks fail.
    """
    required = ("record_id", "group_id", "keep_for_training", "rank_label")
    missing = [column for column in required if column not in records.columns]
    if missing:
        raise ValueError(f"records is missing required column(s): {missing}")
    antigen_context_column = _select_antigen_context_column(records)
    antibody_identity_column = _select_antibody_identity_column(records)
    if antibody_identity_column is None:
        sequence_columns = ("heavy_chain", "light_chain", "single_chain_sequence")
        sequence_missing = [
            column for column in sequence_columns if column not in records.columns
        ]
        if sequence_missing:
            raise ValueError(
                "records is missing required column(s) for antibody sequence "
                f"fallback: {sequence_missing}"
            )
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
    working["_antigen_context_id"] = _prefixed_non_empty_values(
        working, antigen_context_column
    )
    if antibody_identity_column is None:
        working["_antibody_unit"] = antibody_sequence_hashes(working)
    else:
        working["_antibody_unit"] = _prefixed_non_empty_values(
            working, antibody_identity_column
        )

    train_parts: list[pd.DataFrame] = []
    valid_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    pinned_rows: list[dict[str, object]] = []

    grouped = working.groupby("_antigen_context_id", sort=True)
    for antigen_context_id, context_df in grouped:
        context_df = context_df.copy()
        context_df["_within_antigen_component_id"] = _derive_context_components(
            context_df
        )
        weights = (
            context_df.groupby("_within_antigen_component_id")
            .size()
            .astype(int)
            .to_dict()
        )

        if len(weights) < 3:
            train_parts.append(context_df)
            pinned_rows.append({
                "group_id": _display_groups(context_df),
                "n_records": len(context_df),
                "n_antibody_units": len(weights),
                "reason": (
                    f"antigen_context_id={antigen_context_id}; fewer than 3 "
                    "distinct antibody components"
                ),
            })
            continue

        context_seed = _derive_group_seed(seed, str(antigen_context_id))
        try:
            train_units, valid_units, test_units = _partition_weighted_units(
                weights, valid_fraction, test_fraction, context_seed
            )
        except ValueError as error:
            train_parts.append(context_df)
            pinned_rows.append({
                "group_id": _display_groups(context_df),
                "n_records": len(context_df),
                "n_antibody_units": len(weights),
                "reason": (
                    f"antigen_context_id={antigen_context_id}; "
                    f"partitioning failed: {error}"
                ),
            })
            continue

        n_valid_records = sum(weights[unit] for unit in valid_units)
        n_test_records = sum(weights[unit] for unit in test_units)
        if n_valid_records < min_eval_records or n_test_records < min_eval_records:
            train_parts.append(context_df)
            pinned_rows.append({
                "group_id": _display_groups(context_df),
                "n_records": len(context_df),
                "n_antibody_units": len(weights),
                "reason": (
                    f"antigen_context_id={antigen_context_id}; "
                    f"valid/test would have fewer than min_eval_records={min_eval_records} "
                    f"records (got valid={n_valid_records}, test={n_test_records})"
                ),
            })
            continue

        unit = context_df["_within_antigen_component_id"]
        train_parts.append(context_df.loc[unit.isin(train_units)])
        valid_parts.append(context_df.loc[unit.isin(valid_units)])
        test_parts.append(context_df.loc[unit.isin(test_units)])

    train = _concat_sorted(train_parts, working.columns)
    valid = _concat_sorted(valid_parts, working.columns)
    test = _concat_sorted(test_parts, working.columns)
    if valid.empty or test.empty:
        raise ValueError(
            "No antigen context had enough antibody components to populate both "
            "valid and test under the within-antigen protocol; lower "
            "min_eval_records or valid_fraction/test_fraction, or provide richer "
            "antigen/antibody annotations."
        )

    leakage_report = _build_within_antigen_leakage_report(train=train, valid=valid, test=test)
    failed = leakage_report[leakage_report["status"] != "PASS"]
    if not failed.empty:
        raise ValueError(f"Split leakage check failed: {failed.to_dict(orient='records')}")

    summary = _build_summary("within_antigen_split", train=train, valid=valid, test=test)
    pinned_groups = pd.DataFrame(pinned_rows, columns=PINNED_GROUPS_COLUMNS)

    return WithinAntigenSplitResult(
        train=_drop_within_antigen_helpers(train),
        valid=_drop_within_antigen_helpers(valid),
        test=_drop_within_antigen_helpers(test),
        summary=summary,
        leakage_report=leakage_report,
        pinned_groups=pinned_groups,
    )


def _select_antigen_context_column(records: pd.DataFrame) -> str:
    for column in ("antigen_cluster_id", "antigen_sequence_key"):
        if column in records.columns:
            return column
    raise ValueError(
        "records must include an antigen context column: antigen_cluster_id "
        "(preferred) or antigen_sequence_key"
    )


def _select_antibody_identity_column(records: pd.DataFrame) -> str | None:
    for column in ("antibody_cluster_id", "antibody_sequence_key"):
        if column in records.columns:
            return column
    return None


def _prefixed_non_empty_values(records: pd.DataFrame, column: str) -> pd.Series:
    values = records[column]
    missing_mask = values.isna() | values.astype(str).str.strip().eq("")
    if missing_mask.any():
        examples = records.loc[missing_mask, "record_id"].astype(str).head(10).tolist()
        raise ValueError(
            f"records contains null/empty {column} values for record_id(s): {examples}"
        )
    return column + ":" + values.astype(str)


def _derive_context_components(context_df: pd.DataFrame) -> pd.Series:
    component_input = context_df[["record_id", "_antibody_unit"]].copy()
    link_columns = ["_antibody_unit"]
    for column in ("measurement_family_id", "interaction_key"):
        if column not in context_df.columns:
            continue
        component_input[column] = _nullable_link_values(context_df[column])
        link_columns.append(column)
    return derive_link_components(component_input, tuple(link_columns))


def _nullable_link_values(values: pd.Series) -> pd.Series:
    normalized = values.astype("string").str.strip()
    return normalized.mask(normalized.eq(""))


def _display_groups(context_df: pd.DataFrame, limit: int = 8) -> str:
    groups = sorted(context_df["group_id"].astype(str).unique())
    if len(groups) <= limit:
        return ",".join(groups)
    shown = ",".join(groups[:limit])
    return f"{shown},...(+{len(groups) - limit})"


def _drop_within_antigen_helpers(records: pd.DataFrame) -> pd.DataFrame:
    helper_columns = [
        column for column in (
            "_antigen_context_id",
            "_antibody_unit",
            "_within_antigen_component_id",
        )
        if column in records.columns
    ]
    return records.drop(columns=helper_columns)
