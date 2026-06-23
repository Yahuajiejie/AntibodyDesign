"""Dual cold-start split: unseen antigen AND unseen antibody (section 5.5).

Answers: can the model rank previously unseen antibody clusters against an
antigen cluster that was also unseen during training?

This protocol is deliberately NOT routed through the antibody/antigen engine in
``entity_cold_start.py``:

* there is no seen-in-train eligibility filter -- both antibody clusters and
  antigen clusters must remain unseen;
* the indivisible assignment unit is a multi-entity connected component built
  from four relationships;
* statistical feasibility is audited before a valid split is claimed.

Genuinely shared helpers (component assignment, overlap/eligibility reports,
artifact writers) are reused from ``common``/``audits``/``results``.
"""
from __future__ import annotations

from itertools import combinations

import pandas as pd

from ..annotations import join_entity_annotations, join_representation_annotations
from .audits import (
    LEAKAGE_COLUMNS,
    _build_summary,
    _build_unit_assignments,
    _overlap_report,
    build_group_eligibility,
)
from .common import (
    _assign_component_splits,
    _combine_excluded_records,
    _drop_split_helpers,
    _partition_weighted_units,
    _trainable_records,
    _validate_fraction_and_eval_size,
    derive_link_components,
)
from .results import DualColdStartSplitResult

# Four relationships whose transitive closure forms one indivisible component.
DUAL_LINK_COLUMNS = (
    "antibody_cluster_id",
    "antigen_cluster_id",
    "measurement_family_id",
    "interaction_key",
)
_DUAL_IDENTITY_COLUMNS = (
    "measurement_family_id",
    "antibody_sequence_key",
    "antibody_cluster_id",
    "antigen_sequence_key",
    "antigen_cluster_id",
    "interaction_key",
)
# Entity overlaps that must be zero across train/valid/test.
_LEAKAGE_CHECKS = (
    ("record_id_overlap", "record_id"),
    ("measurement_family_overlap", "measurement_family_id"),
    ("antibody_sequence_overlap", "antibody_sequence_key"),
    ("antibody_cluster_overlap", "antibody_cluster_id"),
    ("antigen_sequence_overlap", "antigen_sequence_key"),
    ("antigen_cluster_overlap", "antigen_cluster_id"),
    ("interaction_overlap", "interaction_key"),
)
COMPONENT_SUMMARY_COLUMNS = (
    "component_id", "n_records", "n_groups",
    "n_antibody_sequence_keys", "n_antibody_clusters",
    "n_antigen_sequence_keys", "n_antigen_clusters",
    "n_measurement_families", "n_interactions",
    "record_fraction", "group_fraction",
    "antibody_cluster_fraction", "antigen_cluster_fraction",
)


def build_dual_cold_start_split(
    records: pd.DataFrame,
    entity_annotations: pd.DataFrame,
    *,
    valid_fraction: float,
    test_fraction: float,
    seed: int,
    min_eval_records: int,
    representation_annotations: pd.DataFrame | None = None,
) -> DualColdStartSplitResult:
    """Build a dual cold-start train/valid/test split (section 5.5).

    Args:
        records: Base processed records (entity identity NOT required inline).
        entity_annotations: Required separate annotation table keyed by
            ``record_id`` (see ``annotations.load_entity_annotations``).
        valid_fraction: Fraction of records (by component weight) for valid.
        test_fraction: Fraction of records (by component weight) for test.
        seed: Deterministic assignment seed.
        min_eval_records: Minimum protocol-eligible records per evaluation group.
        representation_annotations: Optional; when provided the antigen
            effective-input hash is materialized and checked for cross-split
            collisions (fail-loud). When absent, effective-input is not audited.

    Raises:
        ValueError: On invalid inputs, missing annotation columns, a leakage
            failure (entity overlap, group overlap, or effective-input
            collision), or when the indivisible components make a non-empty,
            evaluable valid/test split impossible (protocol-infeasible).
    """
    _validate_fraction_and_eval_size(valid_fraction, test_fraction, min_eval_records)
    if entity_annotations is None:
        raise ValueError("dual_cold_start requires entity_annotations")

    working = join_entity_annotations(records, entity_annotations)
    effective_audited = representation_annotations is not None
    if effective_audited:
        working = join_representation_annotations(
            working,
            representation_annotations,
            sequence_type="antigen",
            sequence_key_column="antigen_sequence_key",
            out_column="effective_antigen_input_hash",
        )

    working, pre_split_excluded = _prepare(working, base_columns=records.columns)
    working["_component_id"] = derive_link_components(working, DUAL_LINK_COLUMNS)
    component_summary = _build_component_summary(working)

    weights = working.groupby("_component_id", sort=True).size().astype(int).to_dict()
    try:
        train_units, valid_units, test_units = _partition_weighted_units(
            weights, valid_fraction, test_fraction, seed
        )
    except ValueError as error:
        raise ValueError(_infeasible_message(
            component_summary, valid_fraction, test_fraction, min_eval_records,
            reason=str(error),
        )) from error

    assigned = _assign_component_splits(
        working, train_units=train_units, valid_units=valid_units, test_units=test_units
    )
    raw_train = assigned.loc[assigned["_assigned_split"] == "train"].copy()
    raw_valid = assigned.loc[assigned["_assigned_split"] == "valid"].copy()
    raw_test = assigned.loc[assigned["_assigned_split"] == "test"].copy()

    valid, valid_report, valid_excluded = build_group_eligibility(
        raw_valid, split_name="valid", min_eval_records=min_eval_records
    )
    test, test_report, test_excluded = build_group_eligibility(
        raw_test, split_name="test", min_eval_records=min_eval_records
    )
    if valid.empty or test.empty:
        raise ValueError(_infeasible_message(
            component_summary, valid_fraction, test_fraction, min_eval_records,
            reason="no evaluable valid/test group remained after component assignment",
        ))

    leakage_report = _build_dual_leakage_report(
        {"train": raw_train, "valid": raw_valid, "test": raw_test},
        audit_effective=effective_audited,
    )
    failed = leakage_report.loc[leakage_report["status"] != "PASS"]
    if not failed.empty:
        _raise_leakage(failed, assigned, representation_annotations, effective_audited)

    eligibility_report = pd.concat([valid_report, test_report], ignore_index=True)
    excluded_records = _combine_excluded_records(
        pre_split_excluded, valid_excluded, test_excluded, columns=records.columns
    )
    unit_assignments = _build_unit_assignments(
        assigned, entity_column="antigen_cluster_id", validation_folds=None
    )

    train = _drop_split_helpers(raw_train)
    valid = _drop_split_helpers(valid)
    test = _drop_split_helpers(test)
    summary = _build_summary("dual_cold_start", train=train, valid=valid, test=test)

    return DualColdStartSplitResult(
        protocol="dual_cold_start",
        train=train,
        valid=valid,
        test=test,
        summary=summary,
        leakage_report=leakage_report,
        eligibility_report=eligibility_report,
        excluded_records=excluded_records,
        unit_assignments=unit_assignments,
        component_summary=component_summary,
    )


def _prepare(
    working: pd.DataFrame,
    *,
    base_columns: pd.Index,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate columns/ids, filter to trainable rows, capture pre-split drops."""
    required = (
        "record_id", "group_id", "dataset_id", "keep_for_training", "rank_label",
        *_DUAL_IDENTITY_COLUMNS,
    )
    missing = [column for column in required if column not in working.columns]
    if missing:
        raise ValueError(
            f"records/entity annotations missing dual cold-start column(s): {missing}"
        )
    if working.empty:
        raise ValueError("records must be non-empty")
    record_ids = working["record_id"].astype(str)
    if working["record_id"].isna().any() or record_ids.duplicated().any():
        duplicates = record_ids[record_ids.duplicated()].tolist()
        raise ValueError(
            "dual cold-start requires non-null unique record_id values; "
            f"duplicates={duplicates[:10]}"
        )

    trainable = _trainable_records(working)
    if trainable.empty:
        raise ValueError("records contains no trainable rows")
    null_identity = trainable[list(_DUAL_IDENTITY_COLUMNS)].isna().any(axis=1)
    empty_identity = trainable[list(_DUAL_IDENTITY_COLUMNS)].apply(
        lambda column: column.astype(str).str.strip().eq("")
    ).any(axis=1)
    if (null_identity | empty_identity).any():
        bad = trainable.loc[null_identity | empty_identity, "record_id"].astype(str).tolist()
        raise ValueError(
            "dual cold-start identity fields must be non-null and non-empty; "
            f"record_id={bad[:10]}"
        )

    trainable_ids = set(trainable["record_id"].astype(str))
    dropped_mask = ~working["record_id"].astype(str).isin(trainable_ids)
    pre_split_excluded = working.loc[dropped_mask, list(base_columns)].copy()
    if not pre_split_excluded.empty:
        pre_split_excluded["_assigned_split"] = "not_assigned"
        pre_split_excluded["protocol_exclusion_reason"] = "not_trainable"
    return trainable.copy(), pre_split_excluded


def _build_component_summary(working: pd.DataFrame) -> pd.DataFrame:
    """Per-component feasibility statistics, sorted largest (by records) first."""
    total_records = len(working)
    total_groups = working["group_id"].astype(str).nunique()
    total_ab_clusters = working["antibody_cluster_id"].astype(str).nunique()
    total_ag_clusters = working["antigen_cluster_id"].astype(str).nunique()

    def _frac(value: int, total: int) -> float:
        return float(value) / float(total) if total else 0.0

    rows = []
    for component_id, component in working.groupby("_component_id", sort=True):
        n_records = int(len(component))
        n_groups = int(component["group_id"].astype(str).nunique())
        n_ab_clusters = int(component["antibody_cluster_id"].astype(str).nunique())
        n_ag_clusters = int(component["antigen_cluster_id"].astype(str).nunique())
        rows.append({
            "component_id": str(component_id),
            "n_records": n_records,
            "n_groups": n_groups,
            "n_antibody_sequence_keys": int(
                component["antibody_sequence_key"].astype(str).nunique()
            ),
            "n_antibody_clusters": n_ab_clusters,
            "n_antigen_sequence_keys": int(
                component["antigen_sequence_key"].astype(str).nunique()
            ),
            "n_antigen_clusters": n_ag_clusters,
            "n_measurement_families": int(
                component["measurement_family_id"].astype(str).nunique()
            ),
            "n_interactions": int(component["interaction_key"].astype(str).nunique()),
            "record_fraction": _frac(n_records, total_records),
            "group_fraction": _frac(n_groups, total_groups),
            "antibody_cluster_fraction": _frac(n_ab_clusters, total_ab_clusters),
            "antigen_cluster_fraction": _frac(n_ag_clusters, total_ag_clusters),
        })
    summary = pd.DataFrame(rows, columns=COMPONENT_SUMMARY_COLUMNS)
    return summary.sort_values(
        ["record_fraction", "component_id"], ascending=[False, True]
    ).reset_index(drop=True)


def _build_dual_leakage_report(
    splits: dict[str, pd.DataFrame],
    *,
    audit_effective: bool,
) -> pd.DataFrame:
    rows = [_overlap_report(name, splits, column) for name, column in _LEAKAGE_CHECKS]
    # group_id is audited but is NOT a component link: if entity isolation still
    # leaves a group spanning splits, this fails loudly rather than silently
    # changing component construction.
    rows.append(_overlap_report("group_id_overlap", splits, "group_id"))
    if audit_effective:
        rows.append(_overlap_report(
            "effective_antigen_input_overlap", splits, "effective_antigen_input_hash"
        ))
    return pd.DataFrame(rows, columns=LEAKAGE_COLUMNS)


def _raise_leakage(
    failed: pd.DataFrame,
    assigned: pd.DataFrame,
    representation_annotations: pd.DataFrame | None,
    effective_audited: bool,
) -> None:
    names = set(failed["check_name"])
    if effective_audited and "effective_antigen_input_overlap" in names:
        diagnostics = _effective_collision_diagnostics(
            assigned, representation_annotations
        )
        raise ValueError(
            "dual_cold_start effective-input collision across splits: " + diagnostics
        )
    raise ValueError(
        f"dual_cold_start leakage check failed: {failed.to_dict(orient='records')}"
    )


def _effective_collision_diagnostics(
    assigned: pd.DataFrame,
    representation_annotations: pd.DataFrame | None,
) -> str:
    by_split = {
        name: assigned.loc[assigned["_assigned_split"] == name]
        for name in ("train", "valid", "test")
    }
    rep_id_map: dict[str, str] = {}
    if (
        representation_annotations is not None
        and "representation_id" in representation_annotations.columns
    ):
        rep = representation_annotations.loc[
            representation_annotations["sequence_type"].astype(str) == "antigen"
        ]
        rep_id_map = dict(zip(
            rep["effective_input_hash"].astype(str),
            rep["representation_id"].astype(str),
        ))
    split_hashes = {
        name: set(frame["effective_antigen_input_hash"].astype(str))
        for name, frame in by_split.items()
    }
    parts = []
    for left, right in combinations(("train", "valid", "test"), 2):
        for collision in sorted(split_hashes[left] & split_hashes[right]):
            left_keys = sorted(set(by_split[left].loc[
                by_split[left]["effective_antigen_input_hash"].astype(str) == collision,
                "antigen_sequence_key",
            ].astype(str)))
            right_keys = sorted(set(by_split[right].loc[
                by_split[right]["effective_antigen_input_hash"].astype(str) == collision,
                "antigen_sequence_key",
            ].astype(str)))
            parts.append(
                f"hash={collision} representation_id={rep_id_map.get(collision, 'n/a')} "
                f"split_pair={left}/{right} {left}_keys={left_keys[:5]} "
                f"{right}_keys={right_keys[:5]}"
            )
    return "; ".join(parts)


def _infeasible_message(
    component_summary: pd.DataFrame,
    valid_fraction: float,
    test_fraction: float,
    min_eval_records: int,
    *,
    reason: str,
) -> str:
    if component_summary.empty:
        detail = "no components"
    else:
        largest = component_summary.iloc[0]
        detail = (
            f"largest component {largest['component_id']} holds "
            f"record_fraction={largest['record_fraction']:.3f}, "
            f"group_fraction={largest['group_fraction']:.3f}, "
            f"antibody_cluster_fraction={largest['antibody_cluster_fraction']:.3f}, "
            f"antigen_cluster_fraction={largest['antigen_cluster_fraction']:.3f}"
        )
    return (
        "dual_cold_start is protocol-infeasible for "
        f"valid_fraction={valid_fraction}, test_fraction={test_fraction}, "
        f"min_eval_records={min_eval_records}: {reason}. "
        f"n_components={len(component_summary)}; {detail}"
    )
