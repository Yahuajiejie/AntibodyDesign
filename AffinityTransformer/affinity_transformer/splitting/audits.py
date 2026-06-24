"""Overlap, leakage, eligibility and summary report builders."""
from __future__ import annotations

import json
from itertools import combinations

import pandas as pd

from .common import _trainable_records


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


def _build_entity_protocol_leakage_report(
    protocol: str,
    *,
    assigned: dict[str, pd.DataFrame],
    eligible: dict[str, pd.DataFrame],
    require_train_group: bool,
    strict_known_counterpart: bool = True,
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
        if strict_known_counterpart:
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
        ])
        # effective-input overlap is audited only when the hash was materialized
        # (representation annotations supplied, or an embedded column present).
        # When absent it is intentionally skipped; callers note this in the
        # split manifest.
        if "effective_antigen_input_hash" in train.columns:
            rows.append(_overlap_report(
                "effective_antigen_input_overlap",
                assigned,
                "effective_antigen_input_hash",
            ))
        if strict_known_counterpart:
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


def build_group_eligibility(
    holdout: pd.DataFrame,
    *,
    split_name: str,
    min_eval_records: int,
    input_column: str = "antibody_sequence_key",
    label_column: str = "rank_label",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Filter a holdout to evaluable groups (no seen-in-train logic).

    A ``group_id`` is evaluable when it has at least ``min_eval_records``
    records, two distinct ``input_column`` values, and two distinct numeric
    ``label_column`` values. Records in non-evaluable groups are returned in the
    excluded frame with a ``protocol_exclusion_reason``.

    Returns:
        ``(eligible, report, excluded)`` where ``report`` uses
        ``ELIGIBILITY_COLUMNS`` and ``excluded`` carries the original columns
        plus ``protocol_exclusion_reason``.
    """
    eligible_groups: set[str] = set()
    group_failure_reasons: dict[str, str] = {}
    report_rows: list[dict[str, object]] = []
    for group_id, assigned_group in holdout.groupby(
        holdout["group_id"].astype(str), sort=True
    ):
        labels = pd.to_numeric(assigned_group[label_column], errors="coerce")
        n_unique_inputs = int(assigned_group[input_column].astype(str).nunique())
        n_unique_labels = int(labels.nunique())
        failures = []
        if len(assigned_group) < min_eval_records:
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
            "n_candidate_records": int(len(assigned_group)),
            "n_eligible_records": int(len(assigned_group)) if not failures else 0,
            "n_unique_inputs": n_unique_inputs,
            "n_unique_labels": n_unique_labels,
            "status": status,
            "reason": ";".join(failures),
        })

    eligible = holdout.loc[
        holdout["group_id"].astype(str).isin(eligible_groups)
    ].copy()
    eligible_ids = set(eligible["record_id"].astype(str))
    excluded = holdout.loc[
        ~holdout["record_id"].astype(str).isin(eligible_ids)
    ].copy()
    if not excluded.empty:
        excluded["protocol_exclusion_reason"] = [
            "group_not_evaluable:" + group_failure_reasons.get(
                str(group_id), "no_protocol_eligible_records"
            )
            for group_id in excluded["group_id"].astype(str)
        ]
    report = pd.DataFrame(report_rows, columns=ELIGIBILITY_COLUMNS)
    return eligible, report, excluded


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


def _build_within_antigen_leakage_report(**splits: pd.DataFrame) -> pd.DataFrame:
    """Leakage checks for antigen-context-local antibody holdout.

    Antigen contexts themselves are expected to appear in train and holdout:
    this is a known-antigen protocol.  What must not cross split boundaries is
    the antibody unit/component *within the same antigen context*.  Optional
    measurement-family and interaction checks are scoped the same way so
    technical duplicates only constrain the antigen context they belong to.
    """
    scoped = {
        split_name: split_records.assign(
            _context_antibody_unit=_scoped_values(split_records, "_antibody_unit"),
            _context_component_id=_scoped_values(
                split_records, "_within_antigen_component_id"
            ),
        )
        for split_name, split_records in splits.items()
    }
    rows = [
        _overlap_report("record_id_overlap", scoped, "record_id"),
        _overlap_report(
            "within_antigen_antibody_unit_overlap",
            scoped,
            "_context_antibody_unit",
        ),
        _overlap_report(
            "within_antigen_component_overlap",
            scoped,
            "_context_component_id",
        ),
        _coverage_report(
            "valid_antigen_context_seen_in_train",
            scoped["train"],
            scoped["valid"],
            "_antigen_context_id",
        ),
        _coverage_report(
            "test_antigen_context_seen_in_train",
            scoped["train"],
            scoped["test"],
            "_antigen_context_id",
        ),
    ]

    for column in ("measurement_family_id", "interaction_key"):
        if column not in scoped["train"].columns:
            continue
        scoped_with_column = {
            split_name: split_records.assign(
                **{f"_context_{column}": _scoped_optional_values(split_records, column)}
            )
            for split_name, split_records in scoped.items()
        }
        rows.append(_overlap_report_ignoring_missing(
            f"within_antigen_{column}_overlap",
            scoped_with_column,
            f"_context_{column}",
        ))

    return pd.DataFrame(rows, columns=LEAKAGE_COLUMNS)


def _scoped_values(records: pd.DataFrame, column: str) -> pd.Series:
    return records["_antigen_context_id"].astype(str) + "\x1f" + records[column].astype(str)


def _scoped_optional_values(records: pd.DataFrame, column: str) -> pd.Series:
    values = records[column].astype("string").str.strip()
    scoped = records["_antigen_context_id"].astype("string") + "\x1f" + values
    return scoped.mask(values.isna() | values.eq(""))


def _overlap_report_ignoring_missing(
    check_name: str,
    splits: dict[str, pd.DataFrame],
    column: str,
) -> dict[str, object]:
    violations: list[str] = []
    for left_name, right_name in combinations(splits, 2):
        left = set(splits[left_name][column].dropna().astype(str))
        right = set(splits[right_name][column].dropna().astype(str))
        overlap = sorted(left & right)
        if overlap:
            violations.append(f"{left_name}/{right_name}: {overlap[:10]}")

    return {
        "check_name": check_name,
        "status": "PASS" if not violations else "FAIL",
        "n_violations": len(violations),
        "details": "; ".join(violations),
    }


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
