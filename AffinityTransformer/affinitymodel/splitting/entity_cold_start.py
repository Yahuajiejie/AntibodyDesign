"""Entity cold-start splits and K-folds for antibody and antigen protocols.

Both protocols share one engine parameterized by ``protocol``; see
docs/future/entity_cold_start_protocols.md sections 5.3 and 5.4.
"""
from __future__ import annotations

import hashlib

import pandas as pd

from ..annotations import join_entity_annotations, join_representation_annotations
from .common import (
    COLD_START_IDENTITY_COLUMNS,
    _assign_component_splits,
    _assign_weighted_units_to_folds,
    _combine_excluded_records,
    _drop_split_helpers,
    _partition_weighted_units,
    _trainable_records,
    _validate_fraction_and_eval_size,
)
from .audits import (
    ELIGIBILITY_COLUMNS,
    _build_entity_protocol_leakage_report,
    _build_summary,
    _build_unit_assignments,
    build_group_eligibility,
)
from .results import EntityColdStartFold, EntityColdStartSplitResult


_PROTOCOL_IDENTITY_COLUMNS = {
    "antibody_cold_start": (
        "measurement_family_id",
        "antibody_sequence_key",
        "antibody_cluster_id",
        "antigen_sequence_key",
        "interaction_key",
    ),
    "antigen_cold_start": (
        "measurement_family_id",
        "antibody_sequence_key",
        "antibody_cluster_id",
        "antigen_sequence_key",
        "antigen_cluster_id",
        "interaction_key",
    ),
}


ENTITY_COLD_START_STRATEGIES = {
    "antibody_cold_start_split",
    "antigen_cold_start_split",
}


def build_antibody_cold_start_split(
    records: pd.DataFrame,
    entity_annotations: pd.DataFrame | None = None,
    *,
    valid_fraction: float,
    test_fraction: float,
    seed: int,
    min_eval_records: int = 2,
    require_train_group: bool = True,
    strict_known_counterpart: bool = False,
) -> EntityColdStartSplitResult:
    """Split globally by antibody components for entity holdout evaluation.

    By default, this answers a global antibody-entity holdout question: can the
    model handle antibody clusters that never appeared in training, regardless
    of whether the paired antigen also appeared in training? When
    ``strict_known_counterpart=True``, it switches to the narrower controlled
    question: unseen antibodies on train-seen antigens (and, when
    ``require_train_group=True``, train-seen groups).

    Args:
        records: Base processed records. Entity identity columns are NOT
            required here when ``entity_annotations`` is supplied.
        entity_annotations: Optional separate annotation table keyed by
            ``record_id`` (see ``annotations.load_entity_annotations``). When
            provided it is validated and joined transiently; when ``None`` the
            identity columns are read from ``records`` directly (legacy path).
        valid_fraction: Fraction of records (by component weight) for valid.
        test_fraction: Fraction of records (by component weight) for test.
        seed: Deterministic assignment seed.
        min_eval_records: Minimum protocol-eligible records per evaluation group.
        require_train_group: In strict mode, require valid/test records to
            belong to a ``group_id`` already present in train.
        strict_known_counterpart: When True, apply the narrower known-antigen
            eligibility filter. Defaults to False.
    """
    prepared = _resolve_cold_start_inputs(
        records, entity_annotations, protocol="antibody_cold_start"
    )
    return _build_entity_cold_start_split(
        prepared,
        protocol="antibody_cold_start",
        valid_fraction=valid_fraction,
        test_fraction=test_fraction,
        seed=seed,
        min_eval_records=min_eval_records,
        require_train_group=require_train_group,
        strict_known_counterpart=strict_known_counterpart,
    )


def _resolve_cold_start_inputs(
    records: pd.DataFrame,
    entity_annotations: pd.DataFrame | None,
    *,
    protocol: str,
    representation_annotations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return records augmented with entity columns for split construction.

    With ``entity_annotations=None`` the legacy embedded-column behaviour is
    preserved untouched. Otherwise the annotation is validated and joined onto
    ``records`` by ``record_id`` (entity columns never persist into base split
    files). When ``representation_annotations`` is supplied (antigen only), the
    per-antigen ``effective_antigen_input_hash`` is materialized so the leakage
    audit can check effective-input overlap.
    """
    working = (
        records if entity_annotations is None
        else join_entity_annotations(records, entity_annotations)
    )
    if representation_annotations is not None:
        if protocol != "antigen_cold_start":
            raise ValueError(
                "representation_annotations are only supported for the "
                "antigen_cold_start protocol"
            )
        working = join_representation_annotations(
            working,
            representation_annotations,
            sequence_type="antigen",
            sequence_key_column="antigen_sequence_key",
            out_column="effective_antigen_input_hash",
        )
    return working


def build_antigen_cold_start_split(
    records: pd.DataFrame,
    entity_annotations: pd.DataFrame | None = None,
    *,
    valid_fraction: float,
    test_fraction: float,
    seed: int,
    min_eval_records: int = 2,
    representation_annotations: pd.DataFrame | None = None,
    strict_known_counterpart: bool = False,
) -> EntityColdStartSplitResult:
    """Split by antigen components for entity holdout evaluation.

    By default, this answers a global antigen-entity holdout question: can the
    model handle antigen clusters that never appeared in training, regardless
    of whether the paired antibody also appeared in training? When
    ``strict_known_counterpart=True``, it switches to the narrower controlled
    question: unseen antigens with train-seen antibody clusters.

    Args:
        records: Base processed records. Entity identity columns are NOT
            required here when ``entity_annotations`` is supplied.
        entity_annotations: Optional separate annotation table keyed by
            ``record_id``. When ``None`` identity columns are read from
            ``records`` directly (legacy embedded path).
        valid_fraction: Fraction of records (by component weight) for valid.
        test_fraction: Fraction of records (by component weight) for test.
        seed: Deterministic assignment seed.
        min_eval_records: Minimum protocol-eligible records per evaluation group.
        representation_annotations: Optional table (see
            ``annotations.load_representation_annotations``). When provided, the
            antigen effective-input hash is materialized and checked for
            cross-split overlap; when absent, effective-input is not audited.
        strict_known_counterpart: When True, apply the narrower train-seen
            antibody eligibility filter. Defaults to False.
    """
    prepared = _resolve_cold_start_inputs(
        records,
        entity_annotations,
        protocol="antigen_cold_start",
        representation_annotations=representation_annotations,
    )
    return _build_entity_cold_start_split(
        prepared,
        protocol="antigen_cold_start",
        valid_fraction=valid_fraction,
        test_fraction=test_fraction,
        seed=seed,
        min_eval_records=min_eval_records,
        require_train_group=False,
        strict_known_counterpart=strict_known_counterpart,
    )


def build_antibody_cold_start_kfolds(
    records: pd.DataFrame,
    n_splits: int,
    seed: int,
    *,
    min_eval_records: int = 2,
    require_train_group: bool = True,
    strict_known_counterpart: bool = False,
) -> list[EntityColdStartFold]:
    """Build antibody-component-isolated development folds."""
    return _build_entity_cold_start_kfolds(
        records,
        protocol="antibody_cold_start",
        n_splits=n_splits,
        seed=seed,
        min_eval_records=min_eval_records,
        require_train_group=require_train_group,
        strict_known_counterpart=strict_known_counterpart,
    )


def build_antigen_cold_start_kfolds(
    records: pd.DataFrame,
    n_splits: int,
    seed: int,
    *,
    min_eval_records: int = 2,
    strict_known_counterpart: bool = False,
) -> list[EntityColdStartFold]:
    """Build antigen-component-isolated development folds."""
    return _build_entity_cold_start_kfolds(
        records,
        protocol="antigen_cold_start",
        n_splits=n_splits,
        seed=seed,
        min_eval_records=min_eval_records,
        require_train_group=False,
        strict_known_counterpart=strict_known_counterpart,
    )


def _build_entity_cold_start_split(
    records: pd.DataFrame,
    *,
    protocol: str,
    valid_fraction: float,
    test_fraction: float,
    seed: int,
    min_eval_records: int,
    require_train_group: bool,
    strict_known_counterpart: bool,
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

    valid, valid_report, valid_excluded = _select_entity_holdout_records(
        raw_train,
        raw_valid,
        protocol=protocol,
        split_name="valid",
        min_eval_records=min_eval_records,
        require_train_group=require_train_group,
        strict_known_counterpart=strict_known_counterpart,
    )
    test, test_report, test_excluded = _select_entity_holdout_records(
        raw_train,
        raw_test,
        protocol=protocol,
        split_name="test",
        min_eval_records=min_eval_records,
        require_train_group=require_train_group,
        strict_known_counterpart=strict_known_counterpart,
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
        strict_known_counterpart=strict_known_counterpart,
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
    strict_known_counterpart: bool,
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
        valid, eligibility, fold_excluded = _select_entity_holdout_records(
            raw_train,
            raw_valid,
            protocol=protocol,
            split_name="valid",
            min_eval_records=min_eval_records,
            require_train_group=require_train_group,
            strict_known_counterpart=strict_known_counterpart,
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
            strict_known_counterpart=strict_known_counterpart,
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
    identity_columns = _PROTOCOL_IDENTITY_COLUMNS[protocol]
    required = (
        "record_id", "group_id", "dataset_id", "keep_for_training", "rank_label",
        *identity_columns,
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
    null_identity = working[list(identity_columns)].isna().any(axis=1)
    empty_identity = working[list(identity_columns)].apply(
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
    # antigen_sequence_key -> antigen_cluster_id is only validated when the
    # column is available. Antigen cold-start always carries it (unchanged);
    # antibody cold-start does not require antigen_cluster_id (section 5.3).
    if "antigen_cluster_id" in working.columns:
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
        # Indivisible unit = antigen_cluster_id + measurement_family_id
        # (entity_cold_start_protocols.md section 5.4). antigen_sequence_key is
        # redundant (it maps to exactly one antigen_cluster_id) and
        # effective_antigen_input_hash is a CONDITIONAL audit, not a component
        # link -- colliding effective inputs across distinct clusters are
        # flagged by the leakage report, not silently merged here.
        link_columns = (
            "antigen_cluster_id", "measurement_family_id",
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


def _select_entity_holdout_records(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    *,
    protocol: str,
    split_name: str,
    min_eval_records: int,
    require_train_group: bool,
    strict_known_counterpart: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if strict_known_counterpart:
        return _select_protocol_eligible_records(
            train,
            holdout,
            protocol=protocol,
            split_name=split_name,
            min_eval_records=min_eval_records,
            require_train_group=require_train_group,
        )
    _protocol_entity_column(protocol)  # validates the protocol name
    return build_group_eligibility(
        holdout,
        split_name=split_name,
        min_eval_records=min_eval_records,
        input_column="antibody_sequence_key",
        label_column="rank_label",
    )


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
