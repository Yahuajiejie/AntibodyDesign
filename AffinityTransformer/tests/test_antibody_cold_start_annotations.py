"""Tests for the antibody cold-start protocol with separate entity annotations.

Covers entity_cold_start_protocols.md section 5.3: known antigen, unseen
antibody. Entity identity comes from a separate annotation table; the base
records schema is never extended.
"""

from __future__ import annotations

import pandas as pd
import pytest

from affinity_transformer.annotations import (
    join_entity_annotations,
    load_entity_annotations,
    validate_entity_annotations,
)
from affinity_transformer.splits import (
    _select_protocol_eligible_records,
    build_antibody_cold_start_manifest,
    build_antibody_cold_start_split,
    frame_hash,
    write_antibody_cold_start_split,
)

_ENTITY_COLUMNS = (
    "measurement_family_id",
    "antibody_sequence_key",
    "antibody_cluster_id",
    "antigen_sequence_key",
    "interaction_key",
)


def _base_and_annotations(
    n_antibodies: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean records split into a base table and a separate annotation table.

    Each antibody (its own cluster) is measured under two known antigens A and
    B, mirroring the embedded-column fixture but with identity fields kept out
    of the base schema.
    """
    base_rows = []
    annotation_rows = []
    for antibody_index in range(n_antibodies):
        for antigen in ("A", "B"):
            record_id = f"{antigen}/ab-{antibody_index}"
            base_rows.append({
                "record_id": record_id,
                "group_id": f"group/{antigen}",
                "dataset_id": "study/table",
                "keep_for_training": True,
                "rank_label": float(antibody_index),
                "label_kind": "experimental",
            })
            annotation_rows.append({
                "record_id": record_id,
                "measurement_family_id": f"mf/{record_id}",
                "antibody_sequence_key": f"ab-seq/ab-{antibody_index}",
                "antibody_cluster_id": f"ab-cluster/ab-{antibody_index}",
                "antigen_sequence_key": f"ag-seq/{antigen}",
                "antigen_cluster_id": f"ag-cluster/{antigen}",
                "interaction_key": f"interaction/{antigen}/ab-{antibody_index}",
            })
    return pd.DataFrame(base_rows), pd.DataFrame(annotation_rows)


def _clusters(frame: pd.DataFrame, column: str) -> set[str]:
    return set(frame[column].astype(str))


def _split_result(seed: int = 7, **kwargs):
    records, annotations = _base_and_annotations()
    return records, annotations, build_antibody_cold_start_split(
        records,
        annotations,
        valid_fraction=0.25,
        test_fraction=0.25,
        seed=seed,
        min_eval_records=2,
        **kwargs,
    )


# 1. Entity annotations are loaded and validated separately from records.
def test_annotations_load_validate_and_join_separately(tmp_path):
    records, annotations = _base_and_annotations()
    path = tmp_path / "entity_annotations.parquet"
    annotations.to_parquet(path, index=False)

    loaded = load_entity_annotations(path)
    validate_entity_annotations(records, loaded)
    joined = join_entity_annotations(records, loaded)

    # Base records never carry entity identity columns.
    for column in _ENTITY_COLUMNS:
        assert column not in records.columns
        assert column in joined.columns
    assert len(joined) == len(records)
    assert list(joined["record_id"]) == list(records["record_id"])


# 2. Missing annotation rows are rejected.
def test_missing_annotation_rows_rejected():
    records, annotations = _base_and_annotations()
    annotations = annotations.iloc[1:].reset_index(drop=True)
    with pytest.raises(ValueError, match="missing rows"):
        validate_entity_annotations(records, annotations)


# 3. Duplicate record_id rows in annotations are rejected.
def test_duplicate_annotation_record_id_rejected():
    records, annotations = _base_and_annotations()
    annotations = pd.concat(
        [annotations, annotations.iloc[[0]]], ignore_index=True
    )
    with pytest.raises(ValueError, match="duplicate record_id"):
        validate_entity_annotations(records, annotations)


# Many-to-one consistency: one cluster may hold many sequence keys (allowed),
# but a sequence key mapping to two clusters is rejected.
def test_many_sequence_keys_per_cluster_allowed_but_not_reverse():
    records, annotations = _base_and_annotations()
    # Two distinct sequence keys -> one shared cluster: allowed.
    annotations.loc[
        annotations["antibody_cluster_id"] == "ab-cluster/ab-1", "antibody_cluster_id"
    ] = "ab-cluster/ab-0"
    validate_entity_annotations(records, annotations)  # must not raise

    # One sequence key -> two clusters: rejected.
    bad = annotations.copy()
    mask = bad["antibody_sequence_key"] == "ab-seq/ab-2"
    first_index = bad.loc[mask].index[0]
    bad.loc[first_index, "antibody_cluster_id"] = "ab-cluster/divergent"
    with pytest.raises(ValueError, match="exactly one antibody_cluster_id"):
        validate_entity_annotations(records, bad)


# 4/5/6. Cluster / measurement family / interaction keys disjoint across splits.
def test_identity_keys_disjoint_across_splits():
    _, _, result = _split_result()
    for column in ("antibody_cluster_id", "measurement_family_id", "interaction_key"):
        train = _clusters(result.train, column)
        valid = _clusters(result.valid, column)
        test = _clusters(result.test, column)
        assert train.isdisjoint(valid), column
        assert train.isdisjoint(test), column
        assert valid.isdisjoint(test), column
    # Antibody sequence keys are isolated too.
    train_seq = _clusters(result.train, "antibody_sequence_key")
    assert train_seq.isdisjoint(_clusters(result.valid, "antibody_sequence_key"))


# 7. Valid/test only keep records whose antigen is already present in train.
def test_eval_antigens_subset_of_train():
    _, _, result = _split_result()
    train_antigens = _clusters(result.train, "antigen_sequence_key")
    assert _clusters(result.valid, "antigen_sequence_key") <= train_antigens
    assert _clusters(result.test, "antigen_sequence_key") <= train_antigens


# 8. When require_train_group=True, valid/test groups are present in train.
def test_eval_groups_subset_of_train_when_required():
    _, _, result = _split_result(require_train_group=True)
    train_groups = _clusters(result.train, "group_id")
    assert _clusters(result.valid, "group_id") <= train_groups
    assert _clusters(result.test, "group_id") <= train_groups


# 8 (semantics). require_train_group toggles the group eligibility rule.
def test_require_train_group_controls_new_group_exclusion():
    train = pd.DataFrame([
        {"record_id": "t1", "group_id": "g1", "antigen_sequence_key": "agA",
         "antibody_sequence_key": "abT1", "rank_label": 0.0},
        {"record_id": "t2", "group_id": "g1", "antigen_sequence_key": "agA",
         "antibody_sequence_key": "abT2", "rank_label": 1.0},
    ])
    holdout = pd.DataFrame([
        # Seen group, seen antigen -> eligible.
        {"record_id": "h1", "group_id": "g1", "antigen_sequence_key": "agA",
         "antibody_sequence_key": "abH1", "rank_label": 0.0},
        {"record_id": "h2", "group_id": "g1", "antigen_sequence_key": "agA",
         "antibody_sequence_key": "abH2", "rank_label": 1.0},
        # New group, seen antigen -> depends on require_train_group.
        {"record_id": "h3", "group_id": "gNEW", "antigen_sequence_key": "agA",
         "antibody_sequence_key": "abH3", "rank_label": 0.0},
        {"record_id": "h4", "group_id": "gNEW", "antigen_sequence_key": "agA",
         "antibody_sequence_key": "abH4", "rank_label": 1.0},
    ])

    eligible_strict, _, excluded_strict = _select_protocol_eligible_records(
        train, holdout, protocol="antibody_cold_start", split_name="valid",
        min_eval_records=2, require_train_group=True,
    )
    assert set(eligible_strict["record_id"]) == {"h1", "h2"}
    new_group_reasons = set(
        excluded_strict.loc[
            excluded_strict["record_id"].isin({"h3", "h4"}),
            "protocol_exclusion_reason",
        ]
    )
    assert new_group_reasons == {"group_not_seen_in_train"}

    eligible_relaxed, _, _ = _select_protocol_eligible_records(
        train, holdout, protocol="antibody_cold_start", split_name="valid",
        min_eval_records=2, require_train_group=False,
    )
    assert set(eligible_relaxed["record_id"]) == {"h1", "h2", "h3", "h4"}


# 9. Ineligible records are written to excluded_records with reasons.
def test_unseen_antigen_records_excluded_with_reason():
    train = pd.DataFrame([
        {"record_id": "t1", "group_id": "g1", "antigen_sequence_key": "agA",
         "antibody_sequence_key": "abT1", "rank_label": 0.0},
        {"record_id": "t2", "group_id": "g1", "antigen_sequence_key": "agA",
         "antibody_sequence_key": "abT2", "rank_label": 1.0},
    ])
    holdout = pd.DataFrame([
        {"record_id": "h1", "group_id": "g1", "antigen_sequence_key": "agA",
         "antibody_sequence_key": "abH1", "rank_label": 0.0},
        {"record_id": "h2", "group_id": "g1", "antigen_sequence_key": "agA",
         "antibody_sequence_key": "abH2", "rank_label": 1.0},
        # Antigen never seen in train -> excluded regardless of group.
        {"record_id": "h3", "group_id": "g1", "antigen_sequence_key": "agZ",
         "antibody_sequence_key": "abH3", "rank_label": 2.0},
    ])
    _, _, excluded = _select_protocol_eligible_records(
        train, holdout, protocol="antibody_cold_start", split_name="valid",
        min_eval_records=2, require_train_group=True,
    )
    assert "h3" in set(excluded["record_id"])
    assert excluded.loc[
        excluded["record_id"] == "h3", "protocol_exclusion_reason"
    ].iloc[0] == "antigen_sequence_not_seen_in_train"


# 10. Same seed produces the same split.
def test_same_seed_is_deterministic():
    records, annotations = _base_and_annotations()
    kwargs = dict(valid_fraction=0.25, test_fraction=0.25, seed=99, min_eval_records=2)
    first = build_antibody_cold_start_split(records, annotations, **kwargs)
    second = build_antibody_cold_start_split(records, annotations, **kwargs)
    for name in ("train", "valid", "test"):
        left = getattr(first, name)["record_id"].tolist()
        right = getattr(second, name)["record_id"].tolist()
        assert left == right, name


# Writer: base files keep base schema; manifest carries hashes and parameters.
def test_writer_keeps_base_schema_and_writes_manifest(tmp_path):
    import yaml

    records, annotations = _base_and_annotations()
    result = build_antibody_cold_start_split(
        records, annotations, valid_fraction=0.25, test_fraction=0.25, seed=7,
        min_eval_records=2,
    )
    manifest = build_antibody_cold_start_manifest(
        seed=7, valid_fraction=0.25, test_fraction=0.25, min_eval_records=2,
        require_train_group=True,
        input_records_hash=frame_hash(records),
        entity_annotations_hash=frame_hash(annotations),
    )
    write_antibody_cold_start_split(result, tmp_path, manifest=manifest)

    for name in ("train", "valid", "test"):
        frame = pd.read_parquet(tmp_path / f"{name}.parquet")
        for column in _ENTITY_COLUMNS:
            assert column not in frame.columns
    for artifact in (
        "component_assignments.parquet", "eligibility_report.csv",
        "excluded_records.parquet", "leakage_report.csv", "summary.csv",
        "split_manifest.yaml",
    ):
        assert (tmp_path / artifact).exists(), artifact

    loaded_manifest = yaml.safe_load((tmp_path / "split_manifest.yaml").read_text())
    assert loaded_manifest["protocol"] == "antibody_cold_start"
    assert loaded_manifest["seed"] == 7
    assert loaded_manifest["require_train_group"] is True
    assert loaded_manifest["input_records_hash"]
    assert loaded_manifest["entity_annotations_hash"]

    # Debug mode keeps entity columns for inspection.
    debug_dir = tmp_path / "debug"
    write_antibody_cold_start_split(result, debug_dir, manifest=manifest, debug=True)
    debug_train = pd.read_parquet(debug_dir / "train.parquet")
    assert "antibody_cluster_id" in debug_train.columns


# Backward compatibility: embedded-column path still works without annotations.
def test_embedded_identity_path_still_supported():
    records, annotations = _base_and_annotations()
    embedded = join_entity_annotations(records, annotations)
    result = build_antibody_cold_start_split(
        embedded, valid_fraction=0.25, test_fraction=0.25, seed=7, min_eval_records=2
    )
    assert _clusters(result.train, "antibody_cluster_id").isdisjoint(
        _clusters(result.valid, "antibody_cluster_id")
    )
