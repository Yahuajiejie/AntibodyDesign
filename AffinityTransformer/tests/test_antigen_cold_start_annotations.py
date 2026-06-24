"""Tests for the antigen cold-start protocol with separate entity annotations.

By default this protocol is a global antigen-entity holdout: held-out antigen
clusters are unseen, while the paired antibody may or may not have appeared in
train. ``strict_known_counterpart=True`` narrows the evaluation to train-seen
antibodies. Entity identity comes from a separate annotation table; the base
records schema is never extended. Representation annotations (effective-input
hashes) are optional.
"""

from __future__ import annotations

import pandas as pd
import pytest

from affinity_transformer.annotations import (
    join_entity_annotations,
    validate_representation_annotations,
)
from affinity_transformer.splits import (
    build_antigen_cold_start_manifest,
    build_antigen_cold_start_split,
    frame_hash,
    write_antigen_cold_start_split,
)

_ENTITY_COLUMNS = (
    "measurement_family_id",
    "antibody_sequence_key",
    "antibody_cluster_id",
    "antigen_sequence_key",
    "antigen_cluster_id",
    "interaction_key",
)


def _base_and_annotations(n_antigens: int = 9) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Base records + separate annotations.

    Each antigen (its own cluster) is measured by three shared antibodies
    (seen across every antigen) plus one antigen-unique antibody. When an
    antigen is held out, the shared antibodies remain train-seen while the
    unique antibody is a Dual cold-start record.
    """
    base_rows, ann_rows = [], []
    for i in range(n_antigens):
        ag = f"ag-{i}"
        for label, ab in enumerate(("shared-a", "shared-b", "shared-c")):
            rid = f"{ag}/{ab}"
            base_rows.append({
                "record_id": rid, "group_id": f"group/{ag}",
                "dataset_id": "study/table", "keep_for_training": True,
                "rank_label": float(label), "label_kind": "experimental",
            })
            ann_rows.append({
                "record_id": rid, "measurement_family_id": f"mf/{rid}",
                "antibody_sequence_key": f"ab-seq/{ab}",
                "antibody_cluster_id": f"ab-cluster/{ab}",
                "antigen_sequence_key": f"ag-seq/{ag}",
                "antigen_cluster_id": f"ag-cluster/{ag}",
                "interaction_key": f"interaction/{ag}/{ab}",
            })
        rid = f"{ag}/unique"
        base_rows.append({
            "record_id": rid, "group_id": f"group/{ag}",
            "dataset_id": "study/table", "keep_for_training": True,
            "rank_label": 3.0, "label_kind": "experimental",
        })
        ann_rows.append({
            "record_id": rid, "measurement_family_id": f"mf/{rid}",
            "antibody_sequence_key": f"ab-seq/unique-{ag}",
            "antibody_cluster_id": f"ab-cluster/unique-{ag}",
            "antigen_sequence_key": f"ag-seq/{ag}",
            "antigen_cluster_id": f"ag-cluster/{ag}",
            "interaction_key": f"interaction/{ag}/unique",
        })
    return pd.DataFrame(base_rows), pd.DataFrame(ann_rows)


def _representation(annotations: pd.DataFrame, *, collide: bool = False) -> pd.DataFrame:
    keys = sorted(set(annotations["antigen_sequence_key"]))
    rows = []
    for key in keys:
        eff = "eff/SAME" if collide else f"eff/{key}"
        rows.append({
            "sequence_type": "antigen", "sequence_key": key,
            "effective_input_hash": eff,
        })
    return pd.DataFrame(rows)


def _clusters(frame: pd.DataFrame, column: str) -> set[str]:
    return set(frame[column].astype(str))


def _split(seed: int = 5, **kwargs):
    records, annotations = _base_and_annotations()
    return build_antigen_cold_start_split(
        records, annotations, valid_fraction=0.22, test_fraction=0.22,
        seed=seed, min_eval_records=2, **kwargs,
    )


# 1. Separate entity annotations are accepted.
def test_separate_annotations_accepted():
    result = _split()
    assert not result.train.empty and not result.valid.empty and not result.test.empty


# 2. Embedded-column compatibility path remains unchanged (no annotations arg).
def test_embedded_column_path_supported():
    records, annotations = _base_and_annotations()
    embedded = join_entity_annotations(records, annotations)
    result = build_antigen_cold_start_split(
        embedded, valid_fraction=0.22, test_fraction=0.22, seed=5, min_eval_records=2
    )
    assert _clusters(result.train, "antigen_cluster_id").isdisjoint(
        _clusters(result.valid, "antigen_cluster_id")
    )


# 3. Missing or duplicate entity annotation rows are rejected.
def test_missing_and_duplicate_annotations_rejected():
    records, annotations = _base_and_annotations()
    with pytest.raises(ValueError, match="missing rows"):
        build_antigen_cold_start_split(
            records, annotations.iloc[1:].reset_index(drop=True),
            valid_fraction=0.22, test_fraction=0.22, seed=5, min_eval_records=2,
        )
    dup = pd.concat([annotations, annotations.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate record_id"):
        build_antigen_cold_start_split(
            records, dup, valid_fraction=0.22, test_fraction=0.22, seed=5,
            min_eval_records=2,
        )


# 4/5/6/7. No overlap across splits for the isolated entity keys.
def test_entity_keys_disjoint_across_splits():
    result = _split()
    for column in (
        "antigen_sequence_key", "antigen_cluster_id",
        "measurement_family_id", "interaction_key",
    ):
        a = _clusters(result.train, column)
        b = _clusters(result.valid, column)
        c = _clusters(result.test, column)
        assert a.isdisjoint(b) and a.isdisjoint(c) and b.isdisjoint(c), column
    # record_id disjoint too.
    assert _clusters(result.train, "record_id").isdisjoint(
        _clusters(result.valid, "record_id")
    )


# 8. Default valid/test keeps all held-out antigen records, including
# antigen-unique antibody clusters.
def test_default_keeps_unseen_antibody_counterparts():
    result = _split()
    assert result.excluded_records.empty
    assert any(
        value.startswith("ab-cluster/unique-")
        for value in _clusters(result.valid, "antibody_cluster_id")
    )
    checks = set(result.leakage_report["check_name"])
    assert "valid_antibody_seen_in_train" not in checks


# 9. Strict mode keeps only antibody clusters seen in train and records
# dual-ish rows in excluded_records.
def test_strict_eval_antibodies_subset_of_train():
    result = _split(strict_known_counterpart=True)
    train_ab = _clusters(result.train, "antibody_cluster_id")
    assert _clusters(result.valid, "antibody_cluster_id") <= train_ab
    assert _clusters(result.test, "antibody_cluster_id") <= train_ab
    assert not result.excluded_records.empty
    assert set(result.excluded_records["protocol_exclusion_reason"]) == {
        "antibody_cluster_not_seen_in_train"
    }


# 10. Optional representation annotations detect effective-input overlap.
def test_representation_detects_effective_input_collision():
    records, annotations = _base_and_annotations()
    colliding = _representation(annotations, collide=True)
    with pytest.raises(ValueError, match="effective_antigen_input_overlap"):
        build_antigen_cold_start_split(
            records, annotations, valid_fraction=0.22, test_fraction=0.22,
            seed=5, min_eval_records=2, representation_annotations=colliding,
        )


def test_representation_without_collision_passes_and_audits():
    records, annotations = _base_and_annotations()
    rep = _representation(annotations, collide=False)
    result = build_antigen_cold_start_split(
        records, annotations, valid_fraction=0.22, test_fraction=0.22,
        seed=5, min_eval_records=2, representation_annotations=rep,
    )
    checks = set(result.leakage_report["check_name"])
    assert "effective_antigen_input_overlap" in checks


# 11. The split works without representation annotations (no effective audit).
def test_works_without_representation():
    result = _split()
    checks = set(result.leakage_report["check_name"])
    assert "effective_antigen_input_overlap" not in checks
    # antigen isolation still enforced
    assert _clusters(result.train, "antigen_cluster_id").isdisjoint(
        _clusters(result.valid, "antigen_cluster_id")
    )


# 12. Same seed produces identical output.
def test_same_seed_deterministic():
    records, annotations = _base_and_annotations()
    kwargs = dict(valid_fraction=0.22, test_fraction=0.22, seed=5, min_eval_records=2)
    first = build_antigen_cold_start_split(records, annotations, **kwargs)
    second = build_antigen_cold_start_split(records, annotations, **kwargs)
    for name in ("train", "valid", "test"):
        assert getattr(first, name)["record_id"].tolist() == getattr(
            second, name
        )["record_id"].tolist(), name


# Representation validation rejects incomplete coverage.
def test_representation_validation_requires_full_coverage():
    records, annotations = _base_and_annotations()
    joined = join_entity_annotations(records, annotations)
    rep = _representation(annotations).iloc[1:].reset_index(drop=True)  # drop one key
    with pytest.raises(ValueError, match="missing antigen rows"):
        validate_representation_annotations(
            joined, rep, sequence_type="antigen",
            sequence_key_column="antigen_sequence_key",
        )


# Writer: base files keep base schema; manifest reflects audit state.
def test_writer_base_schema_and_manifest(tmp_path):
    import yaml

    records, annotations = _base_and_annotations()
    result = build_antigen_cold_start_split(
        records, annotations, valid_fraction=0.22, test_fraction=0.22,
        seed=5, min_eval_records=2,
    )
    manifest = build_antigen_cold_start_manifest(
        seed=5, valid_fraction=0.22, test_fraction=0.22, min_eval_records=2,
        input_records_hash=frame_hash(records),
        entity_annotations_hash=frame_hash(annotations),
        representation_annotations_hash=None,
        effective_input_audited=False,
    )
    write_antigen_cold_start_split(result, tmp_path, manifest=manifest)
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
    loaded = yaml.safe_load((tmp_path / "split_manifest.yaml").read_text())
    assert loaded["protocol"] == "antigen_cold_start"
    assert loaded["effective_input_audited"] is False
    assert loaded["representation_annotations_hash"] is None
