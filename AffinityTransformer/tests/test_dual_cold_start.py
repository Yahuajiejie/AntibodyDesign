"""Tests for the Dual cold-start protocol (unseen antigen AND unseen antibody).

Covers entity_cold_start_protocols.md section 5.5. Entity identity comes from a
separate annotation table; representation annotations are optional.
"""

from __future__ import annotations

import pandas as pd
import pytest

from affinity_transformer.splitting.common import derive_link_components
from affinity_transformer.splitting.dual_cold_start import DUAL_LINK_COLUMNS
from affinity_transformer.splits import (
    build_dual_cold_start_manifest,
    build_dual_cold_start_split,
    frame_hash,
    write_dual_cold_start_split,
)

_ENTITY_COLUMNS = (
    "measurement_family_id", "antibody_sequence_key", "antibody_cluster_id",
    "antigen_sequence_key", "antigen_cluster_id", "interaction_key",
)


def _blocks(n_blocks: int = 6, thin: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``n_blocks`` disjoint antibody-antigen blocks -> one component each.

    Each block has two antigens (agA, agB) sharing antibodies ab0/ab1 (so the
    block is one connected component). agA is always evaluable (3 labels). When
    ``thin``, agB has two records with the same label -> non-evaluable, so any
    block routed to valid/test contributes excluded records.
    """
    base, ann = [], []
    for b in range(n_blocks):
        specs = [
            (f"b{b}-agA", [("ab0", 0.0), ("ab1", 1.0), ("ab2", 2.0)]),
            (f"b{b}-agB", [("ab0", 5.0), ("ab1", 5.0)] if thin
                          else [("ab0", 3.0), ("ab1", 4.0)]),
        ]
        for ag, members in specs:
            for ab, label in members:
                rid = f"{ag}/{ab}"
                base.append({
                    "record_id": rid, "group_id": f"group/{ag}",
                    "dataset_id": "study/table", "keep_for_training": True,
                    "rank_label": label, "label_kind": "experimental",
                })
                ann.append({
                    "record_id": rid, "measurement_family_id": f"mf/{rid}",
                    "antibody_sequence_key": f"abs/{b}-{ab}",
                    "antibody_cluster_id": f"abc/{b}-{ab}",
                    "antigen_sequence_key": f"ags/{ag}",
                    "antigen_cluster_id": f"agc/{ag}",
                    "interaction_key": f"int/{ag}/{ab}",
                })
    return pd.DataFrame(base), pd.DataFrame(ann)


def _representation(annotations: pd.DataFrame, *, collide: bool = False) -> pd.DataFrame:
    keys = sorted(set(annotations["antigen_sequence_key"]))
    return pd.DataFrame([
        {"sequence_type": "antigen", "sequence_key": k, "representation_id": "rep-v1",
         "effective_input_hash": "eff/SAME" if collide else f"eff/{k}"}
        for k in keys
    ])


def _split(seed: int = 3, thin: bool = True, **kwargs):
    records, ann = _blocks(thin=thin)
    return records, ann, build_dual_cold_start_split(
        records, ann, valid_fraction=0.2, test_fraction=0.2, seed=seed,
        min_eval_records=2, **kwargs,
    )


def _s(frame, col):
    return set(frame[col].astype(str))


# 1. All four entity relationships participate in transitive component construction.
def test_all_four_relationships_link_components():
    rows = [
        {"record_id": "r0", "antibody_cluster_id": "A", "antigen_cluster_id": "g0",
         "measurement_family_id": "m0", "interaction_key": "i0"},
        {"record_id": "r1", "antibody_cluster_id": "A", "antigen_cluster_id": "g1",
         "measurement_family_id": "m1", "interaction_key": "i1"},  # links via antibody
        {"record_id": "r2", "antibody_cluster_id": "B", "antigen_cluster_id": "g1",
         "measurement_family_id": "m2", "interaction_key": "i2"},  # links via antigen
        {"record_id": "r3", "antibody_cluster_id": "C", "antigen_cluster_id": "g3",
         "measurement_family_id": "m2", "interaction_key": "i3"},  # links via mf
        {"record_id": "r4", "antibody_cluster_id": "D", "antigen_cluster_id": "g4",
         "measurement_family_id": "m4", "interaction_key": "i3"},  # links via interaction
        {"record_id": "x9", "antibody_cluster_id": "Z", "antigen_cluster_id": "gz",
         "measurement_family_id": "mz", "interaction_key": "iz"},  # isolated
    ]
    frame = pd.DataFrame(rows)
    comp = derive_link_components(frame, DUAL_LINK_COLUMNS)
    chain = set(comp.iloc[:5])
    assert len(chain) == 1, "r0..r4 must collapse into one transitive component"
    assert comp.iloc[5] not in chain, "isolated record stays separate"


# 2. A component is never split across train/valid/test.
def test_component_never_split():
    records, ann, result = _split()
    tagged = []
    for name, frame in (("train", result.train), ("valid", result.valid), ("test", result.test)):
        f = frame.copy(); f["_split"] = name; tagged.append(f)
    allrows = pd.concat(tagged, ignore_index=True)
    allrows["_component_id"] = derive_link_components(allrows, DUAL_LINK_COLUMNS)
    for _, comp in allrows.groupby("_component_id"):
        assert comp["_split"].nunique() == 1


# 3/4/5/6. Exact and clustered identities + mf/interaction + record_id disjoint.
def test_entity_keys_disjoint_across_splits():
    _, _, result = _split()
    for col in (
        "record_id", "measurement_family_id",
        "antibody_sequence_key", "antibody_cluster_id",
        "antigen_sequence_key", "antigen_cluster_id", "interaction_key",
    ):
        a, b, c = _s(result.train, col), _s(result.valid, col), _s(result.test, col)
        assert a.isdisjoint(b) and a.isdisjoint(c) and b.isdisjoint(c), col


# 7. Evaluation groups satisfy minimum record/input/label requirements.
def test_eval_groups_meet_minimums():
    _, _, result = _split()
    for frame in (result.valid, result.test):
        for _, group in frame.groupby("group_id"):
            assert len(group) >= 2
            assert group["antibody_sequence_key"].astype(str).nunique() >= 2
            assert pd.to_numeric(group["rank_label"]).nunique() >= 2


# 8. Ineligible evaluation records appear in excluded_records with reasons.
def test_ineligible_records_excluded_with_reason():
    _, _, result = _split(thin=True)
    assert not result.excluded_records.empty
    reasons = set(result.excluded_records["protocol_exclusion_reason"])
    assert any("group_not_evaluable" in r for r in reasons)


# 9. Largest-component statistics are correct.
def test_component_summary_largest_is_correct():
    records, ann, result = _split()
    cs = result.component_summary
    # sorted largest-first by record_fraction
    assert list(cs["record_fraction"]) == sorted(cs["record_fraction"], reverse=True)
    # each block is one component: 6 components, each 5 records (3 + 2)
    assert len(cs) == 6
    largest = cs.iloc[0]
    total = len(records)  # all trainable
    assert largest["n_records"] == 5
    assert abs(largest["record_fraction"] - 5 / total) < 1e-9
    assert largest["n_antigen_clusters"] == 2  # agA + agB
    assert largest["n_antibody_clusters"] == 3  # ab0, ab1, ab2


# 10. A mega-component that makes valid/test impossible fails explicitly.
def test_mega_component_is_infeasible():
    records, ann = _blocks(n_blocks=6)
    # Collapse everything into ONE component via a shared measurement family.
    ann = ann.copy()
    ann["measurement_family_id"] = "mf/ALL"
    with pytest.raises(ValueError, match="protocol-infeasible"):
        build_dual_cold_start_split(
            records, ann, valid_fraction=0.2, test_fraction=0.2, seed=3,
            min_eval_records=2,
        )


# 11. Representation annotations detect effective-input collisions (fail loud).
def test_representation_collision_fails_loud():
    records, ann = _blocks(thin=False)
    rep = _representation(ann, collide=True)
    with pytest.raises(ValueError, match="effective-input collision"):
        build_dual_cold_start_split(
            records, ann, valid_fraction=0.2, test_fraction=0.2, seed=3,
            min_eval_records=2, representation_annotations=rep,
        )


# 12. The split works without representation annotations.
def test_works_without_representation():
    _, _, result = _split(thin=False)
    checks = set(result.leakage_report["check_name"])
    assert "effective_antigen_input_overlap" not in checks
    assert (result.leakage_report["status"] == "PASS").all()


def test_representation_without_collision_audits():
    records, ann = _blocks(thin=False)
    rep = _representation(ann, collide=False)
    result = build_dual_cold_start_split(
        records, ann, valid_fraction=0.2, test_fraction=0.2, seed=3,
        min_eval_records=2, representation_annotations=rep,
    )
    assert "effective_antigen_input_overlap" in set(result.leakage_report["check_name"])
    assert (result.leakage_report["status"] == "PASS").all()


# 13. The same seed produces identical membership and ordering.
def test_same_seed_deterministic():
    records, ann = _blocks(thin=True)
    kwargs = dict(valid_fraction=0.2, test_fraction=0.2, seed=3, min_eval_records=2)
    a = build_dual_cold_start_split(records, ann, **kwargs)
    b = build_dual_cold_start_split(records, ann, **kwargs)
    for name in ("train", "valid", "test"):
        assert getattr(a, name)["record_id"].tolist() == getattr(b, name)["record_id"].tolist()


# Writer: base columns only; manifest carries component stats.
def test_writer_base_schema_and_manifest(tmp_path):
    import yaml

    records, ann, result = _split(thin=False)
    manifest = build_dual_cold_start_manifest(
        seed=3, valid_fraction=0.2, test_fraction=0.2, min_eval_records=2,
        input_records_hash=frame_hash(records),
        entity_annotations_hash=frame_hash(ann),
        representation_annotations_hash=None,
        effective_input_audited=False,
        component_summary=result.component_summary,
    )
    write_dual_cold_start_split(result, tmp_path, manifest=manifest)
    for name in ("train", "valid", "test"):
        frame = pd.read_parquet(tmp_path / f"{name}.parquet")
        for column in _ENTITY_COLUMNS:
            assert column not in frame.columns
    for artifact in (
        "train.parquet", "valid.parquet", "test.parquet", "split_manifest.yaml",
        "component_assignments.parquet", "component_summary.csv",
        "eligibility_report.csv", "excluded_records.parquet",
        "leakage_report.csv", "summary.csv",
    ):
        assert (tmp_path / artifact).exists(), artifact
    loaded = yaml.safe_load((tmp_path / "split_manifest.yaml").read_text())
    assert loaded["protocol"] == "dual_cold_start"
    assert loaded["n_components"] == 6
    assert loaded["effective_input_audited"] is False
    assert 0.0 < loaded["largest_component_record_fraction"] <= 1.0
