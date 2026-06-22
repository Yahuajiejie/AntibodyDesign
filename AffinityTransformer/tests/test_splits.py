"""Tests for affinity_transformer.splits."""

from __future__ import annotations

import pandas as pd
import pytest

from affinity_transformer.antigen_clustering import compute_antigen_clusters
from affinity_transformer.splits import build_group_kfolds, build_splits


def test_debug_record_split_has_no_record_id_leakage(toy_records):
    result = build_splits(
        toy_records,
        strategy="debug_record_split",
        valid_fraction=0.2,
        test_fraction=0.2,
        seed=0,
    )

    train_ids = set(result.train["record_id"])
    valid_ids = set(result.valid["record_id"])
    test_ids = set(result.test["record_id"])

    assert not (train_ids & valid_ids)
    assert not (train_ids & test_ids)
    assert not (valid_ids & test_ids)
    assert (result.leakage_report["status"] == "PASS").all()


def test_group_holdout_split_has_no_group_id_leakage(toy_records):
    result = build_splits(
        toy_records,
        strategy="group_holdout_split",
        valid_fraction=0.2,
        test_fraction=0.2,
        seed=0,
    )

    train_groups = set(result.train["group_id"])
    valid_groups = set(result.valid["group_id"])
    test_groups = set(result.test["group_id"])

    assert not (train_groups & valid_groups)
    assert not (train_groups & test_groups)
    assert not (valid_groups & test_groups)
    assert "group_id_overlap" in result.leakage_report["check_name"].tolist()
    assert (result.leakage_report["status"] == "PASS").all()


def test_group_holdout_split_keeps_oversized_group_in_train_when_possible():
    rows = []
    for index in range(100):
        rows.append(_split_row(f"big/{index}", "big_group", "study/big"))
    for group_id in ["small_a", "small_b", "small_c"]:
        for index in range(2):
            rows.append(_split_row(f"{group_id}/{index}", group_id, f"study/{group_id}"))
    records = pd.DataFrame(rows)

    result = build_splits(
        records,
        strategy="group_holdout_split",
        valid_fraction=0.1,
        test_fraction=0.1,
        seed=0,
    )

    assert "big_group" in set(result.train["group_id"])
    assert "big_group" not in set(result.valid["group_id"])
    assert "big_group" not in set(result.test["group_id"])
    assert len(result.train) > len(result.valid)
    assert len(result.train) > len(result.test)


def test_split_summary_reports_all_splits(toy_records):
    result = build_splits(
        toy_records,
        strategy="group_holdout_split",
        valid_fraction=0.2,
        test_fraction=0.2,
        seed=1,
    )

    assert result.summary["split"].tolist() == ["train", "valid", "test"]
    assert result.summary["n_records"].sum() == len(toy_records)
    assert set(result.summary["strategy"]) == {"group_holdout_split"}


def test_build_splits_rejects_unknown_strategy(toy_records):
    try:
        build_splits(toy_records, "bad_strategy", 0.2, 0.2, seed=0)
    except ValueError as exc:
        assert "strategy" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_group_kfolds_have_no_leakage_and_cover_each_record_once():
    records = pd.DataFrame(
        _split_row(f"group_{group}/{index}", f"group_{group}", f"study/{group}")
        for group in range(7)
        for index in range(group + 1)
    )

    folds = build_group_kfolds(records, n_splits=3, seed=17)

    assert len(folds) == 3
    validation_ids = []
    for fold in folds:
        train_groups = set(fold.train["group_id"])
        valid_groups = set(fold.valid["group_id"])
        assert train_groups.isdisjoint(valid_groups)
        validation_ids.extend(fold.valid["record_id"].tolist())
    assert sorted(validation_ids) == sorted(records["record_id"].tolist())


def test_group_kfolds_are_deterministic_for_seed():
    records = pd.DataFrame(
        _split_row(f"record_{group}", f"group_{group}", f"study/{group}")
        for group in range(8)
    )

    first = build_group_kfolds(records, n_splits=4, seed=9)
    second = build_group_kfolds(records.sample(frac=1.0), n_splits=4, seed=9)

    assert [set(fold.valid["group_id"]) for fold in first] == [
        set(fold.valid["group_id"]) for fold in second
    ]


def test_group_kfolds_reject_more_folds_than_groups():
    records = pd.DataFrame([
        _split_row("record_a", "group_a", "study/a"),
        _split_row("record_b", "group_b", "study/b"),
    ])

    try:
        build_group_kfolds(records, n_splits=3, seed=0)
    except ValueError as exc:
        assert "exceeds the number of groups" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def _split_row(record_id: str, group_id: str, dataset_id: str) -> dict[str, object]:
    return {
        "record_id": record_id,
        "group_id": group_id,
        "dataset_id": dataset_id,
        "keep_for_training": True,
        "rank_label": 1.0,
        "label_kind": "experimental",
        "antigen_source": "retrieved",
    }


def _antigen_cluster_row(record_id: str, antigen_key: str, antigen_sequence: str) -> dict[str, object]:
    return {
        **_split_row(record_id, group_id=f"{antigen_key}/kd", dataset_id="studyX/tableX"),
        "antigen_key": antigen_key,
    } | {"antigen_sequence": antigen_sequence}


def _mutate(seq: str, pos: int, new_char: str) -> str:
    return seq[:pos] + new_char + seq[pos + 1:]


_BASE_A = "MKTAYIAKQRQISFVKSHFSRQLEVRLGLIESQ" * 6
_BASE_B = "AAGGCCTTAAGGCCTTAAGGCCTTAAGGCCTTA" * 6
_BASE_C = "QWQWQWQWQWQWQWQWQWQWQWQWQWQWQWQWQ" * 6
_BASE_D = "RRSSRRSSRRSSRRSSRRSSRRSSRRSSRRSS" * 6


def _point_mutant_family_records(
    prefix: str, base: str, n_antigens: int, records_per_antigen: int = 3,
) -> list[dict]:
    rows = []
    for i in range(n_antigens):
        sequence = _mutate(base, i % len(base), "X")
        for j in range(records_per_antigen):
            rows.append(_antigen_cluster_row(f"{prefix}_{i}/{j}", f"{prefix}_{i}", sequence))
    return rows


def test_antigen_cluster_holdout_split_keeps_near_duplicate_antigens_together():
    # Family Ag is 30 point-mutant variants of the same base sequence
    # (>99% pairwise identity) -- under group_holdout_split they could land
    # on opposite sides since they're different group_ids; under
    # antigen_cluster_holdout_split they must all land together. Families
    # B/C/D are unrelated filler so there are >=3 total cluster units to
    # split among (the partitioner requires at least 3 units).
    records = pd.DataFrame(
        _point_mutant_family_records("Ag", _BASE_A, n_antigens=30)
        + _point_mutant_family_records("B", _BASE_B, n_antigens=5)
        + _point_mutant_family_records("C", _BASE_C, n_antigens=5)
        + _point_mutant_family_records("D", _BASE_D, n_antigens=5)
    )
    clusters = compute_antigen_clusters(records, similarity_threshold=0.9, linkage_method="average")
    ag_cluster_ids = set(clusters.loc[clusters["antigen_key"].str.startswith("Ag_"), "antigen_cluster_id"])
    assert len(ag_cluster_ids) == 1  # sanity: family Ag really did cluster as one

    result = build_splits(
        records, strategy="antigen_cluster_holdout_split",
        valid_fraction=0.3, test_fraction=0.2, seed=0, antigen_clusters=clusters,
    )

    train_keys = {key for key in result.train["antigen_key"] if key.startswith("Ag_")}
    valid_keys = {key for key in result.valid["antigen_key"] if key.startswith("Ag_")}
    test_keys = {key for key in result.test["antigen_key"] if key.startswith("Ag_")}
    # Family Ag is a single antigen_cluster_id, so it must land entirely on
    # one side -- at most one of these three sets is non-empty.
    assert sum(bool(s) for s in (train_keys, valid_keys, test_keys)) == 1


def test_antigen_cluster_holdout_split_leakage_report_includes_antigen_cluster_overlap():
    records = pd.DataFrame(
        _point_mutant_family_records("A", _BASE_A, n_antigens=15)
        + _point_mutant_family_records("B", _BASE_B, n_antigens=15)
        + _point_mutant_family_records("C", _BASE_C, n_antigens=15)
        + _point_mutant_family_records("D", _BASE_D, n_antigens=15)
    )

    clusters = compute_antigen_clusters(records, similarity_threshold=0.9, linkage_method="average")
    assert clusters["antigen_cluster_id"].nunique() == 4  # sanity: 4 well-separated families
    result = build_splits(
        records, strategy="antigen_cluster_holdout_split",
        valid_fraction=0.25, test_fraction=0.25, seed=0, antigen_clusters=clusters,
    )

    assert "antigen_cluster_overlap" in result.leakage_report["check_name"].tolist()
    assert "group_id_overlap" in result.leakage_report["check_name"].tolist()
    assert (result.leakage_report["status"] == "PASS").all()
    # the helper column used internally for the leakage check must not leak
    # into the user-facing output
    assert "_antigen_cluster_id" not in result.train.columns
    assert "_antigen_cluster_id" not in result.valid.columns
    assert "_antigen_cluster_id" not in result.test.columns


def test_antigen_cluster_holdout_split_requires_antigen_clusters_argument():
    records = pd.DataFrame(_point_mutant_family_records("A", _BASE_A, n_antigens=10))
    with pytest.raises(ValueError, match="antigen_clusters is required"):
        build_splits(
            records, strategy="antigen_cluster_holdout_split",
            valid_fraction=0.2, test_fraction=0.2, seed=0,
        )


def test_antigen_cluster_holdout_split_rejects_unmapped_antigen_key():
    records = pd.DataFrame(_point_mutant_family_records("A", _BASE_A, n_antigens=10))
    clusters = compute_antigen_clusters(records, similarity_threshold=0.9)
    stray_row = pd.DataFrame([_antigen_cluster_row("stray/0", "NotInClusters", "Z" * 50)])
    records_with_stray = pd.concat([records, stray_row], ignore_index=True)

    with pytest.raises(ValueError, match="missing from antigen_clusters"):
        build_splits(
            records_with_stray, strategy="antigen_cluster_holdout_split",
            valid_fraction=0.2, test_fraction=0.2, seed=0, antigen_clusters=clusters,
        )
