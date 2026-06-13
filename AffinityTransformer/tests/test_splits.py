"""Tests for affinity_transformer.splits."""

from __future__ import annotations

from affinity_transformer.splits import build_splits


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
