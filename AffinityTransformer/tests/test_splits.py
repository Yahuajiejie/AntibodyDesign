"""Tests for affinity_transformer.splits."""

from __future__ import annotations

import pandas as pd

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
