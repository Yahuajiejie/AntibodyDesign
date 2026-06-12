"""Tests for affinity_transformer.dataset (spec §5.2 / §7.2 / §7.3)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from affinity_transformer.dataset import (
    GROUP_COLUMNS,
    PAIR_COLUMNS,
    REQUIRED_COLUMNS,
    AffinityGroupExample,
    AffinityRecordDataset,
    ListwiseAffinityDataset,
    PairwiseAffinityDataset,
    build_groups,
    build_pairs,
    filter_trainable_records,
    load_records,
)

FV_GROUP = "studyA/tableA/agA/neg_log10_kd_M/experimental"
BINARY_GROUP = "studyD/tableD/agD/bind/binary"
SINGLE_LABEL_GROUP = "studyE/tableE/agE/neg_log10_kd_M/experimental"


# ── load_records ─────────────────────────────────────────────────────────────


def test_load_records_rejects_missing_required_columns(tmp_path, toy_records):
    # load record需要检查是否缺失必要列
    incomplete = toy_records.drop(columns=["group_id"])
    path = tmp_path / "records.parquet"
    incomplete.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="group_id"):
        load_records(path)


def test_load_records_round_trip(tmp_path, toy_records):
    # load record只负责原封不动搬表格
    path = tmp_path / "records.parquet"
    toy_records.to_parquet(path, index=False)

    loaded = load_records(path)

    assert set(REQUIRED_COLUMNS).issubset(loaded.columns)
    assert len(loaded) == len(toy_records)


def test_load_records_missing_file(tmp_path):
    # load record 能正确处理确实文件
    with pytest.raises(FileNotFoundError):
        load_records(tmp_path / "does_not_exist.parquet")


def test_load_records_unsupported_extension(tmp_path):
    # load record只处理两类文件 其他的都不行
    path = tmp_path / "records.txt"
    path.write_text("not a table")

    with pytest.raises(ValueError):
        load_records(path)


# ── filter_trainable_records ─────────────────────────────────────────────────


def test_filter_trainable_records_drops_excluded_and_nonfinite(toy_records):
    # train records只包含keep for training = True 和 rank_label 有限的数值
    filtered = filter_trainable_records(toy_records)

    assert "studyF/tableF/2" not in filtered["record_id"].values  # keep_for_training=False
    assert "studyG/tableG/2" not in filtered["record_id"].values  # rank_label = NaN
    assert filtered["keep_for_training"].all()
    assert filtered["rank_label"].apply(math.isfinite).all()


def test_filter_trainable_records_does_not_mutate_input(toy_records):
    # filter函数不准修改原始表格
    original = toy_records.copy(deep=True)

    filter_trainable_records(toy_records)

    pd.testing.assert_frame_equal(toy_records, original)


def test_filter_trainable_records_rejects_missing_columns():
    # filter函数拒绝缺失的行
    with pytest.raises(ValueError):
        filter_trainable_records(pd.DataFrame({"record_id": ["a"]}))


# ── build_pairs ───────────────────────────────────────────────────────────────


def test_build_pairs_never_crosses_group_id(toy_records):
    # build_pairs不准跨越组别
    pairs = build_pairs(toy_records, max_pairs_per_group=10, seed=0)

    records_by_id = toy_records.set_index("record_id")
    for _, pair in pairs.iterrows():
        group_i = records_by_id.loc[pair["record_id_i"], "group_id"]
        group_j = records_by_id.loc[pair["record_id_j"], "group_id"]
        assert group_i == group_j == pair["group_id"]


def test_build_pairs_skips_equal_labels(toy_records):
    # build pair不能自己跟自己比
    pairs = build_pairs(toy_records, max_pairs_per_group=10, seed=0)

    assert (pairs["group_id"] == SINGLE_LABEL_GROUP).sum() == 0
    assert not (pairs["label_i"] == pairs["label_j"]).any()


def test_build_pairs_binary_group_only_pairs_across_classes(toy_records):
    # 一定是一个正样本，一个负样本
    pairs = build_pairs(toy_records, max_pairs_per_group=10, seed=0)

    binary_pairs = pairs[pairs["group_id"] == BINARY_GROUP]
    assert len(binary_pairs) == 4  # 2 positive x 2 negative
    assert set(zip(binary_pairs["label_i"], binary_pairs["label_j"])) == {(1.0, 0.0)}


def test_build_pairs_reproducible_with_fixed_seed(toy_records):
    # 同一个种子，弄出来的pair是一样的
    pairs_a = build_pairs(toy_records, max_pairs_per_group=2, seed=42)
    pairs_b = build_pairs(toy_records, max_pairs_per_group=2, seed=42)

    pd.testing.assert_frame_equal(pairs_a, pairs_b)


def test_build_pairs_respects_max_pairs_per_group(toy_records):
    # 不准超上限
    pairs = build_pairs(toy_records, max_pairs_per_group=2, seed=0)

    assert (pairs["group_id"] == FV_GROUP).sum() == 2


def test_build_pairs_rejects_invalid_max_pairs_per_group(toy_records):
    # 不准乱设上限
    with pytest.raises(ValueError):
        build_pairs(toy_records, max_pairs_per_group=0, seed=0)


def test_build_pairs_empty_when_no_trainable_pairs():
    # 没有可训练对时，不构建pairs
    empty = pd.DataFrame(columns=REQUIRED_COLUMNS)

    pairs = build_pairs(empty, max_pairs_per_group=10, seed=0)

    assert list(pairs.columns) == list(PAIR_COLUMNS)
    assert len(pairs) == 0


# ── build_groups ─────────────────────────────────────────────────────────────


def test_build_groups_never_crosses_group_id(toy_records):
    groups = build_groups(toy_records, max_group_size=None, seed=0)

    records_by_id = toy_records.set_index("record_id")
    for _, row in groups.iterrows():
        assert records_by_id.loc[row["record_id"], "group_id"] == row["group_id"]


def test_build_groups_skips_single_label_groups(toy_records):
    groups = build_groups(toy_records, max_group_size=None, seed=0)

    assert (groups["group_id"] == SINGLE_LABEL_GROUP).sum() == 0


def test_build_groups_keeps_binary_group_with_both_classes(toy_records):
    groups = build_groups(toy_records, max_group_size=None, seed=0)

    binary_groups = groups[groups["group_id"] == BINARY_GROUP]
    assert len(binary_groups) == 4  # 2 positive + 2 negative
    assert set(binary_groups["rank_label"]) == {0.0, 1.0}


def test_build_groups_reproducible_with_fixed_seed(toy_records):
    groups_a = build_groups(toy_records, max_group_size=2, seed=42)
    groups_b = build_groups(toy_records, max_group_size=2, seed=42)

    pd.testing.assert_frame_equal(groups_a, groups_b)


def test_build_groups_respects_max_group_size(toy_records):
    groups = build_groups(toy_records, max_group_size=2, seed=0)

    assert (groups["group_id"] == FV_GROUP).sum() == 2


def test_build_groups_rejects_invalid_max_group_size(toy_records):
    with pytest.raises(ValueError):
        build_groups(toy_records, max_group_size=1, seed=0)


def test_build_groups_empty_when_no_trainable_groups():
    empty = pd.DataFrame(columns=REQUIRED_COLUMNS)

    groups = build_groups(empty, max_group_size=None, seed=0)

    assert list(groups.columns) == list(GROUP_COLUMNS)
    assert len(groups) == 0


# ── AffinityRecordDataset / PairwiseAffinityDataset ─────────────────────────


def test_affinity_record_dataset_basic(toy_records):
    trainable = filter_trainable_records(toy_records)
    dataset = AffinityRecordDataset(trainable)

    assert len(dataset) == len(trainable)
    example = dataset[0]
    assert example.record_id == trainable.iloc[0]["record_id"]
    assert isinstance(example.rank_label, float)


def test_affinity_record_dataset_rejects_missing_columns():
    with pytest.raises(ValueError):
        AffinityRecordDataset(pd.DataFrame({"record_id": ["a"]}))


def test_pairwise_affinity_dataset_basic(toy_records):
    trainable = filter_trainable_records(toy_records)
    pairs = build_pairs(toy_records, max_pairs_per_group=10, seed=0)
    dataset = PairwiseAffinityDataset(trainable, pairs)

    assert len(dataset) == len(pairs)
    pair_example = dataset[0]
    assert pair_example.left.record_id != pair_example.right.record_id
    assert pair_example.y_ij in (0.0, 1.0)
    assert pair_example.group_id == pairs.iloc[0]["group_id"]


def test_pairwise_affinity_dataset_rejects_missing_pair_columns(toy_records):
    trainable = filter_trainable_records(toy_records)

    with pytest.raises(ValueError):
        PairwiseAffinityDataset(trainable, pd.DataFrame({"pair_id": ["a"]}))


# ── ListwiseAffinityDataset ──────────────────────────────────────────────────


def test_listwise_affinity_dataset_basic(toy_records):
    trainable = filter_trainable_records(toy_records)
    groups = build_groups(toy_records, max_group_size=None, seed=0)
    dataset = ListwiseAffinityDataset(trainable, groups)

    assert len(dataset) == groups["group_id"].nunique()

    fv_index = next(i for i in range(len(dataset)) if dataset[i].group_id == FV_GROUP)
    fv_example = dataset[fv_index]

    assert isinstance(fv_example, AffinityGroupExample)
    assert len(fv_example.examples) == 3
    assert {ex.record_id for ex in fv_example.examples} == set(
        groups.loc[groups["group_id"] == FV_GROUP, "record_id"]
    )
    assert len({ex.rank_label for ex in fv_example.examples}) == 3


def test_listwise_affinity_dataset_binary_group_label_kind(toy_records):
    trainable = filter_trainable_records(toy_records)
    groups = build_groups(toy_records, max_group_size=None, seed=0)
    dataset = ListwiseAffinityDataset(trainable, groups)

    binary_index = next(i for i in range(len(dataset)) if dataset[i].group_id == BINARY_GROUP)
    binary_example = dataset[binary_index]

    assert binary_example.label_kind == "binary"
    assert {ex.rank_label for ex in binary_example.examples} == {0.0, 1.0}


def test_listwise_affinity_dataset_rejects_missing_group_columns(toy_records):
    trainable = filter_trainable_records(toy_records)

    with pytest.raises(ValueError):
        ListwiseAffinityDataset(trainable, pd.DataFrame({"group_id": ["a"]}))


def test_listwise_affinity_dataset_rejects_missing_record_columns():
    groups = pd.DataFrame(columns=GROUP_COLUMNS)

    with pytest.raises(ValueError):
        ListwiseAffinityDataset(pd.DataFrame({"record_id": ["a"]}), groups)
