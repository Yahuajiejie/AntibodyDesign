"""Tests for affinity_transformer.dataset (spec §5.2 / §7.2 / §7.3)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

import affinity_transformer.dataset.pair_sampling.large_group as large_group_module
import affinity_transformer.dataset.pairs as pairs_module
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
LARGE_GROUP = "studyLarge/tableLarge/agLarge/neg_log10_kd_M/experimental"


def _standard_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "record_id": "record",
        "dataset_id": "studyLarge/tableLarge",
        "study_id": "studyLarge",
        "table_id": "tableLarge",
        "source_file": "data/binding/studyLarge/tableLarge.csv",
        "source_row": 0,
        "antibody_id": "ab",
        "antibody_type": "Fv",
        "heavy_chain": "QVQLVQSGAEVKKPGASVKVSCKAS",
        "light_chain": "DIQMTQSPSSLSASVGDRVTITC",
        "single_chain_sequence": None,
        "antigen_key": "agLarge",
        "antigen_name": "Antigen Large",
        "antigen_sequence": "MKTAYIAKQRQISFVKSHFSRQLE",
        "antigen_source": "provided",
        "assay_name": "SPR",
        "assay_type": "binding",
        "metric_name": "neg_log10_kd_M",
        "metric_value_raw": "1.0",
        "metric_value_numeric": 1.0,
        "metric_unit": "-log10(KD/M)",
        "metric_direction": "higher_is_better",
        "transform_rule": "rank_label = neg_log10_kd_M",
        "rank_label": 1.0,
        "label_kind": "experimental",
        "group_id": LARGE_GROUP,
        "keep_for_training": True,
        "drop_reason": None,
    }
    row.update(overrides)
    return row


def _large_continuous_records(n_records: int) -> pd.DataFrame:
    return pd.DataFrame([
        _standard_row(
            record_id=f"large/{i:05d}",
            antibody_id=f"ab-{i:05d}",
            source_row=i,
            rank_label=float(i),
            metric_value_raw=str(i),
            metric_value_numeric=float(i),
        )
        for i in range(n_records)
    ])


def _large_binary_records(n_negative: int, n_positive: int) -> pd.DataFrame:
    rows = []
    for i in range(n_negative):
        rows.append(_standard_row(
            record_id=f"binary/neg/{i:05d}",
            antibody_id=f"neg-{i:05d}",
            source_row=i,
            label_kind="binary",
            rank_label=0.0,
            metric_name="bind",
            metric_value_raw="0",
            metric_value_numeric=0.0,
            metric_unit=None,
            transform_rule="rank_label = bind (0/1)",
            group_id="studyLarge/tableLarge/agLarge/bind/binary",
        ))
    for i in range(n_positive):
        rows.append(_standard_row(
            record_id=f"binary/pos/{i:05d}",
            antibody_id=f"pos-{i:05d}",
            source_row=n_negative + i,
            label_kind="binary",
            rank_label=1.0,
            metric_name="bind",
            metric_value_raw="1",
            metric_value_numeric=1.0,
            metric_unit=None,
            transform_rule="rank_label = bind (0/1)",
            group_id="studyLarge/tableLarge/agLarge/bind/binary",
        ))
    return pd.DataFrame(rows)


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


def test_build_pairs_supports_capped_proportional_sampling(toy_records):
    pairs = build_pairs(
        toy_records,
        max_pairs_per_group=3,
        seed=0,
        pair_sample_strategy="capped_proportional",
        pair_fraction=0.5,
        min_pairs_per_group=1,
    )

    assert (pairs["group_id"] == FV_GROUP).sum() == 2  # ceil(3 candidate pairs * 0.5)
    assert (pairs["group_id"] == BINARY_GROUP).sum() == 2  # ceil(4 candidate pairs * 0.5)


def test_build_pairs_rejects_invalid_proportional_sampling(toy_records):
    with pytest.raises(ValueError, match="pair_fraction"):
        build_pairs(
            toy_records,
            max_pairs_per_group=3,
            seed=0,
            pair_sample_strategy="capped_proportional",
            pair_fraction=None,
        )


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


def test_build_pairs_large_group_does_not_enumerate_all_pairs(monkeypatch):
    records = _large_continuous_records(20_000)

    def fail_if_called(group):
        raise AssertionError("_candidate_pairs must not be called for large groups")

    monkeypatch.setattr(pairs_module, "_candidate_pairs", fail_if_called)

    pairs = build_pairs(
        records,
        max_pairs_per_group=40,
        seed=0,
        large_group_threshold=1_000,
        pair_enumeration_limit=1_000,
        label_block_count=5,
        intra_block_pairs_per_large_group=10,
    )

    assert len(pairs) == 50
    assert not (pairs["label_i"] == pairs["label_j"]).any()


def test_build_pairs_large_continuous_group_reproducible_with_fixed_seed():
    records = _large_continuous_records(20_000)

    pairs_a = build_pairs(
        records,
        max_pairs_per_group=40,
        seed=123,
        large_group_threshold=1_000,
        pair_enumeration_limit=1_000,
        label_block_count=5,
        intra_block_pairs_per_large_group=10,
    )
    pairs_b = build_pairs(
        records,
        max_pairs_per_group=40,
        seed=123,
        large_group_threshold=1_000,
        pair_enumeration_limit=1_000,
        label_block_count=5,
        intra_block_pairs_per_large_group=10,
    )

    pd.testing.assert_frame_equal(pairs_a, pairs_b)


def test_build_pairs_large_block_sampler_includes_cross_and_intra_block_pairs():
    records = _large_continuous_records(20_000)

    pairs = build_pairs(
        records,
        max_pairs_per_group=40,
        seed=0,
        large_group_threshold=1_000,
        pair_enumeration_limit=1_000,
        label_block_count=5,
        intra_block_pairs_per_large_group=10,
    )

    block_i = (pairs["label_i"] // 4000).astype(int)
    block_j = (pairs["label_j"] // 4000).astype(int)
    assert (block_i != block_j).any()
    assert (block_i == block_j).any()


def _degree_and_adjacency(pairs: pd.DataFrame):
    import collections

    degree: dict[str, int] = collections.defaultdict(int)
    adjacency: dict[str, set[str]] = collections.defaultdict(set)
    for _, row in pairs.iterrows():
        a, b = row["record_id_i"], row["record_id_j"]
        degree[a] += 1
        degree[b] += 1
        adjacency[a].add(b)
        adjacency[b].add(a)
    return degree, adjacency


def _assert_connected(pairs: pd.DataFrame, record_ids) -> None:
    """The whole group must be one connected component (no record stranded)."""
    import collections

    record_ids = list(record_ids)
    _, adjacency = _degree_and_adjacency(pairs)

    visited = {record_ids[0]}
    queue = collections.deque([record_ids[0]])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    assert visited == set(record_ids), "pair graph is not fully connected"


def _assert_connected_with_min_degree_two(pairs: pd.DataFrame, record_ids) -> None:
    """Shared shape-check for the tree-shaped strategies below.

    Every record must appear in at least 2 edges (no degree-1 leaves left
    after `_add_redundancy_edges`) and the whole group must be one
    connected component (no record stranded by itself).
    """
    record_ids = list(record_ids)
    degree, _ = _degree_and_adjacency(pairs)

    if len(record_ids) <= 3:
        return  # too few records for "every node has 2 distinct neighbors" to be meaningful

    for record_id in record_ids:
        assert degree[record_id] >= 2, f"{record_id} has degree {degree[record_id]} < 2"

    _assert_connected(pairs, record_ids)


def test_build_pairs_balanced_tree_strategy_basic_properties():
    n = 500
    records = _large_continuous_records(n)

    pairs = build_pairs(records, max_pairs_per_group=10**9, seed=0, pair_sample_strategy="balanced_tree")

    assert not (pairs["label_i"] == pairs["label_j"]).any()
    assert n - 1 <= len(pairs) <= 3 * n  # backbone (n-1) plus at most 2 redundancy edges/node
    _assert_connected_with_min_degree_two(pairs, records["record_id"])

    pairs_again = build_pairs(records, max_pairs_per_group=10**9, seed=0, pair_sample_strategy="balanced_tree")
    pd.testing.assert_frame_equal(pairs, pairs_again)  # deterministic: same seed -> identical tree


def test_build_pairs_randomized_bst_strategy_basic_properties():
    n = 500
    records = _large_continuous_records(n)

    pairs = build_pairs(records, max_pairs_per_group=10**9, seed=0, pair_sample_strategy="randomized_bst")

    assert not (pairs["label_i"] == pairs["label_j"]).any()
    assert n - 1 <= len(pairs) <= 3 * n
    _assert_connected_with_min_degree_two(pairs, records["record_id"])

    pairs_same_seed = build_pairs(
        records, max_pairs_per_group=10**9, seed=0, pair_sample_strategy="randomized_bst"
    )
    pd.testing.assert_frame_equal(pairs, pairs_same_seed)  # same seed -> same insertion order

    pairs_other_seed = build_pairs(
        records, max_pairs_per_group=10**9, seed=1, pair_sample_strategy="randomized_bst"
    )
    assert not pairs.equals(pairs_other_seed)  # different seed -> different tree shape


def test_build_pairs_balanced_tree_add_redundancy_edges_false_yields_bare_backbone():
    n = 500
    records = _large_continuous_records(n)  # all-distinct rank_label -> no ties to drop

    backbone = build_pairs(
        records, max_pairs_per_group=10**9, seed=0,
        pair_sample_strategy="balanced_tree", add_redundancy_edges=False,
    )
    with_redundancy = build_pairs(
        records, max_pairs_per_group=10**9, seed=0,
        pair_sample_strategy="balanced_tree", add_redundancy_edges=True,
    )

    assert not (backbone["label_i"] == backbone["label_j"]).any()
    assert len(backbone) == n - 1  # exactly the tree backbone: no ties here, no redundancy edges
    _assert_connected(backbone, records["record_id"])  # a tree is connected on its own

    degree, _ = _degree_and_adjacency(backbone)
    assert any(degree[r] == 1 for r in records["record_id"])  # leaves are no longer patched to >=2

    # The backbone is a strict subset of what add_redundancy_edges=True produces.
    backbone_keys = set(zip(backbone["record_id_i"], backbone["record_id_j"]))
    redundancy_keys = set(zip(with_redundancy["record_id_i"], with_redundancy["record_id_j"]))
    assert backbone_keys.issubset(redundancy_keys)
    assert len(with_redundancy) > len(backbone)

    # Still deterministic for a fixed seed.
    backbone_again = build_pairs(
        records, max_pairs_per_group=10**9, seed=0,
        pair_sample_strategy="balanced_tree", add_redundancy_edges=False,
    )
    pd.testing.assert_frame_equal(backbone, backbone_again)


def test_build_pairs_randomized_bst_add_redundancy_edges_false_yields_bare_backbone():
    n = 500
    records = _large_continuous_records(n)

    backbone = build_pairs(
        records, max_pairs_per_group=10**9, seed=0,
        pair_sample_strategy="randomized_bst", add_redundancy_edges=False,
    )
    with_redundancy = build_pairs(
        records, max_pairs_per_group=10**9, seed=0,
        pair_sample_strategy="randomized_bst", add_redundancy_edges=True,
    )

    assert not (backbone["label_i"] == backbone["label_j"]).any()
    assert len(backbone) == n - 1  # bare insertion-order backbone, no ties, no redundancy edges
    _assert_connected(backbone, records["record_id"])

    degree, _ = _degree_and_adjacency(backbone)
    assert any(degree[r] == 1 for r in records["record_id"])

    backbone_keys = set(zip(backbone["record_id_i"], backbone["record_id_j"]))
    redundancy_keys = set(zip(with_redundancy["record_id_i"], with_redundancy["record_id_j"]))
    assert backbone_keys.issubset(redundancy_keys)
    assert len(with_redundancy) > len(backbone)

    # Same seed -> same insertion order even with redundancy edges off.
    backbone_same_seed = build_pairs(
        records, max_pairs_per_group=10**9, seed=0,
        pair_sample_strategy="randomized_bst", add_redundancy_edges=False,
    )
    pd.testing.assert_frame_equal(backbone, backbone_same_seed)

    # Different seed -> different insertion order even with redundancy edges off.
    backbone_other_seed = build_pairs(
        records, max_pairs_per_group=10**9, seed=1,
        pair_sample_strategy="randomized_bst", add_redundancy_edges=False,
    )
    assert not backbone.equals(backbone_other_seed)


def test_build_pairs_rejects_dropped_heap_array_strategy(toy_records):
    # heap_array was implemented, measured, and dropped (systematic bias toward
    # one half of the affinity range) -- it must stay rejected, not silently
    # resurface as a valid value.
    with pytest.raises(ValueError, match="pair_sample_strategy"):
        build_pairs(toy_records, max_pairs_per_group=10, seed=0, pair_sample_strategy="heap_array")


def test_build_pairs_large_imbalanced_binary_group_samples_across_classes(monkeypatch):
    records = _large_binary_records(n_negative=5_000, n_positive=10)

    def fail_if_called(group, label_block_count):
        raise AssertionError("binary large groups should not build quantile blocks")

    monkeypatch.setattr(large_group_module, "_build_label_blocks", fail_if_called)

    pairs = build_pairs(
        records,
        max_pairs_per_group=30,
        seed=0,
        large_group_threshold=1_000,
        pair_enumeration_limit=1_000,
        label_block_count=5,
        intra_block_pairs_per_large_group=10,
    )

    assert len(pairs) == 30
    assert {frozenset(labels) for labels in zip(pairs["label_i"], pairs["label_j"])} == {
        frozenset({0.0, 1.0})
    }


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
