"""Protocol tests for global antibody- and antigen-cold-start splitting."""

from __future__ import annotations

import pandas as pd
import pytest

from affinity_transformer.dataset import build_pairs
from affinity_transformer.splits import (
    build_antibody_cold_start_kfolds,
    build_antibody_cold_start_split,
    build_antigen_cold_start_kfolds,
    build_antigen_cold_start_split,
    build_splits,
)


def _row(
    *,
    record_id: str,
    group_id: str,
    antibody: str,
    antibody_cluster: str,
    antigen: str,
    antigen_cluster: str,
    label: float,
    measurement_family: str | None = None,
) -> dict[str, object]:
    return {
        "record_id": record_id,
        "group_id": group_id,
        "dataset_id": "study/table",
        "keep_for_training": True,
        "rank_label": label,
        "label_kind": "experimental",
        "measurement_family_id": measurement_family or f"mf/{record_id}",
        "antibody_sequence_key": f"ab-seq/{antibody}",
        "antibody_cluster_id": f"ab-cluster/{antibody_cluster}",
        "antigen_sequence_key": f"ag-seq/{antigen}",
        "antigen_cluster_id": f"ag-cluster/{antigen_cluster}",
        "interaction_key": f"interaction/{antigen}/{antibody}",
        "effective_antigen_input_hash": f"effective/{antigen}",
    }


def _antibody_cold_records(n_antibodies: int = 12) -> pd.DataFrame:
    rows = []
    for antibody_index in range(n_antibodies):
        for antigen in ("A", "B"):
            rows.append(_row(
                record_id=f"{antigen}/ab-{antibody_index}",
                group_id=f"group/{antigen}",
                antibody=f"ab-{antibody_index}",
                antibody_cluster=f"ab-{antibody_index}",
                antigen=antigen,
                antigen_cluster=antigen,
                label=float(antibody_index),
            ))
    return pd.DataFrame(rows)


def _antigen_cold_records(n_antigens: int = 9) -> pd.DataFrame:
    rows = []
    for antigen_index in range(n_antigens):
        antigen = f"ag-{antigen_index}"
        for antibody_index, antibody in enumerate(("shared-a", "shared-b", "shared-c")):
            rows.append(_row(
                record_id=f"{antigen}/{antibody}",
                group_id=f"group/{antigen}",
                antibody=antibody,
                antibody_cluster=antibody,
                antigen=antigen,
                antigen_cluster=antigen,
                label=float(antibody_index),
            ))
        # This record becomes dual-cold whenever its antigen is held out.
        rows.append(_row(
            record_id=f"{antigen}/unique",
            group_id=f"group/{antigen}",
            antibody=f"unique-{antigen}",
            antibody_cluster=f"unique-{antigen}",
            antigen=antigen,
            antigen_cluster=antigen,
            label=3.0,
        ))
    return pd.DataFrame(rows)


def _clusters(frame: pd.DataFrame, column: str) -> set[str]:
    return set(frame[column].astype(str))


def test_antibody_cold_start_is_global_not_group_local_and_builds_pairs():
    records = _antibody_cold_records()
    result = build_antibody_cold_start_split(
        records,
        valid_fraction=0.25,
        test_fraction=0.25,
        seed=7,
        min_eval_records=2,
    )

    train_clusters = _clusters(result.train, "antibody_cluster_id")
    valid_clusters = _clusters(result.valid, "antibody_cluster_id")
    test_clusters = _clusters(result.test, "antibody_cluster_id")
    assert train_clusters.isdisjoint(valid_clusters)
    assert train_clusters.isdisjoint(test_clusters)
    assert valid_clusters.isdisjoint(test_clusters)

    # The same antibody occurs under two antigens, but its global assignment
    # is identical in every group.
    split_by_record = {
        str(record_id): split_name
        for split_name, frame in (
            ("train", result.train), ("valid", result.valid), ("test", result.test)
        )
        for record_id in frame["record_id"]
    }
    for antibody_index in range(12):
        assert split_by_record[f"A/ab-{antibody_index}"] == split_by_record[
            f"B/ab-{antibody_index}"
        ]

    # Both exact antigens and both groups are known from train.
    assert _clusters(result.valid, "antigen_sequence_key") <= _clusters(
        result.train, "antigen_sequence_key"
    )
    assert _clusters(result.valid, "group_id") <= _clusters(result.train, "group_id")
    assert build_pairs(result.train, max_pairs_per_group=100, seed=0).shape[0] > 0
    assert build_pairs(result.valid, max_pairs_per_group=100, seed=0).shape[0] > 0
    assert result.excluded_records.empty


def test_antibody_cold_start_measurement_family_merges_antibody_clusters():
    records = _antibody_cold_records()
    # Bridge two antibody clusters through one measurement family. They must
    # become one component and therefore share a split globally.
    bridge_mask = records["antibody_cluster_id"].isin(
        {"ab-cluster/ab-0", "ab-cluster/ab-1"}
    )
    records.loc[bridge_mask, "measurement_family_id"] = "mf/shared-bridge"
    result = build_antibody_cold_start_split(
        records, valid_fraction=0.25, test_fraction=0.25, seed=3
    )
    all_outputs = pd.concat([result.train, result.valid, result.test], ignore_index=True)
    locations = {}
    for split_name, frame in (
        ("train", result.train), ("valid", result.valid), ("test", result.test)
    ):
        for cluster_id in frame["antibody_cluster_id"].astype(str).unique():
            locations.setdefault(cluster_id, set()).add(split_name)
    assert locations["ab-cluster/ab-0"] == locations["ab-cluster/ab-1"]
    assert len(all_outputs) == len(records)


def test_antigen_cold_start_keeps_only_train_seen_antibodies_for_evaluation():
    records = _antigen_cold_records()
    result = build_antigen_cold_start_split(
        records,
        valid_fraction=0.22,
        test_fraction=0.22,
        seed=5,
        min_eval_records=2,
    )

    train_antigens = _clusters(result.train, "antigen_cluster_id")
    valid_antigens = _clusters(result.valid, "antigen_cluster_id")
    test_antigens = _clusters(result.test, "antigen_cluster_id")
    assert train_antigens.isdisjoint(valid_antigens)
    assert train_antigens.isdisjoint(test_antigens)
    assert valid_antigens.isdisjoint(test_antigens)
    train_antibodies = _clusters(result.train, "antibody_cluster_id")
    assert _clusters(result.valid, "antibody_cluster_id") <= train_antibodies
    assert _clusters(result.test, "antibody_cluster_id") <= train_antibodies

    assert not result.excluded_records.empty
    assert set(result.excluded_records["protocol_exclusion_reason"]) == {
        "antibody_cluster_not_seen_in_train"
    }
    assert build_pairs(result.valid, max_pairs_per_group=100, seed=0).shape[0] > 0
    assert build_pairs(result.test, max_pairs_per_group=100, seed=0).shape[0] > 0


def test_antibody_cold_start_kfolds_isolate_global_antibody_components():
    records = _antibody_cold_records()
    folds = build_antibody_cold_start_kfolds(records, n_splits=3, seed=11)

    validation_cluster_counts: dict[str, int] = {}
    for fold in folds:
        train_clusters = _clusters(fold.train, "antibody_cluster_id")
        valid_clusters = _clusters(fold.valid, "antibody_cluster_id")
        assert train_clusters.isdisjoint(valid_clusters)
        assert _clusters(fold.valid, "group_id") <= _clusters(fold.train, "group_id")
        for cluster_id in valid_clusters:
            validation_cluster_counts[cluster_id] = validation_cluster_counts.get(cluster_id, 0) + 1
    assert validation_cluster_counts
    assert set(validation_cluster_counts.values()) == {1}


def test_antigen_cold_start_kfolds_isolate_antigens_and_require_seen_antibodies():
    records = _antigen_cold_records()
    folds = build_antigen_cold_start_kfolds(records, n_splits=3, seed=13)

    validation_antigen_counts: dict[str, int] = {}
    for fold in folds:
        train_antigens = _clusters(fold.train, "antigen_cluster_id")
        valid_antigens = _clusters(fold.valid, "antigen_cluster_id")
        assert train_antigens.isdisjoint(valid_antigens)
        assert _clusters(fold.valid, "antibody_cluster_id") <= _clusters(
            fold.train, "antibody_cluster_id"
        )
        assert "antibody_cluster_not_seen_in_train" in set(
            fold.excluded_records["protocol_exclusion_reason"]
        )
        for antigen_id in valid_antigens:
            validation_antigen_counts[antigen_id] = validation_antigen_counts.get(antigen_id, 0) + 1
    assert set(validation_antigen_counts.values()) == {1}


def test_cold_start_split_requires_precomputed_identity_fields():
    records = _antibody_cold_records().drop(columns=["antibody_cluster_id"])

    with pytest.raises(ValueError, match="cold-start identity column"):
        build_antibody_cold_start_split(
            records, valid_fraction=0.2, test_fraction=0.2, seed=0
        )


def test_cold_start_split_rejects_conflicting_labels_for_identical_interaction():
    records = _antibody_cold_records()
    duplicate = records.iloc[[0]].copy()
    duplicate["record_id"] = "conflicting-duplicate"
    duplicate["measurement_family_id"] = "mf/conflicting-duplicate"
    duplicate["rank_label"] = 999.0
    records = pd.concat([records, duplicate], ignore_index=True)

    with pytest.raises(ValueError, match="conflicting rank labels"):
        build_antibody_cold_start_split(
            records, valid_fraction=0.2, test_fraction=0.2, seed=0
        )


def test_group_holdout_split_dispatch_remains_unchanged():
    records = pd.DataFrame([
        {
            "record_id": f"r-{group_index}-{record_index}",
            "group_id": f"group-{group_index}",
            "dataset_id": "study/table",
            "keep_for_training": True,
            "rank_label": float(record_index),
            "label_kind": "experimental",
        }
        for group_index in range(5)
        for record_index in range(2)
    ])

    result = build_splits(
        records,
        strategy="group_holdout_split",
        valid_fraction=0.2,
        test_fraction=0.2,
        seed=0,
    )

    train_groups = set(result.train["group_id"])
    valid_groups = set(result.valid["group_id"])
    test_groups = set(result.test["group_id"])
    assert train_groups.isdisjoint(valid_groups)
    assert train_groups.isdisjoint(test_groups)
    assert valid_groups.isdisjoint(test_groups)
