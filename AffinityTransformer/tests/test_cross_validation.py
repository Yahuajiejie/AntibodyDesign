"""Group K-fold orchestration tests."""

from __future__ import annotations

import json

import pandas as pd
import pytest
import torch
import yaml

from affinity_transformer.config import load_config
from affinity_transformer.training.cross_validation import (
    run_group_kfold_cross_validation,
)


def test_group_kfold_runner_trains_each_fold_and_aggregates(
    tmp_path, toy_records
):
    train_records = toy_records[toy_records["group_id"].isin(
        toy_records["group_id"].drop_duplicates().iloc[:3]
    )]
    valid_records = toy_records[toy_records["group_id"].isin(
        toy_records["group_id"].drop_duplicates().iloc[3:5]
    )]
    train_path = tmp_path / "train.csv"
    valid_path = tmp_path / "valid.csv"
    train_records.to_csv(train_path, index=False)
    valid_records.to_csv(valid_path, index=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "data": {
            "train_path": str(train_path),
            "valid_path": str(valid_path),
            "test_path": None,
            "max_pairs_per_group": 20,
            "seed": 0,
        },
        "model": {
            "antibody_encoder": "fake",
            "antigen_encoder": None,
            "d_model": 8,
            "use_cross_attention": False,
        },
        "train": {
            "batch_size": 2,
            "lr": 1e-3,
            "epochs": 1,
            "device": "cpu",
        },
        "cross_validation": {
            "enabled": True,
            "n_splits": 3,
            "source": "train_valid",
            "seed": 11,
        },
    }), encoding="utf-8")
    config = load_config(config_path)
    calls = []

    def fake_runner(config_path_arg, config_arg, output_dir, fold_train, fold_valid, test):
        del config_path_arg, config_arg, output_dir
        train = pd.read_parquet(fold_train)
        valid = pd.read_parquet(fold_valid)
        assert set(train["group_id"]).isdisjoint(set(valid["group_id"]))
        assert test is None
        calls.append((len(train), len(valid)))
        return {"valid_macro_spearman": float(len(calls))}

    output_dir = tmp_path / "cv-output"
    metrics = run_group_kfold_cross_validation(
        config_path, config, output_dir, fake_runner
    )

    assert len(calls) == 3
    assignments = pd.read_csv(output_dir / "fold_assignments.csv")
    expected_ids = set(pd.concat([train_records, valid_records])["record_id"])
    expected_ids -= set(toy_records.loc[
        ~toy_records["keep_for_training"] | toy_records["rank_label"].isna(),
        "record_id",
    ])
    assert set(assignments["record_id"]) == expected_ids
    assert assignments["record_id"].is_unique
    summary = json.loads(
        (output_dir / "cross_validation_summary.json").read_text(encoding="utf-8")
    )
    assert summary["n_splits"] == 3
    assert summary["metrics"]["valid_macro_spearman"]["mean"] == pytest.approx(2.0)
    assert metrics["cv_valid_macro_spearman_mean"] == pytest.approx(2.0)


def test_group_kfold_reseeds_each_model_before_runner(tmp_path, toy_records):
    records_path = tmp_path / "records.csv"
    toy_records.to_csv(records_path, index=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "data": {
            "train_path": str(records_path), "valid_path": None,
            "test_path": None, "max_pairs_per_group": 20, "seed": 41,
        },
        "model": {
            "antibody_encoder": "fake", "antigen_encoder": None,
            "d_model": 8, "use_cross_attention": False,
        },
        "train": {"batch_size": 2, "lr": 1e-3, "epochs": 1, "device": "cpu"},
        "cross_validation": {
            "enabled": True, "n_splits": 3, "source": "train", "seed": 11,
        },
    }), encoding="utf-8")
    config = load_config(config_path)
    samples = []

    def fake_runner(*args):
        samples.append(float(torch.rand(())))
        # Consume a variable amount of RNG state; the next fold must still
        # start from its own deterministic seed.
        torch.rand(len(samples) * 7)
        return {"valid_macro_spearman": 0.0}

    run_group_kfold_cross_validation(config_path, config, tmp_path / "out", fake_runner)
    expected = []
    for fold_index in range(3):
        torch.manual_seed(41 + fold_index)
        expected.append(float(torch.rand(())))
    assert samples == expected


def _write_protocol_records(tmp_path, toy_records, protocol):
    template = toy_records.iloc[0].to_dict()
    rows = []
    if protocol == "antibody_cold_start":
        for antibody_index in range(12):
            for antigen in ("A", "B"):
                row = dict(template)
                row.update({
                    "record_id": f"{antigen}/ab-{antibody_index}",
                    "group_id": f"group/{antigen}",
                    "dataset_id": "study/table",
                    "keep_for_training": True,
                    "rank_label": float(antibody_index),
                    "label_kind": "experimental",
                    "measurement_family_id": f"mf/{antigen}/ab-{antibody_index}",
                    "antibody_sequence_key": f"ab-seq/{antibody_index}",
                    "antibody_cluster_id": f"ab-cluster/{antibody_index}",
                    "antigen_sequence_key": f"ag-seq/{antigen}",
                    "antigen_cluster_id": f"ag-cluster/{antigen}",
                    "interaction_key": f"interaction/{antigen}/{antibody_index}",
                    "effective_antigen_input_hash": f"effective/{antigen}",
                })
                rows.append(row)
    else:
        for antigen_index in range(9):
            antigen = f"ag-{antigen_index}"
            for antibody_index, antibody in enumerate(
                ("shared-a", "shared-b", "shared-c", f"unique-{antigen}")
            ):
                row = dict(template)
                row.update({
                    "record_id": f"{antigen}/{antibody}",
                    "group_id": f"group/{antigen}",
                    "dataset_id": "study/table",
                    "keep_for_training": True,
                    "rank_label": float(antibody_index),
                    "label_kind": "experimental",
                    "measurement_family_id": f"mf/{antigen}/{antibody}",
                    "antibody_sequence_key": f"ab-seq/{antibody}",
                    "antibody_cluster_id": f"ab-cluster/{antibody}",
                    "antigen_sequence_key": f"ag-seq/{antigen}",
                    "antigen_cluster_id": f"ag-cluster/{antigen}",
                    "interaction_key": f"interaction/{antigen}/{antibody}",
                    "effective_antigen_input_hash": f"effective/{antigen}",
                })
                rows.append(row)
    path = tmp_path / f"{protocol}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.mark.parametrize(
    "protocol",
    ["antibody_cold_start", "antigen_cold_start"],
)
def test_cross_validation_dispatches_entity_protocols(
    tmp_path, toy_records, protocol
):
    records_path = _write_protocol_records(tmp_path, toy_records, protocol)
    config_path = tmp_path / f"{protocol}.yaml"
    config_path.write_text(yaml.safe_dump({
        "data": {
            "train_path": str(records_path), "valid_path": None,
            "test_path": None, "max_pairs_per_group": 20, "seed": 19,
        },
        "model": {
            "antibody_encoder": "fake", "antigen_encoder": None,
            "d_model": 8, "use_cross_attention": False,
        },
        "train": {"batch_size": 2, "lr": 1e-3, "epochs": 1, "device": "cpu"},
        "cross_validation": {
            "enabled": True,
            "n_splits": 3,
            "source": "train",
            "seed": 23,
            "protocol": protocol,
            "min_eval_records": 2,
        },
    }), encoding="utf-8")
    config = load_config(config_path)
    calls = []

    def fake_runner(config_path_arg, config_arg, output_dir, fold_train, fold_valid, test):
        del config_path_arg, config_arg, output_dir
        train = pd.read_parquet(fold_train)
        valid = pd.read_parquet(fold_valid)
        assert test is None
        if protocol == "antibody_cold_start":
            assert set(train["antibody_cluster_id"]).isdisjoint(
                set(valid["antibody_cluster_id"])
            )
            assert set(valid["antigen_sequence_key"]) <= set(
                train["antigen_sequence_key"]
            )
        else:
            assert set(train["antigen_cluster_id"]).isdisjoint(
                set(valid["antigen_cluster_id"])
            )
            assert set(valid["antibody_cluster_id"]) <= set(
                train["antibody_cluster_id"]
            )
        calls.append(len(valid))
        return {"valid_macro_spearman": 0.5}

    output_dir = tmp_path / f"out-{protocol}"
    run_group_kfold_cross_validation(
        config_path, config, output_dir, fake_runner
    )

    assert len(calls) == 3
    assert (output_dir / "unit_assignments.parquet").exists()
    for fold_index in range(1, 4):
        fold_dir = output_dir / f"fold_{fold_index:02d}"
        assert (fold_dir / "eligibility_report.csv").exists()
        assert (fold_dir / "leakage_report.csv").exists()
        assert (fold_dir / "excluded_records.parquet").exists()
