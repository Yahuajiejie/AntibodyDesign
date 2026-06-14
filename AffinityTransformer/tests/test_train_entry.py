"""Tests for root train.py entry helpers."""

from __future__ import annotations

import yaml

import train
from affinity_transformer.config import load_config
from affinity_transformer.model import AffinityRanker


def test_run_training_writes_checkpoint_and_metrics(
    tmp_path,
    toy_records,
    antibody_tokenizer,
    make_fake_encoder,
    monkeypatch,
):
    records_path = tmp_path / "records.csv"
    toy_records.to_csv(records_path, index=False)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "data": {
            "train_path": str(records_path),
            "valid_path": str(records_path),
            "test_path": str(records_path),
            "max_pairs_per_group": 50,
            "seed": 0,
        },
        "model": {
            "antibody_encoder": "fake",
            "antigen_encoder": None,
            "d_model": 16,
            "use_cross_attention": False,
        },
        "train": {
            "batch_size": 4,
            "lr": 1.0e-3,
            "epochs": 1,
            "device": "cpu",
        },
    }))

    def fake_build_model_and_tokenizers(model_config):
        model = AffinityRanker(
            antibody_encoder=make_fake_encoder(model_config.d_model),
            antigen_encoder=None,
            d_model=model_config.d_model,
            use_cross_attention=model_config.use_cross_attention,
        )
        return model, antibody_tokenizer, None

    monkeypatch.setattr(train, "build_model_and_tokenizers", fake_build_model_and_tokenizers)

    output_dir = tmp_path / "out"
    metrics = train.run_training(config_path, output_dir)

    assert (output_dir / "checkpoint.pt").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "predictions.csv").exists()
    assert (output_dir / "test_predictions.csv").exists()
    assert "valid_weighted_spearman" in metrics
    assert "test_weighted_spearman" in metrics


def test_resolve_data_paths_applies_record_filter_before_split(tmp_path, toy_records):
    records_path = tmp_path / "all_records.csv"
    toy_records.to_csv(records_path, index=False)
    split_dir = tmp_path / "splits"

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "data": {
            "all_records_path": str(records_path),
            "train_path": None,
            "valid_path": None,
            "test_path": None,
            "split_strategy": "group_holdout_split",
            "split_dir": str(split_dir),
            "valid_fraction": 0.2,
            "test_fraction": 0.2,
            "max_pairs_per_group": 50,
            "seed": 0,
            "filter": {
                "include_dataset_ids": ["studyA/tableA", "studyB/tableB", "studyD/tableD"],
                "min_trainable_records_per_group": 2,
                "min_unique_labels_per_group": 2,
            },
        },
        "model": {
            "antibody_encoder": "fake",
            "antigen_encoder": None,
            "d_model": 16,
            "use_cross_attention": False,
        },
        "train": {
            "batch_size": 4,
            "lr": 1.0e-3,
            "epochs": 1,
            "device": "cpu",
        },
    }))

    config = load_config(config_path)
    train_path, valid_path, test_path = train._resolve_data_paths(config)

    assert train_path.exists()
    assert valid_path.exists()
    assert test_path.exists()
    assert (split_dir / "filtered_records.parquet").exists()
    assert (split_dir / "filter_summary.csv").exists()
