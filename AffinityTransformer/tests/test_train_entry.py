"""Tests for root train.py entry helpers."""

from __future__ import annotations

import json

import pandas as pd
import pytest
import torch
import yaml

import train
from affinity_transformer.config import load_config
from affinity_transformer.dataset import AffinityRecordDataset, filter_trainable_records
from affinity_transformer.embeddings import (
    EmbeddingItem,
    collect_embedding_requests,
    write_embedding_cache,
)
from affinity_transformer.model import AffinityRanker


class _FakeCacheExtractor:
    def __init__(self, name, revision, tokenizer_revision, dim):
        self.encoder_name = name
        self.encoder_revision = revision
        self.tokenizer_revision = tokenizer_revision
        self.dim = dim

    def encode(self, requests):
        return {
            request.sequence_hash: EmbeddingItem.from_values(
                torch.arange(3 * self.dim, dtype=torch.float32).reshape(3, self.dim)
            )
            for request in requests
        }

    def metadata(self):
        return {
            "embedding_layer": -1,
            "max_length": None,
            "long_sequence_strategy": "error",
        }


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

    monkeypatch.setattr(
        "affinity_transformer.training.online.build_model_and_tokenizers",
        fake_build_model_and_tokenizers,
    )

    output_dir = tmp_path / "out"
    metrics = train.run_training(config_path, output_dir)

    assert (output_dir / "checkpoint.pt").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "predictions.csv").exists()
    assert (output_dir / "test_predictions.csv").exists()
    assert "valid_weighted_spearman" in metrics
    assert "test_weighted_spearman" in metrics


def test_run_training_seeds_before_runner(tmp_path, toy_records, monkeypatch):
    records_path = tmp_path / "records.csv"
    toy_records.to_csv(records_path, index=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "data": {
            "train_path": str(records_path), "valid_path": None,
            "test_path": None, "max_pairs_per_group": 20, "seed": 73,
        },
        "model": {
            "antibody_encoder": "fake", "antigen_encoder": None,
            "d_model": 8, "use_cross_attention": False,
        },
        "train": {"batch_size": 2, "lr": 1e-3, "epochs": 1, "device": "cpu"},
    }), encoding="utf-8")
    samples = []

    def fake_runner(*args):
        samples.append(float(torch.rand(())))
        torch.rand(100)
        return {"sample": samples[-1]}

    monkeypatch.setattr(train, "run_online_training", fake_runner)
    train.run_training(config_path, tmp_path / "out-a")
    train.run_training(config_path, tmp_path / "out-b")

    assert samples[0] == samples[1]


def test_run_training_rejects_unimplemented_objective_before_runner(
    tmp_path, toy_records, monkeypatch
):
    records_path = tmp_path / "records.csv"
    toy_records.to_csv(records_path, index=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "data": {
            "train_path": str(records_path), "valid_path": None,
            "test_path": None, "max_pairs_per_group": 20, "seed": 0,
        },
        "model": {
            "antibody_encoder": {
                "name": "esm2_t12_35M", "revision": "main",
                "tokenizer_revision": "main", "mode": "frozen_online",
                "embedding_layer": -1, "cache_dir": None,
                "max_length": None, "long_sequence_strategy": "error",
            },
            "antigen_encoder": None,
            "interaction": {
                "kind": "antibody_only", "d_model": 480, "num_layers": 0,
                "num_heads": 1, "ffn_multiplier": 4.0, "dropout": 0.0,
                "pooling": "masked_mean", "bidirectional": False,
            },
            "objective": {
                "name": "listwise_listnet", "temperature": 1.0,
                "sigma": 1.0, "pointwise_loss": "huber",
            },
        },
        "train": {"batch_size": 2, "lr": 1e-3, "epochs": 1, "device": "cpu"},
    }), encoding="utf-8")
    called = False

    def fake_runner(*args):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(train, "run_online_training", fake_runner)
    with pytest.raises(NotImplementedError, match="no complete training runner"):
        train.run_training(config_path, tmp_path / "out")
    assert called is False


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
    train_path, valid_path, test_path = train.resolve_data_paths(config)

    assert train_path.exists()
    assert valid_path.exists()
    assert test_path.exists()
    assert (split_dir / "filtered_records.parquet").exists()
    assert (split_dir / "filter_summary.csv").exists()


@pytest.mark.parametrize(
    ("fusion_kind", "num_layers"),
    [("concat", 0), ("deep_cross_attention", 4)],
)
def test_run_training_completes_cached_ranknet_entries(
    tmp_path, toy_records, fusion_kind, num_layers
):
    records = filter_trainable_records(toy_records)
    records_path = tmp_path / "records.csv"
    records.to_csv(records_path, index=False)
    dataset = AffinityRecordDataset(records)
    requests = collect_embedding_requests([dataset[index] for index in range(len(dataset))])
    antibody_requests = [request for request in requests if request.sequence_type == "antibody"]
    antigen_requests = [request for request in requests if request.sequence_type == "antigen"]
    antibody_cache = tmp_path / "antibody-cache"
    antigen_cache = tmp_path / "antigen-cache"
    write_embedding_cache(
        antibody_requests,
        _FakeCacheExtractor("fake-ab", "ab-rev-1", "ab-tokenizer-1", 5),
        antibody_cache,
    )
    write_embedding_cache(
        antigen_requests,
        _FakeCacheExtractor("fake-ag", "ag-rev-1", "ag-tokenizer-1", 7),
        antigen_cache,
    )
    config_path = tmp_path / f"cached-{fusion_kind}.yaml"
    config_path.write_text(yaml.safe_dump({
        "data": {
            "train_path": str(records_path),
            "valid_path": str(records_path),
            "test_path": str(records_path),
            "max_pairs_per_group": 50,
            "seed": 0,
        },
        "model": {
            "antibody_encoder": {
                "name": "fake-ab", "revision": "ab-rev-1",
                "tokenizer_revision": "ab-tokenizer-1", "mode": "frozen_cached",
                "embedding_layer": -1, "cache_dir": str(antibody_cache),
                "max_length": None, "long_sequence_strategy": "error",
            },
            "antigen_encoder": {
                "name": "fake-ag", "revision": "ag-rev-1",
                "tokenizer_revision": "ag-tokenizer-1", "mode": "frozen_cached",
                "embedding_layer": -1, "cache_dir": str(antigen_cache),
                "max_length": None, "long_sequence_strategy": "error",
            },
            "interaction": {
                "kind": fusion_kind, "d_model": 8, "num_layers": num_layers,
                "num_heads": 2, "ffn_multiplier": 4.0, "dropout": 0.0,
                "pooling": "masked_mean", "bidirectional": True,
            },
            "objective": {
                "name": "pairwise_ranknet", "temperature": 1.0,
                "sigma": 1.0, "pointwise_loss": "huber",
            },
        },
        "train": {"batch_size": 4, "lr": 1.0e-3, "epochs": 1, "device": "cpu"},
    }), encoding="utf-8")

    output_dir = tmp_path / f"cached-{fusion_kind}-output"
    metrics = train.run_training(config_path, output_dir)

    assert (output_dir / "checkpoint.pt").exists()
    assert (output_dir / "predictions.csv").exists()
    assert (output_dir / "test_predictions.csv").exists()
    assert (output_dir / "embedding_metadata_refs.yaml").exists()
    assert (output_dir / "resource_metrics.json").exists()
    assert "valid_weighted_spearman" in metrics
    checkpoint = torch.load(output_dir / "checkpoint.pt", weights_only=False)
    assert set(checkpoint["embedding_metadata_hashes"]) == {"antibody", "antigen"}
    metadata_refs = yaml.safe_load(
        (output_dir / "embedding_metadata_refs.yaml").read_text(encoding="utf-8")
    )
    assert metadata_refs["antibody"]["embedding_dim"] == 5
    assert metadata_refs["antigen"]["embedding_dim"] == 7
    assert metadata_refs["antibody"]["coverage"] == 1.0
    predictions = pd.read_csv(output_dir / "predictions.csv")
    assert len(predictions) == len(records)
    assert predictions["record_id"].is_unique
    resources = json.loads(
        (output_dir / "resource_metrics.json").read_text(encoding="utf-8")
    )
    assert resources["fusion_kind"] == fusion_kind
    assert resources["interaction_num_layers"] == num_layers
    assert resources["trainable_parameter_count"] > 0
    assert resources["pair_examples_per_second"] > 0
    assert set(resources["antibody"]["sequence_length"]) == {"p50", "p90", "p95", "max"}
    if fusion_kind == "deep_cross_attention":
        assert resources["attention_cell_upper_bound_per_batch"] > 0
    else:
        assert resources["attention_cell_upper_bound_per_batch"] == 0
