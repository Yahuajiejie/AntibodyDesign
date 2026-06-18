"""Integration test for the shared cached-training prerequisites."""

from pathlib import Path

import pandas as pd
import torch
import yaml

from affinity_transformer.config import load_config
from affinity_transformer.dataset import AffinityExample, AffinityPairExample
from affinity_transformer.embeddings import (
    EmbeddingItem,
    ShardedEmbeddingStore,
    collect_embedding_requests,
    collate_embedding_batch,
    collate_pair_embedding_batch,
    validate_embedding_cache,
    write_embedding_cache,
)
from affinity_transformer.model import build_ranker
from affinity_transformer.trainer import Trainer


class _FakeExtractor:
    def __init__(self, name: str, revision: str, tokenizer_revision: str, dim: int) -> None:
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
            "special_tokens_removed": True,
        }


def _example(record_id: str, heavy_chain: str, antigen_sequence: str, label: float):
    return AffinityExample(
        record_id=record_id,
        dataset_id="study/table",
        heavy_chain=heavy_chain,
        light_chain=None,
        single_chain_sequence=None,
        antibody_type="VHH",
        antigen_sequence=antigen_sequence,
        antigen_key="ag",
        rank_label=label,
        label_kind="experimental",
        group_id="group",
    )


def test_config_cache_factory_and_embedding_trainer_form_one_closed_loop(tmp_path: Path):
    left = _example("left", "QVQL", "MKT", 2.0)
    right = _example("right", "EVQL", "MKT", 1.0)
    examples = [left, right]
    requests = collect_embedding_requests(examples)
    antibody_requests = [request for request in requests if request.sequence_type == "antibody"]
    antigen_requests = [request for request in requests if request.sequence_type == "antigen"]
    antibody_cache_dir = tmp_path / "ab-cache"
    antigen_cache_dir = tmp_path / "ag-cache"
    antibody_manifest = write_embedding_cache(
        antibody_requests,
        _FakeExtractor("fake-ab", "ab-rev-1", "ab-tokenizer-1", 5),
        antibody_cache_dir,
    )
    antigen_manifest = write_embedding_cache(
        antigen_requests,
        _FakeExtractor("fake-ag", "ag-rev-1", "ag-tokenizer-1", 7),
        antigen_cache_dir,
    )

    records_path = tmp_path / "records.csv"
    records_path.write_text("placeholder\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "data": {
            "train_path": str(records_path), "valid_path": None,
            "max_pairs_per_group": 10, "seed": 0,
        },
        "model": {
            "antibody_encoder": {
                "name": "fake-ab", "revision": "ab-rev-1",
                "tokenizer_revision": "ab-tokenizer-1", "mode": "frozen_cached",
                "embedding_layer": -1, "cache_dir": str(antibody_cache_dir),
                "max_length": None, "long_sequence_strategy": "error",
            },
            "antigen_encoder": {
                "name": "fake-ag", "revision": "ag-rev-1",
                "tokenizer_revision": "ag-tokenizer-1", "mode": "frozen_cached",
                "embedding_layer": -1, "cache_dir": str(antigen_cache_dir),
                "max_length": None, "long_sequence_strategy": "error",
            },
            "interaction": {
                "kind": "concat", "d_model": 8, "num_layers": 0,
                "num_heads": 2, "ffn_multiplier": 4.0, "dropout": 0.0,
                "pooling": "masked_mean", "bidirectional": True,
            },
            "objective": {
                "name": "pairwise_ranknet", "temperature": 1.0,
                "sigma": 1.0, "pointwise_loss": "huber",
            },
        },
        "train": {"batch_size": 1, "lr": 1.0e-3, "epochs": 1, "device": "cpu"},
    }), encoding="utf-8")
    config = load_config(config_path)

    ab_hashes = [request.sequence_hash for request in antibody_requests]
    ag_hashes = [request.sequence_hash for request in antigen_requests]
    antibody_descriptor = validate_embedding_cache(
        antibody_cache_dir, config.model.antibody_encoder, "antibody", ab_hashes
    )
    assert config.model.antigen_encoder is not None
    antigen_descriptor = validate_embedding_cache(
        antigen_cache_dir, config.model.antigen_encoder, "antigen", ag_hashes
    )
    model = build_ranker(config.model, antibody_descriptor, antigen_descriptor)

    antibody_store = ShardedEmbeddingStore(antibody_manifest)
    antigen_store = ShardedEmbeddingStore(antigen_manifest)
    pair = AffinityPairExample("pair", "group", left, right, 1.0)
    train_batch = collate_pair_embedding_batch([pair], antibody_store, antigen_store)
    valid_batch = collate_embedding_batch(examples, antibody_store, antigen_store)
    metadata = pd.DataFrame([
        {"record_id": "left", "dataset_id": "study/table", "label_kind": "experimental"},
        {"record_id": "right", "dataset_id": "study/table", "label_kind": "experimental"},
    ])
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=[train_batch],
        valid_dataloader=[valid_batch],
        valid_record_metadata=metadata,
        embedding_metadata_hashes={
            "antibody": antibody_descriptor.metadata_hash,
            "antigen": antigen_descriptor.metadata_hash,
        },
    )

    trainer.fit()

    assert trainer.global_step == 1
    assert "valid_weighted_spearman" in trainer.history[-1]
