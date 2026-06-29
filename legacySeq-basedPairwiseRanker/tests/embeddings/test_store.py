"""Tests for in-memory and sharded embedding stores."""

from pathlib import Path

import pandas as pd
import pytest
import torch

from affinity_transformer.embeddings import (
    EmbeddingItem,
    EmbeddingNotFoundError,
    InMemoryEmbeddingStore,
    ShardedEmbeddingStore,
)


def test_in_memory_store_raises_contextual_missing_error():
    store = InMemoryEmbeddingStore()

    with pytest.raises(EmbeddingNotFoundError, match="antibody"):
        store.get("missing", "antibody")


def test_sharded_store_reads_manifest_item(tmp_path: Path):
    shard_path = tmp_path / "shard_00000.pt"
    torch.save({"item-1": {"values": torch.arange(12, dtype=torch.float32).reshape(3, 4)}}, shard_path)
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame([
        {
            "sequence_hash": "hash-1",
            "sequence_type": "antigen",
            "encoder_name": "fake",
            "encoder_revision": "rev-1",
            "shard_path": shard_path.name,
            "item_key": "item-1",
            "sequence_length": 3,
            "embedding_length": 3,
            "embedding_dim": 4,
            "dtype": "float32",
        }
    ]).to_csv(manifest_path, index=False)

    item = ShardedEmbeddingStore(manifest_path).get("hash-1", "antigen")

    assert item.values.shape == (3, 4)
    assert item.mask.all()


def test_sharded_store_rejects_manifest_shape_mismatch(tmp_path: Path):
    torch.save({"item": torch.ones(2, 4)}, tmp_path / "shard.pt")
    manifest = pd.DataFrame([
        {
            "sequence_hash": "hash",
            "sequence_type": "antibody",
            "encoder_name": "fake",
            "encoder_revision": "rev",
            "shard_path": "shard.pt",
            "item_key": "item",
            "sequence_length": 2,
            "embedding_length": 3,
            "embedding_dim": 4,
            "dtype": "float32",
        }
    ])
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    # Shape validation now happens at __init__ (eager preload), not at get().
    with pytest.raises(ValueError, match="shape mismatch"):
        ShardedEmbeddingStore(manifest_path)
