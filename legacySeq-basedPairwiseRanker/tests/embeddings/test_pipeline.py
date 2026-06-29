"""Tests for offline embedding cache generation."""

from pathlib import Path

import torch

from affinity_transformer.dataset import AffinityExample
from affinity_transformer.embeddings import (
    EmbeddingItem,
    ShardedEmbeddingStore,
    collect_embedding_requests,
    write_embedding_cache,
)


class FakeExtractor:
    encoder_name = "fake"
    encoder_revision = "rev-1"

    def encode(self, requests):
        return {
            request.sequence_hash: EmbeddingItem.from_values(
                torch.ones(2, 3, dtype=torch.float32)
            )
            for request in requests
        }

    def metadata(self):
        return {"kind": "fake"}


def _example(record_id: str, antigen_sequence: str | None = "MKT") -> AffinityExample:
    return AffinityExample(
        record_id=record_id,
        dataset_id="study/table",
        heavy_chain="QVQL",
        light_chain="DIQM",
        single_chain_sequence=None,
        antibody_type="Fv",
        antigen_sequence=antigen_sequence,
        antigen_key="ag",
        rank_label=1.0,
        label_kind="experimental",
        group_id="study/table/ag/kd/experimental",
    )


def test_collect_requests_deduplicates_sequences():
    requests = collect_embedding_requests([_example("r1"), _example("r2")])

    assert len(requests) == 2
    assert {request.sequence_type for request in requests} == {"antibody", "antigen"}


def test_write_cache_round_trips_through_sharded_store(tmp_path: Path):
    requests = collect_embedding_requests([_example("r1")])

    manifest_path = write_embedding_cache(requests, FakeExtractor(), tmp_path, shard_size=1)
    store = ShardedEmbeddingStore(manifest_path)

    for request in requests:
        item = store.get(request.sequence_hash, request.sequence_type)
        assert item.values.shape == (2, 3)
    assert (tmp_path / "metadata.yaml").exists()
