"""Tests for embedding-backed rank and pair collators."""

import pytest
import torch

from affinity_transformer.dataloader import (
    EmbeddingBatch,
    PairEmbeddingBatch,
    collate_embedding_batch,
    collate_pair_embedding_batch,
)
from affinity_transformer.dataset import AffinityExample, AffinityPairExample
from affinity_transformer.embeddings import (
    AntibodySequenceInput,
    EmbeddingItem,
    EmbeddingNotFoundError,
    InMemoryEmbeddingStore,
    antibody_sequence_hash,
    antigen_sequence_hash,
)


def _example(**overrides: object) -> AffinityExample:
    values: dict[str, object] = {
        "record_id": "r1",
        "dataset_id": "study/table",
        "heavy_chain": "QVQLVQSG",
        "light_chain": "DIQMTQSP",
        "single_chain_sequence": None,
        "antibody_type": "Fv",
        "antigen_sequence": "MKTAYIAK",
        "antigen_key": "ag",
        "rank_label": 1.0,
        "label_kind": "experimental",
        "group_id": "study/table/ag/kd/experimental",
    }
    values.update(overrides)
    return AffinityExample(**values)  # type: ignore[arg-type]


def _antibody_hash(example: AffinityExample) -> str:
    return antibody_sequence_hash(AntibodySequenceInput(
        heavy_chain=example.heavy_chain,
        light_chain=example.light_chain,
        single_chain_sequence=example.single_chain_sequence,
        antibody_type=example.antibody_type,
    ))


def _stores(examples: list[AffinityExample]):
    antibody_store = InMemoryEmbeddingStore()
    antigen_store = InMemoryEmbeddingStore()
    for index, example in enumerate(examples, start=2):
        antibody_store.put(
            _antibody_hash(example),
            "antibody",
            EmbeddingItem.from_values(torch.full((index, 4), float(index))),
        )
        if example.antigen_sequence is not None:
            antigen_store.put(
                antigen_sequence_hash(example.antigen_sequence),
                "antigen",
                EmbeddingItem.from_values(torch.full((index + 1, 6), float(index))),
            )
    return antibody_store, antigen_store


def test_collate_embedding_batch_pads_different_encoder_dimensions():
    first = _example(record_id="r1")
    second = _example(
        record_id="r2",
        antibody_type="VHH",
        heavy_chain="QVKLEESGGG",
        light_chain=None,
        antigen_sequence=None,
    )
    antibody_store, antigen_store = _stores([first, second])

    batch = collate_embedding_batch([first, second], antibody_store, antigen_store)

    assert isinstance(batch, EmbeddingBatch)
    assert batch.antibody_embeddings.shape == (2, 3, 4)
    assert batch.antibody_mask.tolist() == [[True, True, False], [True, True, True]]
    assert batch.antigen_embeddings.shape == (2, 3, 6)
    assert batch.antigen_mask[0].all()
    assert not batch.antigen_mask[1].any()
    assert batch.record_ids == ["r1", "r2"]


def test_collate_embedding_batch_all_missing_antigen_returns_none():
    example = _example(antigen_sequence=None)
    antibody_store, antigen_store = _stores([example])

    batch = collate_embedding_batch([example], antibody_store, antigen_store)

    assert batch.antigen_embeddings is None
    assert batch.antigen_mask is None


def test_collate_embedding_batch_missing_required_antibody_fails():
    example = _example()

    with pytest.raises(EmbeddingNotFoundError, match="antibody"):
        collate_embedding_batch([example], InMemoryEmbeddingStore())


def test_collate_pair_embedding_batch_splits_sides():
    left = _example(record_id="left", rank_label=2.0)
    right = _example(
        record_id="right",
        heavy_chain="QVKLEESGGG",
        light_chain=None,
        antibody_type="VHH",
        rank_label=1.0,
    )
    antibody_store, antigen_store = _stores([left, right])
    pair = AffinityPairExample(
        pair_id="pair",
        group_id=left.group_id,
        left=left,
        right=right,
        y_ij=1.0,
    )

    batch = collate_pair_embedding_batch([pair], antibody_store, antigen_store)

    assert isinstance(batch, PairEmbeddingBatch)
    assert batch.left.record_ids == ["left"]
    assert batch.right.record_ids == ["right"]
    assert batch.y_ij.tolist() == [1.0]
