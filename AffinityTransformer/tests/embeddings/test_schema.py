"""Tests for embedding cache keys and item validation."""

import pytest
import torch

from affinity_transformer.embeddings import (
    AntibodySequenceInput,
    EmbeddingItem,
    antibody_sequence_hash,
    antigen_sequence_hash,
)


def test_antibody_sequence_hash_is_stable_and_structured():
    sequence = AntibodySequenceInput(
        heavy_chain="QVQLVQSG",
        light_chain="DIQMTQSP",
        single_chain_sequence=None,
        antibody_type="Fv",
    )

    assert antibody_sequence_hash(sequence) == antibody_sequence_hash(sequence)
    changed = AntibodySequenceInput(
        heavy_chain=sequence.heavy_chain,
        light_chain=sequence.light_chain,
        single_chain_sequence=None,
        antibody_type="IgG",
    )
    assert antibody_sequence_hash(sequence) != antibody_sequence_hash(changed)


def test_antigen_sequence_hash_rejects_empty_sequence():
    with pytest.raises(ValueError, match="non-empty"):
        antigen_sequence_hash("")


def test_embedding_item_builds_default_mask_and_rejects_nan():
    item = EmbeddingItem.from_values(torch.ones(3, 5))
    assert item.mask.tolist() == [True, True, True]

    invalid = torch.ones(2, 3)
    invalid[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        EmbeddingItem.from_values(invalid)
