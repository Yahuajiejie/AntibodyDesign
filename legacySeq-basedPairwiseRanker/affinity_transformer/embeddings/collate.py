"""Embedding-backed batches and collators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from ..dataset import AffinityExample, AffinityPairExample
from .schema import (
    AntibodySequenceInput,
    EmbeddingItem,
    antibody_embedding_request,
    antigen_embedding_request,
)
from .store import EmbeddingStore


@dataclass
class EmbeddingBatch:
    """One padded record batch backed by cached token embeddings."""

    antibody_embeddings: torch.Tensor
    antibody_mask: torch.Tensor
    antigen_embeddings: torch.Tensor | None
    antigen_mask: torch.Tensor | None
    labels: torch.Tensor
    record_ids: list[str]
    group_ids: list[str]


@dataclass
class PairEmbeddingBatch:
    """Left/right embedding batches and their RankNet targets."""

    left: EmbeddingBatch
    right: EmbeddingBatch
    y_ij: torch.Tensor


def collate_embedding_batch(
    examples: Sequence[AffinityExample],
    antibody_store: EmbeddingStore,
    antigen_store: EmbeddingStore | None = None,
) -> EmbeddingBatch:
    """Load and pad cached embeddings for record examples."""
    if not examples:
        raise ValueError("collate_embedding_batch requires at least one example")

    antibody_items = [
        antibody_store.get(_antibody_request(example).sequence_hash, "antibody")
        for example in examples
    ]
    antibody_embeddings, antibody_mask = _pad_items(antibody_items)
    antigen_embeddings, antigen_mask = _collate_antigens(examples, antigen_store)

    return EmbeddingBatch(
        antibody_embeddings=antibody_embeddings,
        antibody_mask=antibody_mask,
        antigen_embeddings=antigen_embeddings,
        antigen_mask=antigen_mask,
        labels=torch.tensor([example.rank_label for example in examples], dtype=torch.float32),
        record_ids=[example.record_id for example in examples],
        group_ids=[example.group_id for example in examples],
    )


def collate_pair_embedding_batch(
    examples: Sequence[AffinityPairExample],
    antibody_store: EmbeddingStore,
    antigen_store: EmbeddingStore | None = None,
) -> PairEmbeddingBatch:
    """Load and independently pad both sides of pairwise examples."""
    if not examples:
        raise ValueError("collate_pair_embedding_batch requires at least one example")
    left = collate_embedding_batch(
        [example.left for example in examples], antibody_store, antigen_store
    )
    right = collate_embedding_batch(
        [example.right for example in examples], antibody_store, antigen_store
    )
    return PairEmbeddingBatch(
        left=left,
        right=right,
        y_ij=torch.tensor([example.y_ij for example in examples], dtype=torch.float32),
    )


def _antibody_request(example: AffinityExample):
    sequence = AntibodySequenceInput(
        heavy_chain=example.heavy_chain,
        light_chain=example.light_chain,
        single_chain_sequence=example.single_chain_sequence,
        antibody_type=example.antibody_type,
    )
    return antibody_embedding_request(sequence)


def _collate_antigens(
    examples: Sequence[AffinityExample],
    store: EmbeddingStore | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if store is None or all(example.antigen_sequence is None for example in examples):
        return None, None

    items: list[EmbeddingItem | None] = []
    for example in examples:
        if example.antigen_sequence is None:
            items.append(None)
            continue
        request = antigen_embedding_request(example.antigen_sequence)
        items.append(store.get(request.sequence_hash, "antigen"))
    return _pad_optional_items(items)


def _pad_items(items: Sequence[EmbeddingItem]) -> tuple[torch.Tensor, torch.Tensor]:
    if not items:
        raise ValueError("cannot pad an empty embedding item sequence")
    embedding_dim = items[0].values.shape[1]
    dtype = items[0].values.dtype
    for item in items:
        if item.values.shape[1] != embedding_dim:
            raise ValueError("embedding items in one store must share embedding_dim")
        if item.values.dtype != dtype:
            raise ValueError("embedding items in one store must share dtype")

    max_length = max(item.values.shape[0] for item in items)
    values = torch.zeros(len(items), max_length, embedding_dim, dtype=dtype)
    mask = torch.zeros(len(items), max_length, dtype=torch.bool)
    for index, item in enumerate(items):
        length = item.values.shape[0]
        values[index, :length] = item.values
        mask[index, :length] = item.mask
    return values, mask


def _pad_optional_items(
    items: Sequence[EmbeddingItem | None],
) -> tuple[torch.Tensor, torch.Tensor]:
    present = [item for item in items if item is not None]
    if not present:
        raise ValueError("optional embedding items must contain at least one present item")
    present_values, _ = _pad_items(present)
    max_length = present_values.shape[1]
    embedding_dim = present_values.shape[2]
    dtype = present_values.dtype
    values = torch.zeros(len(items), max_length, embedding_dim, dtype=dtype)
    mask = torch.zeros(len(items), max_length, dtype=torch.bool)
    for index, item in enumerate(items):
        if item is None:
            continue
        if item.values.shape[1] != embedding_dim or item.values.dtype != dtype:
            raise ValueError("embedding items in one store must share shape and dtype")
        length = item.values.shape[0]
        values[index, :length] = item.values
        mask[index, :length] = item.mask
    return values, mask
