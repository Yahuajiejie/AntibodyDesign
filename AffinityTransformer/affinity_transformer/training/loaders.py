"""Objective-specific online and embedding-backed DataLoader builders."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from ..config import Config
from ..dataloader import Tokenizer, collate_pair_batch, collate_rank_batch
from ..dataset import (
    AffinityRecordDataset,
    PairwiseAffinityDataset,
    build_pairs,
    filter_trainable_records,
    load_records,
)
from ..embeddings import (
    EmbeddingStore,
    collate_embedding_batch,
    collate_pair_embedding_batch,
)


def build_online_train_loader(
    path: Path,
    config: Config,
    antibody_tokenizer: Tokenizer,
    antigen_tokenizer: Tokenizer | None,
) -> tuple[pd.DataFrame, DataLoader]:
    records = filter_trainable_records(load_records(path))
    pairs = _build_pairs(records, config)
    if pairs.empty:
        raise ValueError(f"No trainable pairs could be built from {path}")
    nw = config.train.num_workers
    loader = DataLoader(
        PairwiseAffinityDataset(records, pairs),
        batch_size=config.train.batch_size,
        shuffle=False,
        collate_fn=partial(
            collate_pair_batch,
            antibody_tokenizer=antibody_tokenizer,
            antigen_tokenizer=antigen_tokenizer,
        ),
        num_workers=nw,
        pin_memory=config.train.pin_memory,
        persistent_workers=nw > 0,
    )
    return records, loader


def build_online_rank_loader(
    path: Path | None,
    config: Config,
    antibody_tokenizer: Tokenizer,
    antigen_tokenizer: Tokenizer | None,
) -> tuple[pd.DataFrame | None, DataLoader | None]:
    if path is None:
        return None, None
    records = filter_trainable_records(load_records(path))
    nw = config.train.num_workers
    loader = DataLoader(
        AffinityRecordDataset(records),
        batch_size=config.train.batch_size,
        shuffle=False,
        collate_fn=partial(
            collate_rank_batch,
            antibody_tokenizer=antibody_tokenizer,
            antigen_tokenizer=antigen_tokenizer,
        ),
        num_workers=nw,
        pin_memory=config.train.pin_memory,
        persistent_workers=nw > 0,
    )
    return records, loader


def build_cached_train_loader(
    records: pd.DataFrame,
    config: Config,
    antibody_store: EmbeddingStore,
    antigen_store: EmbeddingStore,
) -> DataLoader:
    pairs = _build_pairs(records, config)
    if pairs.empty:
        raise ValueError("No trainable pairs could be built for frozen_cached training")
    nw = config.train.num_workers
    return DataLoader(
        PairwiseAffinityDataset(records, pairs),
        batch_size=config.train.batch_size,
        shuffle=False,
        collate_fn=partial(
            collate_pair_embedding_batch,
            antibody_store=antibody_store,
            antigen_store=antigen_store,
        ),
        num_workers=nw,
        pin_memory=config.train.pin_memory,
        persistent_workers=nw > 0,
    )


def build_cached_rank_loader(
    records: pd.DataFrame | None,
    config: Config,
    antibody_store: EmbeddingStore,
    antigen_store: EmbeddingStore,
) -> DataLoader | None:
    if records is None:
        return None
    nw = config.train.num_workers
    return DataLoader(
        AffinityRecordDataset(records),
        batch_size=config.train.batch_size,
        shuffle=False,
        collate_fn=partial(
            collate_embedding_batch,
            antibody_store=antibody_store,
            antigen_store=antigen_store,
        ),
        num_workers=nw,
        pin_memory=config.train.pin_memory,
        persistent_workers=nw > 0,
    )


def _build_pairs(records: pd.DataFrame, config: Config) -> pd.DataFrame:
    return build_pairs(
        records,
        max_pairs_per_group=config.data.max_pairs_per_group,
        seed=config.data.seed,
        pair_sample_strategy=config.data.pair_sample_strategy,
        pair_fraction=config.data.pair_fraction,
        min_pairs_per_group=config.data.min_pairs_per_group,
        large_group_threshold=config.data.large_group_threshold,
        pair_enumeration_limit=config.data.pair_enumeration_limit,
        label_block_count=config.data.label_block_count,
        intra_block_pairs_per_large_group=config.data.intra_block_pairs_per_large_group,
        discrete_label_unique_threshold=config.data.discrete_label_unique_threshold,
        discrete_label_ratio_threshold=config.data.discrete_label_ratio_threshold,
    )
