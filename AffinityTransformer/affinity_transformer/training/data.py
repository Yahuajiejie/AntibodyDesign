"""Data-path resolution and cache-key collection for training runs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from ..config import Config
from ..dataset import AffinityRecordDataset, filter_trainable_records, load_records
from ..embeddings import collect_embedding_requests
from ..record_filter import filter_records, write_filter_outputs
from ..splits import build_splits, write_splits


def resolve_data_paths(config: Config) -> tuple[Path, Path | None, Path | None]:
    """Return explicit split paths, creating configured automatic splits."""
    if config.data.split_strategy == "none":
        if config.data.train_path is None:
            raise ValueError("data.train_path is required when split_strategy='none'")
        return config.data.train_path, config.data.valid_path, config.data.test_path

    if config.data.all_records_path is None or config.data.split_dir is None:
        raise ValueError("automatic split mode requires all_records_path and split_dir")
    records = load_records(config.data.all_records_path)
    if not config.data.record_filter.is_empty():
        filtered = filter_records(records, config.data.record_filter)
        if filtered.empty:
            raise ValueError("data.filter produced an empty records table")
        write_filter_outputs(
            records,
            filtered,
            config.data.record_filter,
            config.data.split_dir / "filtered_records.parquet",
            config.data.split_dir / "filter_summary.csv",
        )
        records = filtered
    split = build_splits(
        records,
        strategy=config.data.split_strategy,
        valid_fraction=config.data.valid_fraction,
        test_fraction=config.data.test_fraction,
        seed=config.data.seed,
    )
    write_splits(split, config.data.split_dir)
    return (
        config.data.split_dir / "train.parquet",
        config.data.split_dir / "valid.parquet",
        config.data.split_dir / "test.parquet",
    )


def load_trainable_records(path: Path, config: Config) -> pd.DataFrame:
    """Load one processed split, apply `config.data.record_filter`, and
    reject an empty trainable view.

    `resolve_data_paths` only applies `config.data.record_filter` in
    automatic-split mode (it builds `filtered_records.parquet` once, before
    splitting). In explicit-path mode (`split_strategy="none"`, what every
    `configs/v065/*.yaml` uses), nothing previously applied the filter at
    all -- this is the one place every split (train/valid/test) actually
    gets loaded for `frozen_cached` training, so applying it here makes
    `record_filter` work the same way regardless of which mode built the
    split, instead of only working for one of the two.
    """
    records = filter_trainable_records(load_records(path))
    if not config.data.record_filter.is_empty():
        filtered = filter_records(records, config.data.record_filter)
        if filtered.empty:
            raise ValueError(f"data.filter removed every trainable record from {path}")
        records = filtered
    if records.empty:
        raise ValueError(f"No trainable records found in {path}")
    return records


def collect_required_embedding_hashes(
    record_tables: Iterable[pd.DataFrame],
) -> dict[str, list[str]]:
    """Collect unique antibody/antigen cache keys across configured splits."""
    hashes: dict[str, set[str]] = {"antibody": set(), "antigen": set()}
    for records in record_tables:
        dataset = AffinityRecordDataset(records)
        examples = [dataset[index] for index in range(len(dataset))]
        for request in collect_embedding_requests(examples):
            hashes[request.sequence_type].add(request.sequence_hash)
    return {sequence_type: sorted(values) for sequence_type, values in hashes.items()}
