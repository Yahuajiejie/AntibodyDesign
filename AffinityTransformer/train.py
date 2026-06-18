#!/usr/bin/env python3
"""Training entry point for AffinityTransformer."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from functools import partial
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from affinity_transformer.config import Config, load_config
from affinity_transformer.dataloader import Tokenizer, collate_pair_batch, collate_rank_batch
from affinity_transformer.dataset import (
    AffinityRecordDataset,
    PairwiseAffinityDataset,
    build_pairs,
    filter_trainable_records,
    load_records,
)
from affinity_transformer.embeddings import (
    CacheDescriptor,
    EmbeddingBatch,
    EmbeddingStore,
    ShardedEmbeddingStore,
    collect_embedding_requests,
    collate_embedding_batch,
    collate_pair_embedding_batch,
    validate_embedding_cache,
)
from affinity_transformer.metrics import compute_group_spearman, summarize_group_spearman
from affinity_transformer.model import build_ranker
from affinity_transformer.record_filter import filter_records, write_filter_outputs
from affinity_transformer.splits import build_splits, write_splits
from affinity_transformer.trainer import Trainer, build_model_and_tokenizers
from affinity_transformer.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or _default_output_dir(args.config)
    run_training(args.config, output_dir)
    print(f"wrote run outputs -> {output_dir}")


def run_training(config_path: Path, output_dir: Path) -> dict[str, float]:
    """Run one configured training job and write its output files."""
    config = load_config(config_path)
    output_dir = ensure_dir(output_dir)
    train_path, valid_path, test_path = _resolve_data_paths(config)

    if config.model.antibody_encoder.mode == "frozen_cached":
        return _run_cached_ranknet(
            config_path,
            config,
            output_dir,
            train_path,
            valid_path,
            test_path,
        )

    return _run_online_training(
        config_path,
        config,
        output_dir,
        train_path,
        valid_path,
        test_path,
    )


def _run_online_training(
    config_path: Path,
    config: Config,
    output_dir: Path,
    train_path: Path,
    valid_path: Path | None,
    test_path: Path | None,
) -> dict[str, float]:
    """Preserve the explicit legacy online-token training path."""

    model, antibody_tokenizer, antigen_tokenizer = build_model_and_tokenizers(config.model)
    train_records, train_loader = _build_train_loader(
        train_path, config, antibody_tokenizer, antigen_tokenizer
    )
    valid_records, valid_loader = _build_rank_loader(
        valid_path, config, antibody_tokenizer, antigen_tokenizer
    )

    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        valid_dataloader=valid_loader,
        valid_record_metadata=None if valid_records is None else _metadata(valid_records),
        output_dir=output_dir,
    )
    trainer.fit()
    trainer.save_checkpoint(output_dir / "checkpoint.pt")

    metrics: dict[str, float] = {}
    if trainer.history:
        metrics.update(trainer.history[-1])

    if valid_records is not None:
        valid_predictions = _predict_records(
            trainer.model, valid_records, antibody_tokenizer, antigen_tokenizer, config.train.batch_size
        )
        valid_predictions["split"] = "valid"
        valid_group_metrics = compute_group_spearman(valid_predictions)
        valid_group_metrics.to_csv(output_dir / "group_metrics.csv", index=False)
        valid_predictions.to_csv(output_dir / "predictions.csv", index=False)
        metrics.update(_flatten_summary(summarize_group_spearman(valid_group_metrics), "valid"))

    if test_path is not None:
        test_records, _ = _build_rank_loader(test_path, config, antibody_tokenizer, antigen_tokenizer)
        if test_records is not None:
            test_predictions = _predict_records(
                trainer.model, test_records, antibody_tokenizer, antigen_tokenizer, config.train.batch_size
            )
            test_predictions["split"] = "test"
            test_group_metrics = compute_group_spearman(test_predictions)
            test_group_metrics.to_csv(output_dir / "test_group_metrics.csv", index=False)
            test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)
            metrics.update(_flatten_summary(summarize_group_spearman(test_group_metrics), "test"))

    _write_json(output_dir / "metrics.json", metrics)
    shutil.copyfile(config_path, output_dir / "config.yaml")
    (output_dir / "run.log").write_text(
        f"config={config_path}\ntrain_path={train_path}\nvalid_path={valid_path}\n"
        f"test_path={test_path}\ntrain_records={len(train_records)}\n",
        encoding="utf-8",
    )
    return metrics


def _run_cached_ranknet(
    config_path: Path,
    config: Config,
    output_dir: Path,
    train_path: Path,
    valid_path: Path | None,
    test_path: Path | None,
) -> dict[str, float]:
    """Run frozen-cache Concat/Deep-Cross-Attention with RankNet."""
    interaction_kind = config.model.interaction.kind
    if interaction_kind not in {"concat", "deep_cross_attention"}:
        raise ValueError(
            "the frozen_cached entry currently supports interaction.kind in "
            f"{{'concat', 'deep_cross_attention'}}; got {interaction_kind!r}"
        )
    if config.model.objective.name != "pairwise_ranknet":
        raise ValueError(
            "the frozen_cached entry currently supports objective.name="
            f"'pairwise_ranknet'; got {config.model.objective.name!r}"
        )
    antigen_config = config.model.antigen_encoder
    if antigen_config is None:
        raise ValueError(f"{interaction_kind} frozen_cached training requires model.antigen_encoder")

    train_records = _load_trainable_records(train_path)
    valid_records = None if valid_path is None else _load_trainable_records(valid_path)
    test_records = None if test_path is None else _load_trainable_records(test_path)
    all_records = [
        records
        for records in (train_records, valid_records, test_records)
        if records is not None
    ]
    required_hashes = _collect_required_embedding_hashes(all_records)

    antibody_config = config.model.antibody_encoder
    assert antibody_config.cache_dir is not None
    assert antigen_config.cache_dir is not None
    antibody_descriptor = validate_embedding_cache(
        antibody_config.cache_dir,
        antibody_config,
        "antibody",
        required_hashes["antibody"],
    )
    antigen_descriptor = validate_embedding_cache(
        antigen_config.cache_dir,
        antigen_config,
        "antigen",
        required_hashes["antigen"],
    )

    # Cache validation deliberately precedes ranker/Trainer construction;
    # Trainer.__init__ creates the optimizer.
    model = build_ranker(config.model, antibody_descriptor, antigen_descriptor)
    antibody_store = ShardedEmbeddingStore(antibody_descriptor.manifest_path)
    antigen_store = ShardedEmbeddingStore(antigen_descriptor.manifest_path)
    train_loader = _build_cached_train_loader(
        train_records,
        config,
        antibody_store,
        antigen_store,
    )
    valid_loader = _build_cached_rank_loader(
        valid_records,
        config,
        antibody_store,
        antigen_store,
    )
    metadata_hashes = {
        "antibody": antibody_descriptor.metadata_hash,
        "antigen": antigen_descriptor.metadata_hash,
    }
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        valid_dataloader=valid_loader,
        valid_record_metadata=(
            None if valid_records is None else _metadata(valid_records)
        ),
        output_dir=output_dir,
        embedding_metadata_hashes=metadata_hashes,
    )
    uses_cuda = trainer.device.type == "cuda" and torch.cuda.is_available()
    if uses_cuda:
        torch.cuda.reset_peak_memory_stats(trainer.device)
        torch.cuda.synchronize(trainer.device)
    training_started = time.perf_counter()
    trainer.fit()
    if uses_cuda:
        torch.cuda.synchronize(trainer.device)
    training_seconds = time.perf_counter() - training_started
    trainer.save_checkpoint(output_dir / "checkpoint.pt")

    metrics: dict[str, float] = {}
    if trainer.history:
        metrics.update(trainer.history[-1])

    if valid_records is not None:
        valid_predictions = _predict_cached_records(
            trainer.model,
            valid_records,
            antibody_store,
            antigen_store,
            config.train.batch_size,
        )
        valid_predictions["split"] = "valid"
        valid_group_metrics = compute_group_spearman(valid_predictions)
        valid_group_metrics.to_csv(output_dir / "group_metrics.csv", index=False)
        valid_predictions.to_csv(output_dir / "predictions.csv", index=False)
        metrics.update(_flatten_summary(summarize_group_spearman(valid_group_metrics), "valid"))

    if test_records is not None:
        test_predictions = _predict_cached_records(
            trainer.model,
            test_records,
            antibody_store,
            antigen_store,
            config.train.batch_size,
        )
        test_predictions["split"] = "test"
        test_group_metrics = compute_group_spearman(test_predictions)
        test_group_metrics.to_csv(output_dir / "test_group_metrics.csv", index=False)
        test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)
        metrics.update(_flatten_summary(summarize_group_spearman(test_group_metrics), "test"))

    _write_json(output_dir / "metrics.json", metrics)
    shutil.copyfile(config_path, output_dir / "config.yaml")
    _write_embedding_metadata_refs(
        output_dir / "embedding_metadata_refs.yaml",
        antibody_descriptor,
        antigen_descriptor,
    )
    _write_resource_metrics(
        output_dir / "resource_metrics.json",
        trainer,
        train_loader,
        antibody_descriptor,
        antigen_descriptor,
        training_seconds,
    )
    (output_dir / "run.log").write_text(
        f"mode=frozen_cached\nfusion_kind={interaction_kind}\nconfig={config_path}\n"
        f"train_path={train_path}\n"
        f"valid_path={valid_path}\ntest_path={test_path}\n"
        f"train_records={len(train_records)}\n",
        encoding="utf-8",
    )
    return metrics


def _resolve_data_paths(config: Config) -> tuple[Path, Path | None, Path | None]:
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


def _build_train_loader(
    path: Path,
    config: Config,
    antibody_tokenizer: Tokenizer,
    antigen_tokenizer: Tokenizer | None,
) -> tuple[pd.DataFrame, DataLoader]:
    records = filter_trainable_records(load_records(path))
    pairs = build_pairs(
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
    if pairs.empty:
        raise ValueError(f"No trainable pairs could be built from {path}")
    dataset = PairwiseAffinityDataset(records, pairs)
    loader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        collate_fn=partial(
            collate_pair_batch,
            antibody_tokenizer=antibody_tokenizer,
            antigen_tokenizer=antigen_tokenizer,
        ),
    )
    return records, loader


def _build_rank_loader(
    path: Path | None,
    config: Config,
    antibody_tokenizer: Tokenizer,
    antigen_tokenizer: Tokenizer | None,
) -> tuple[pd.DataFrame | None, DataLoader | None]:
    if path is None:
        return None, None
    records = filter_trainable_records(load_records(path))
    dataset = AffinityRecordDataset(records)
    loader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        collate_fn=partial(
            collate_rank_batch,
            antibody_tokenizer=antibody_tokenizer,
            antigen_tokenizer=antigen_tokenizer,
        ),
    )
    return records, loader


def _load_trainable_records(path: Path) -> pd.DataFrame:
    records = filter_trainable_records(load_records(path))
    if records.empty:
        raise ValueError(f"No trainable records found in {path}")
    return records


def _collect_required_embedding_hashes(
    record_tables: Iterable[pd.DataFrame],
) -> dict[str, list[str]]:
    """Collect unique cache keys across every configured data split."""
    hashes: dict[str, set[str]] = {"antibody": set(), "antigen": set()}
    for records in record_tables:
        dataset = AffinityRecordDataset(records)
        examples = [dataset[index] for index in range(len(dataset))]
        for request in collect_embedding_requests(examples):
            hashes[request.sequence_type].add(request.sequence_hash)
    return {sequence_type: sorted(values) for sequence_type, values in hashes.items()}


def _build_cached_train_loader(
    records: pd.DataFrame,
    config: Config,
    antibody_store: EmbeddingStore,
    antigen_store: EmbeddingStore,
) -> DataLoader:
    pairs = build_pairs(
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
    if pairs.empty:
        raise ValueError("No trainable pairs could be built for frozen_cached training")
    return DataLoader(
        PairwiseAffinityDataset(records, pairs),
        batch_size=config.train.batch_size,
        shuffle=False,
        collate_fn=partial(
            collate_pair_embedding_batch,
            antibody_store=antibody_store,
            antigen_store=antigen_store,
        ),
    )


def _build_cached_rank_loader(
    records: pd.DataFrame | None,
    config: Config,
    antibody_store: EmbeddingStore,
    antigen_store: EmbeddingStore,
) -> DataLoader | None:
    if records is None:
        return None
    return DataLoader(
        AffinityRecordDataset(records),
        batch_size=config.train.batch_size,
        shuffle=False,
        collate_fn=partial(
            collate_embedding_batch,
            antibody_store=antibody_store,
            antigen_store=antigen_store,
        ),
    )


def _predict_records(
    model,
    records: pd.DataFrame,
    antibody_tokenizer: Tokenizer,
    antigen_tokenizer: Tokenizer | None,
    batch_size: int,
) -> pd.DataFrame:
    dataset = AffinityRecordDataset(records)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(
            collate_rank_batch,
            antibody_tokenizer=antibody_tokenizer,
            antigen_tokenizer=antigen_tokenizer,
        ),
    )
    device = next(model.parameters()).device
    rows: list[dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            moved = type(batch)(
                antibody_tokens=batch.antibody_tokens.to(device),
                antibody_mask=batch.antibody_mask.to(device),
                antigen_tokens=None if batch.antigen_tokens is None else batch.antigen_tokens.to(device),
                antigen_mask=None if batch.antigen_mask is None else batch.antigen_mask.to(device),
                labels=batch.labels.to(device),
                record_ids=batch.record_ids,
                group_ids=batch.group_ids,
            )
            scores = model(moved)
            for record_id, group_id, label, score in zip(
                batch.record_ids, batch.group_ids, batch.labels.tolist(), scores.detach().cpu().tolist()
            ):
                rows.append({
                    "record_id": record_id,
                    "group_id": group_id,
                    "rank_label": label,
                    "score": score,
                })
    return pd.DataFrame(rows).merge(_metadata(records), on="record_id", how="left", validate="many_to_one")


def _predict_cached_records(
    model,
    records: pd.DataFrame,
    antibody_store: EmbeddingStore,
    antigen_store: EmbeddingStore,
    batch_size: int,
) -> pd.DataFrame:
    loader = DataLoader(
        AffinityRecordDataset(records),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(
            collate_embedding_batch,
            antibody_store=antibody_store,
            antigen_store=antigen_store,
        ),
    )
    device = next(model.parameters()).device
    rows: list[dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            moved = EmbeddingBatch(
                antibody_embeddings=batch.antibody_embeddings.to(device),
                antibody_mask=batch.antibody_mask.to(device),
                antigen_embeddings=(
                    None
                    if batch.antigen_embeddings is None
                    else batch.antigen_embeddings.to(device)
                ),
                antigen_mask=(
                    None if batch.antigen_mask is None else batch.antigen_mask.to(device)
                ),
                labels=batch.labels.to(device),
                record_ids=batch.record_ids,
                group_ids=batch.group_ids,
            )
            scores = model(moved)
            for record_id, group_id, label, score in zip(
                batch.record_ids,
                batch.group_ids,
                batch.labels.tolist(),
                scores.detach().cpu().tolist(),
            ):
                rows.append({
                    "record_id": record_id,
                    "group_id": group_id,
                    "rank_label": label,
                    "score": score,
                })
    return pd.DataFrame(rows).merge(
        _metadata(records),
        on="record_id",
        how="left",
        validate="many_to_one",
    )


def _metadata(records: pd.DataFrame) -> pd.DataFrame:
    return records[["record_id", "dataset_id", "label_kind"]].copy()


def _flatten_summary(summary: dict[str, dict[str, float | int]], prefix: str) -> dict[str, float]:
    flat: dict[str, float] = {}
    overall = summary["overall"]
    flat[f"{prefix}_macro_spearman"] = float(overall["macro_spearman"])
    flat[f"{prefix}_weighted_spearman"] = float(overall["weighted_spearman"])
    flat[f"{prefix}_n_valid_groups"] = float(overall["n_valid_groups"])
    flat[f"{prefix}_n_skipped_groups"] = float(overall["n_skipped_groups"])
    return flat


def _write_json(path: Path, payload: dict[str, float]) -> None:
    def clean(value: float) -> float | None:
        return None if isinstance(value, float) and math.isnan(value) else value

    with path.open("w", encoding="utf-8") as handle:
        json.dump({key: clean(value) for key, value in payload.items()}, handle, indent=2)


def _write_embedding_metadata_refs(
    path: Path,
    antibody: CacheDescriptor,
    antigen: CacheDescriptor,
) -> None:
    def serialize(descriptor: CacheDescriptor) -> dict[str, object]:
        return {
            "cache_dir": str(descriptor.cache_dir),
            "manifest_path": str(descriptor.manifest_path),
            "metadata_path": str(descriptor.metadata_path),
            "encoder_name": descriptor.encoder_name,
            "encoder_revision": descriptor.encoder_revision,
            "tokenizer_revision": descriptor.tokenizer_revision,
            "embedding_dim": descriptor.embedding_dim,
            "dtype": descriptor.dtype,
            "metadata_hash": descriptor.metadata_hash,
            "n_items": descriptor.n_items,
            "required_count": descriptor.required_count,
            "coverage": descriptor.coverage,
            "sequence_length": dict(descriptor.sequence_length_summary),
            "embedding_length": dict(descriptor.embedding_length_summary),
            "truncated_count": descriptor.truncated_count,
            "truncation_rate": descriptor.truncation_rate,
        }

    path.write_text(
        yaml.safe_dump(
            {"antibody": serialize(antibody), "antigen": serialize(antigen)},
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_resource_metrics(
    path: Path,
    trainer: Trainer,
    train_loader: DataLoader,
    antibody: CacheDescriptor,
    antigen: CacheDescriptor,
    training_seconds: float,
) -> None:
    completed_epochs = len(trainer.history)
    pairs_per_epoch = len(train_loader.dataset)
    processed_pairs = pairs_per_epoch * completed_epochs
    interaction = trainer.config.model.interaction
    max_ab_length = antibody.embedding_length_summary["max"]
    max_ag_length = antigen.embedding_length_summary["max"]
    attention_cell_upper_bound = (
        int(
            trainer.config.train.batch_size
            * max_ab_length
            * max_ag_length
            * interaction.num_layers
        )
        if interaction.kind == "deep_cross_attention"
        else 0
    )
    peak_gpu_memory = (
        int(torch.cuda.max_memory_allocated(trainer.device))
        if trainer.device.type == "cuda" and torch.cuda.is_available()
        else None
    )
    throughput = processed_pairs / training_seconds if training_seconds > 0 else None
    payload = {
        "fusion_kind": interaction.kind,
        "interaction_num_layers": interaction.num_layers,
        "batch_size": trainer.config.train.batch_size,
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in trainer.model.parameters()
            if parameter.requires_grad
        ),
        "optimizer_steps": trainer.global_step,
        "epochs_completed": completed_epochs,
        "pair_examples_processed": processed_pairs,
        "training_seconds": training_seconds,
        "pair_examples_per_second": throughput,
        "samples_per_second": throughput,
        "peak_gpu_memory_bytes": peak_gpu_memory,
        "attention_cell_upper_bound_per_batch": attention_cell_upper_bound,
        "antibody": {
            "sequence_length": dict(antibody.sequence_length_summary),
            "embedding_length": dict(antibody.embedding_length_summary),
            "truncated_count": antibody.truncated_count,
            "truncation_rate": antibody.truncation_rate,
        },
        "antigen": {
            "sequence_length": dict(antigen.sequence_length_summary),
            "embedding_length": dict(antigen.embedding_length_summary),
            "truncated_count": antigen.truncated_count,
            "truncation_rate": antigen.truncation_rate,
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _default_output_dir(config_path: Path) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return Path("outputs") / f"{config_path.stem}-{stamp}"


if __name__ == "__main__":
    main()
