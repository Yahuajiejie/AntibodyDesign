"""Frozen-cache Concat/Deep-Cross-Attention RankNet runner."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from ..config import Config
from ..embeddings import ShardedEmbeddingStore, validate_embedding_cache
from ..model import build_ranker
from ..trainer import Trainer
from .artifacts import (
    copy_config,
    write_embedding_metadata_refs,
    write_history,
    write_metrics,
    write_resource_metrics,
    write_run_log,
)
from .data import collect_required_embedding_hashes, load_trainable_records
from .evaluation import predict_cached_records, record_metadata, write_split_evaluation
from .loaders import (
    build_cached_rank_loader,
    build_cached_train_loader,
    compute_group_pair_weights,
)


def run_cached_ranknet(
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
        raise ValueError(
            f"{interaction_kind} frozen_cached training requires model.antigen_encoder"
        )

    train_records = load_trainable_records(train_path, config)
    valid_records = None if valid_path is None else load_trainable_records(valid_path, config)
    test_records = None if test_path is None else load_trainable_records(test_path, config)
    required_hashes = collect_required_embedding_hashes(
        records
        for records in (train_records, valid_records, test_records)
        if records is not None
    )

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

    # Validation precedes model/Trainer construction; Trainer creates optimizer.
    model = build_ranker(config.model, antibody_descriptor, antigen_descriptor)
    antibody_store = ShardedEmbeddingStore(antibody_descriptor.manifest_path)
    antigen_store = ShardedEmbeddingStore(antigen_descriptor.manifest_path)
    train_loader = build_cached_train_loader(
        train_records, config, antibody_store, antigen_store
    )
    valid_loader = build_cached_rank_loader(
        valid_records, config, antibody_store, antigen_store
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
            None if valid_records is None else record_metadata(valid_records)
        ),
        output_dir=output_dir,
        embedding_metadata_hashes=metadata_hashes,
        group_weights=compute_group_pair_weights(train_records, config),
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
    write_history(output_dir / "history.csv", trainer.history)

    metrics: dict[str, float] = {}
    if trainer.history:
        metrics.update(trainer.history[-1])
    metrics["selected_epoch"] = float(trainer.model_epoch)
    if trainer.best_metric is not None:
        metrics["selection_metric_value"] = float(trainer.best_metric)
    if valid_records is not None:
        predictions = predict_cached_records(
            trainer.model,
            valid_records,
            antibody_store,
            antigen_store,
            config.train.batch_size,
        )
        metrics.update(write_split_evaluation(predictions, "valid", output_dir))
    if test_records is not None:
        predictions = predict_cached_records(
            trainer.model,
            test_records,
            antibody_store,
            antigen_store,
            config.train.batch_size,
        )
        metrics.update(write_split_evaluation(predictions, "test", output_dir))

    write_metrics(output_dir / "metrics.json", metrics)
    copy_config(config_path, output_dir)
    write_embedding_metadata_refs(
        output_dir / "embedding_metadata_refs.yaml",
        antibody_descriptor,
        antigen_descriptor,
    )
    write_resource_metrics(
        output_dir / "resource_metrics.json",
        trainer,
        train_loader,
        antibody_descriptor,
        antigen_descriptor,
        training_seconds,
    )
    write_run_log(
        output_dir / "run.log",
        config_path=config_path,
        train_path=train_path,
        valid_path=valid_path,
        test_path=test_path,
        train_records=len(train_records),
        mode="frozen_cached",
        fusion_kind=interaction_kind,
    )
    return metrics
