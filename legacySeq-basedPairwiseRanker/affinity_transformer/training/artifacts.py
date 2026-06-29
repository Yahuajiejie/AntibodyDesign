"""Serialization of training metrics, cache references, logs, and resources."""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from ..embeddings import CacheDescriptor
from ..trainer import Trainer


def write_history(path: Path, history: list[dict[str, float]]) -> None:
    """Write per-epoch training history to a CSV file.

    Each row is one epoch. Columns are the union of all keys that appear across
    all epoch dicts (typically ``epoch``, ``train_loss``, and—when a validation
    set is configured—``valid_macro_spearman``, ``valid_weighted_spearman``,
    ``n_valid_groups``, ``n_skipped_groups``).  Epochs that are missing a key
    (e.g. the first epoch before any validation) get an empty string in that
    column.

    Args:
        path: Destination CSV path (``history.csv`` inside the run output dir).
        history: ``Trainer.history`` — a list of per-epoch metric dicts.
            No-ops silently if the list is empty.
    """
    if not history:
        return
    # Preserve insertion order and collect every key that appears in any epoch.
    columns: list[str] = list(dict.fromkeys(key for row in history for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(history)


def write_metrics(path: Path, payload: dict[str, float]) -> None:
    def clean(value: float) -> float | None:
        return None if isinstance(value, float) and math.isnan(value) else value

    path.write_text(
        json.dumps({key: clean(value) for key, value in payload.items()}, indent=2),
        encoding="utf-8",
    )


def copy_config(config_path: Path, output_dir: Path) -> None:
    shutil.copyfile(config_path, output_dir / "config.yaml")


def write_embedding_metadata_refs(
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


def write_resource_metrics(
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
        "antibody": _length_payload(antibody),
        "antigen": _length_payload(antigen),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_run_log(
    path: Path,
    *,
    config_path: Path,
    train_path: Path,
    valid_path: Path | None,
    test_path: Path | None,
    train_records: int,
    mode: str | None,
    fusion_kind: str | None = None,
) -> None:
    lines = [] if mode is None else [f"mode={mode}"]
    if fusion_kind is not None:
        lines.append(f"fusion_kind={fusion_kind}")
    lines.extend([
        f"config={config_path}",
        f"train_path={train_path}",
        f"valid_path={valid_path}",
        f"test_path={test_path}",
        f"train_records={train_records}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _length_payload(descriptor: CacheDescriptor) -> dict[str, object]:
    return {
        "sequence_length": dict(descriptor.sequence_length_summary),
        "embedding_length": dict(descriptor.embedding_length_summary),
        "truncated_count": descriptor.truncated_count,
        "truncation_rate": descriptor.truncation_rate,
    }
