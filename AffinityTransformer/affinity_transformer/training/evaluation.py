"""Record-level prediction and group-metric artifact generation."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..dataloader import RankBatch, Tokenizer, collate_rank_batch
from ..dataset import AffinityRecordDataset
from ..embeddings import EmbeddingBatch, EmbeddingStore, collate_embedding_batch
from ..metrics import compute_group_spearman, summarize_group_spearman


def record_metadata(records: pd.DataFrame) -> pd.DataFrame:
    return records[["record_id", "dataset_id", "label_kind"]].copy()


def predict_online_records(
    model: nn.Module,
    records: pd.DataFrame,
    antibody_tokenizer: Tokenizer,
    antigen_tokenizer: Tokenizer | None,
    batch_size: int,
) -> pd.DataFrame:
    loader = DataLoader(
        AffinityRecordDataset(records),
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
            moved = RankBatch(
                antibody_tokens=batch.antibody_tokens.to(device),
                antibody_mask=batch.antibody_mask.to(device),
                antigen_tokens=(
                    None if batch.antigen_tokens is None else batch.antigen_tokens.to(device)
                ),
                antigen_mask=(
                    None if batch.antigen_mask is None else batch.antigen_mask.to(device)
                ),
                labels=batch.labels.to(device),
                record_ids=batch.record_ids,
                group_ids=batch.group_ids,
            )
            rows.extend(_score_rows(model, moved, batch))
    return _merge_metadata(rows, records)


def predict_cached_records(
    model: nn.Module,
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
            rows.extend(_score_rows(model, moved, batch))
    return _merge_metadata(rows, records)


def write_split_evaluation(
    predictions: pd.DataFrame,
    split: str,
    output_dir: Path,
) -> dict[str, float]:
    """Write predictions/group metrics and return flattened summary metrics."""
    if split not in {"valid", "test"}:
        raise ValueError("evaluation split must be 'valid' or 'test'")
    predictions = predictions.copy()
    predictions["split"] = split
    group_metrics = compute_group_spearman(predictions)
    if split == "valid":
        predictions_path = output_dir / "predictions.csv"
        group_metrics_path = output_dir / "group_metrics.csv"
    else:
        predictions_path = output_dir / "test_predictions.csv"
        group_metrics_path = output_dir / "test_group_metrics.csv"
    group_metrics.to_csv(group_metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    return _flatten_summary(summarize_group_spearman(group_metrics), split)


def _score_rows(model: nn.Module, moved, original) -> list[dict[str, object]]:
    scores = model(moved)
    return [
        {
            "record_id": record_id,
            "group_id": group_id,
            "rank_label": label,
            "score": score,
        }
        for record_id, group_id, label, score in zip(
            original.record_ids,
            original.group_ids,
            original.labels.tolist(),
            scores.detach().cpu().tolist(),
        )
    ]


def _merge_metadata(
    rows: list[dict[str, object]], records: pd.DataFrame
) -> pd.DataFrame:
    return pd.DataFrame(rows).merge(
        record_metadata(records),
        on="record_id",
        how="left",
        validate="many_to_one",
    )


def _flatten_summary(
    summary: dict[str, dict[str, float | int]], prefix: str
) -> dict[str, float]:
    overall = summary["overall"]
    return {
        f"{prefix}_macro_spearman": float(overall["macro_spearman"]),
        f"{prefix}_weighted_spearman": float(overall["weighted_spearman"]),
        f"{prefix}_n_valid_groups": float(overall["n_valid_groups"]),
        f"{prefix}_n_skipped_groups": float(overall["n_skipped_groups"]),
    }
