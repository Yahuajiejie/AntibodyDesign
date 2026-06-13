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

import pandas as pd
import torch
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
from affinity_transformer.metrics import compute_group_spearman, summarize_group_spearman
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


def _resolve_data_paths(config: Config) -> tuple[Path, Path | None, Path | None]:
    if config.data.split_strategy == "none":
        if config.data.train_path is None:
            raise ValueError("data.train_path is required when split_strategy='none'")
        return config.data.train_path, config.data.valid_path, config.data.test_path

    if config.data.all_records_path is None or config.data.split_dir is None:
        raise ValueError("automatic split mode requires all_records_path and split_dir")
    records = load_records(config.data.all_records_path)
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
    pairs = build_pairs(records, config.data.max_pairs_per_group, config.data.seed)
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


def _default_output_dir(config_path: Path) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return Path("outputs") / f"{config_path.stem}-{stamp}"


if __name__ == "__main__":
    main()
