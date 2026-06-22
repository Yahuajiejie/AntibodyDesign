"""Legacy explicit online-token training runner."""

from __future__ import annotations

from pathlib import Path

from ..config import Config
from ..trainer import Trainer, build_model_and_tokenizers
from .artifacts import copy_config, write_history, write_metrics, write_run_log
from .evaluation import predict_online_records, record_metadata, write_split_evaluation
from .loaders import (
    build_online_rank_loader,
    build_online_train_loader,
    compute_group_pair_weights,
)


def run_online_training(
    config_path: Path,
    config: Config,
    output_dir: Path,
    train_path: Path,
    valid_path: Path | None,
    test_path: Path | None,
) -> dict[str, float]:
    """Run the explicit legacy online encoder path without cache fallback."""
    model, antibody_tokenizer, antigen_tokenizer = build_model_and_tokenizers(config.model)
    train_records, train_loader = build_online_train_loader(
        train_path, config, antibody_tokenizer, antigen_tokenizer
    )
    valid_records, valid_loader = build_online_rank_loader(
        valid_path, config, antibody_tokenizer, antigen_tokenizer
    )
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        valid_dataloader=valid_loader,
        valid_record_metadata=(
            None if valid_records is None else record_metadata(valid_records)
        ),
        output_dir=output_dir,
        group_weights=compute_group_pair_weights(train_records, config),
    )
    trainer.fit()
    trainer.save_checkpoint(output_dir / "checkpoint.pt")
    write_history(output_dir / "history.csv", trainer.history)

    metrics: dict[str, float] = {}
    if trainer.history:
        metrics.update(trainer.history[-1])
    metrics["selected_epoch"] = float(trainer.model_epoch)
    if trainer.best_metric is not None:
        metrics["selection_metric_value"] = float(trainer.best_metric)
    if valid_records is not None:
        predictions = predict_online_records(
            trainer.model,
            valid_records,
            antibody_tokenizer,
            antigen_tokenizer,
            config.train.batch_size,
        )
        metrics.update(write_split_evaluation(predictions, "valid", output_dir))
    if test_path is not None:
        test_records, _ = build_online_rank_loader(
            test_path, config, antibody_tokenizer, antigen_tokenizer
        )
        if test_records is not None:
            predictions = predict_online_records(
                trainer.model,
                test_records,
                antibody_tokenizer,
                antigen_tokenizer,
                config.train.batch_size,
            )
            metrics.update(write_split_evaluation(predictions, "test", output_dir))

    write_metrics(output_dir / "metrics.json", metrics)
    copy_config(config_path, output_dir)
    write_run_log(
        output_dir / "run.log",
        config_path=config_path,
        train_path=train_path,
        valid_path=valid_path,
        test_path=test_path,
        train_records=len(train_records),
        mode=None,
    )
    return metrics
