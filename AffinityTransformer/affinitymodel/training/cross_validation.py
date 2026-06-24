"""Group-isolated K-fold training orchestration."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Callable

import pandas as pd

from ..config import Config
from ..dataset import filter_trainable_records, load_records
from ..splits import (
    EntityColdStartFold,
    build_antibody_cold_start_kfolds,
    build_antigen_cold_start_kfolds,
    build_group_kfolds,
)
from ..utils import ensure_dir, set_seed

TrainingRunner = Callable[
    [Path, Config, Path, Path, Path | None, Path | None],
    dict[str, float],
]


def run_group_kfold_cross_validation(
    config_path: Path,
    config: Config,
    output_dir: Path,
    runner: TrainingRunner,
) -> dict[str, float]:
    """Train one independent model per configured protocol fold.

    The configured test table is intentionally not passed to fold runners.
    It remains an untouched final holdout rather than being inspected K times.

    The function name is retained for backwards compatibility. Configurations
    that omit ``cross_validation.protocol`` keep the original group-isolated
    behavior.
    """
    cv = config.cross_validation
    if not cv.enabled:
        raise ValueError("cross-validation runner requires cross_validation.enabled=true")
    if config.data.train_path is None:
        raise ValueError("cross-validation requires data.train_path")

    pool_tables = [load_records(config.data.train_path)]
    if cv.source == "train_valid":
        if config.data.valid_path is None:
            raise ValueError("train_valid cross-validation requires data.valid_path")
        pool_tables.append(load_records(config.data.valid_path))
    pool = filter_trainable_records(pd.concat(pool_tables, ignore_index=True))
    if pool["record_id"].astype(str).duplicated().any():
        duplicates = pool.loc[
            pool["record_id"].astype(str).duplicated(), "record_id"
        ].astype(str).tolist()
        raise ValueError(
            "cross-validation pool contains duplicate record_id values: "
            f"{duplicates[:10]}"
        )

    if cv.protocol == "group_holdout":
        folds = build_group_kfolds(pool, n_splits=cv.n_splits, seed=cv.seed)
    elif cv.protocol == "antibody_cold_start":
        folds = build_antibody_cold_start_kfolds(
            pool,
            n_splits=cv.n_splits,
            seed=cv.seed,
            min_eval_records=cv.min_eval_records,
            require_train_group=cv.require_train_group,
        )
    elif cv.protocol == "antigen_cold_start":
        folds = build_antigen_cold_start_kfolds(
            pool,
            n_splits=cv.n_splits,
            seed=cv.seed,
            min_eval_records=cv.min_eval_records,
        )
    else:  # guarded by config validation
        raise ValueError(f"unsupported cross-validation protocol: {cv.protocol!r}")
    output_dir = ensure_dir(output_dir)
    shutil.copyfile(config_path, output_dir / "config.yaml")
    if folds and isinstance(folds[0], EntityColdStartFold):
        folds[0].unit_assignments.to_parquet(
            output_dir / "unit_assignments.parquet", index=False
        )

    assignments = []
    fold_rows: list[dict[str, float | int]] = []
    for fold in folds:
        # Each fold is an independent model. Reseed at the fold boundary so
        # its initialization does not depend on how much RNG state an earlier
        # fold happened to consume.
        set_seed(config.data.seed + fold.index)
        fold_name = f"fold_{fold.index + 1:02d}"
        fold_dir = ensure_dir(output_dir / fold_name)
        split_dir = ensure_dir(fold_dir / "splits")
        train_path = split_dir / "train.parquet"
        valid_path = split_dir / "valid.parquet"
        fold.train.to_parquet(train_path, index=False)
        fold.valid.to_parquet(valid_path, index=False)

        if isinstance(fold, EntityColdStartFold):
            fold.eligibility_report.to_csv(
                fold_dir / "eligibility_report.csv", index=False
            )
            fold.leakage_report.to_csv(fold_dir / "leakage_report.csv", index=False)
            fold.excluded_records.to_parquet(
                fold_dir / "excluded_records.parquet", index=False
            )

        if isinstance(fold, EntityColdStartFold):
            assignments.extend(
                {
                    "record_id": str(row.record_id),
                    "group_id": str(row.group_id),
                    "fold": fold.index + 1,
                    "protocol": fold.protocol,
                }
                for row in fold.valid[["record_id", "group_id"]].itertuples(index=False)
            )
        else:
            assignments.extend(
                {
                    "record_id": str(row.record_id),
                    "group_id": str(row.group_id),
                    "fold": fold.index + 1,
                }
                for row in fold.valid[["record_id", "group_id"]].itertuples(index=False)
            )
        metrics = runner(
            config_path,
            config,
            fold_dir,
            train_path,
            valid_path,
            None,
        )
        fold_rows.append(
            {
                "fold": fold.index + 1,
                "n_train_records": len(fold.train),
                "n_valid_records": len(fold.valid),
                "n_excluded_records": (
                    len(fold.excluded_records)
                    if isinstance(fold, EntityColdStartFold)
                    else 0
                ),
                **metrics,
            }
        )

    pd.DataFrame(assignments).sort_values("record_id").to_csv(
        output_dir / "fold_assignments.csv", index=False
    )
    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)

    summary = _summarize_validation_metrics(fold_metrics)
    (output_dir / "cross_validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return {
        f"cv_{metric}_{statistic}": float(value)
        for metric, statistics in summary["metrics"].items()
        for statistic, value in statistics.items()
        if statistic in {"mean", "std"}
    }


def _summarize_validation_metrics(fold_metrics: pd.DataFrame) -> dict[str, object]:
    metrics: dict[str, dict[str, float | int]] = {}
    for column in fold_metrics.columns:
        if not column.startswith("valid_"):
            continue
        values = pd.to_numeric(fold_metrics[column], errors="coerce")
        values = values[values.apply(lambda value: math.isfinite(float(value)))]
        if values.empty:
            continue
        metrics[column] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "n_folds": int(len(values)),
        }
    return {
        "n_splits": int(len(fold_metrics)),
        "metrics": metrics,
    }
