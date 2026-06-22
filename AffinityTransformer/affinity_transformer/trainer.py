"""Training loop, validation, and checkpointing (spec docs/programming_spec.md §5.7).

`Trainer` only consumes already-built objects: a constructed ranker and
`DataLoader`s already wired for either embedding-backed cached batches or the
explicit legacy online token batches. This module does not read raw CSVs, does not
build pairs or groups (that is `dataset.py`, spec §5.2), and does not run
`compute_group_spearman` itself (that is `metrics.py`, spec §5.6) -- it only
calls it.

The "string -> concrete model/tokenizer object" mapping that v0.4 deferred
from `dataloader.py`/`model/` lives at the bottom of this module:
`build_model_and_tokenizers` (currently ESM2-only).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Mapping

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import Config, ModelConfig
from .dataloader import (
    EmbeddingBatch,
    PairBatch,
    PairEmbeddingBatch,
    RankBatch,
    Tokenizer,
)
from .metrics import compute_group_spearman, summarize_group_spearman
from .model import AffinityRanker
from .model.losses import ranknet_loss
from .utils import ensure_dir, get_logger

_logger = get_logger(__name__)


def _fmt_seconds(s: float) -> str:
    """Format elapsed seconds as HH:MM:SS."""
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _try_len(loader) -> int | None:
    """Return len(loader) if available, else None (IterableDataset)."""
    try:
        return len(loader)
    except TypeError:
        return None


#: Columns required on `valid_record_metadata` (spec §5.6 input fields not
#: already carried by `RankBatch`: `dataset_id` and `label_kind`).
VALID_RECORD_METADATA_COLUMNS = ("record_id", "dataset_id", "label_kind")


def _move_rank_batch(batch: RankBatch, device: torch.device) -> RankBatch:
    """Return a copy of `batch` with every tensor moved to `device`.

    Args:
        batch: A `RankBatch` (spec §5.3).
        device: Target device.

    Returns:
        A new `RankBatch` with `antibody_tokens`/`antibody_mask`/`labels`
        (and `antigen_tokens`/`antigen_mask`, if not `None`) moved to
        `device`. `record_ids`/`group_ids` are passed through unchanged.
    """
    return RankBatch(
        antibody_tokens=batch.antibody_tokens.to(device),
        antibody_mask=batch.antibody_mask.to(device),
        antigen_tokens=None if batch.antigen_tokens is None else batch.antigen_tokens.to(device),
        antigen_mask=None if batch.antigen_mask is None else batch.antigen_mask.to(device),
        labels=batch.labels.to(device),
        record_ids=batch.record_ids,
        group_ids=batch.group_ids,
    )


def _move_embedding_batch(batch: EmbeddingBatch, device: torch.device) -> EmbeddingBatch:
    """Return an embedding batch with tensors moved to ``device``."""
    return EmbeddingBatch(
        antibody_embeddings=batch.antibody_embeddings.to(device),
        antibody_mask=batch.antibody_mask.to(device),
        antigen_embeddings=(
            None if batch.antigen_embeddings is None else batch.antigen_embeddings.to(device)
        ),
        antigen_mask=None if batch.antigen_mask is None else batch.antigen_mask.to(device),
        labels=batch.labels.to(device),
        record_ids=batch.record_ids,
        group_ids=batch.group_ids,
    )


def _move_pair_batch(
    batch: PairBatch | PairEmbeddingBatch,
    device: torch.device,
) -> PairBatch | PairEmbeddingBatch:
    """Return a copy of `batch` with every tensor moved to `device`.

    Args:
        batch: A `PairBatch` (spec §5.3).
        device: Target device.

    Returns:
        A new `PairBatch` with `left`/`right` moved via `_move_rank_batch`
        and `y_ij` moved to `device`.
    """
    if isinstance(batch, PairEmbeddingBatch):
        return PairEmbeddingBatch(
            left=_move_embedding_batch(batch.left, device),
            right=_move_embedding_batch(batch.right, device),
            y_ij=batch.y_ij.to(device),
        )
    if isinstance(batch, PairBatch):
        return PairBatch(
            left=_move_rank_batch(batch.left, device),
            right=_move_rank_batch(batch.right, device),
            y_ij=batch.y_ij.to(device),
        )
    raise TypeError(
        "pairwise_ranknet requires PairEmbeddingBatch (cached path) or "
        f"PairBatch (legacy online path), got {type(batch).__name__}"
    )


def _move_record_batch(
    batch: RankBatch | EmbeddingBatch,
    device: torch.device,
) -> RankBatch | EmbeddingBatch:
    if isinstance(batch, EmbeddingBatch):
        return _move_embedding_batch(batch, device)
    if isinstance(batch, RankBatch):
        return _move_rank_batch(batch, device)
    raise TypeError(
        "record evaluation requires EmbeddingBatch (cached path) or "
        f"RankBatch (legacy online path), got {type(batch).__name__}"
    )


class Trainer:
    """Runs the ranking training loop, validation, and checkpointing (spec §5.7).

    `Trainer` is built via dependency injection: `model` and the
    `DataLoader`s are already fully constructed, so this class never loads a
    base encoder or reads an embedding shard. The batch type makes the cached
    versus legacy online path explicit.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader | None = None,
        valid_record_metadata: pd.DataFrame | None = None,
        output_dir: Path | None = None,
        early_stopping_patience: int | None = None,
        early_stopping_metric: str = "valid_weighted_spearman",
        embedding_metadata_hashes: Mapping[str, str] | None = None,
        log_every_n_steps: int = 50,
        eval_log_every_n_batches: int = 100,
        group_weights: Mapping[str, float] | None = None,
    ) -> None:
        """Build a `Trainer` around an already-constructed model.

        Args:
            model: The ranking model to train (spec §5.4). Moved to
                `config.train.device`.
            config: Full run configuration (spec §5.1). Stored verbatim and
                included in every checkpoint (spec §5.7 rule 4); also
                supplies `lr`, `epochs`, `device`, and the random `seed`.
            train_dataloader: Yields `PairBatch` (spec §5.3), e.g. from
                `PairwiseAffinityDataset` + `collate_pair_batch`. Must be
                non-empty.
            valid_dataloader: Yields `RankBatch` (spec §5.3), e.g. from
                `AffinityRecordDataset` + `collate_rank_batch`. If `None`,
                `fit` skips validation.
            valid_record_metadata: Table with columns `record_id, dataset_id,
                label_kind` for every `record_id` that can appear in
                `valid_dataloader` -- the columns `compute_group_spearman`
                needs (spec §5.6) that are not already on `RankBatch`.
                Typically a `record_id, dataset_id, label_kind` slice of the
                same `filter_trainable_records` output used to build the
                validation dataset (spec §5.7 rule 2: trainer consumes
                dataset output, it does not re-derive it). Required by
                `evaluate`.
            output_dir: Directory used to save NaN-loss error contexts (spec
                §5.7 rule 5). If `None`, a NaN loss still stops training
                immediately but no error-context file is written.
            early_stopping_patience: If set, `fit` stops early once
                `early_stopping_metric` has not improved for this many
                consecutive epochs (spec §5.7 rule 6). `None` (default)
                disables early stopping.
            early_stopping_metric: Key into `evaluate`'s output (e.g.
                `"valid_weighted_spearman"`, `"valid_macro_spearman"`) used
                for early stopping. Higher is considered better, and `NaN`
                counts as no improvement. Ignored if
                `early_stopping_patience` is `None` or `valid_dataloader` is
                `None`.

            group_weights: Optional `group_id -> weight` map applied to each
                pair's RankNet loss (e.g. from
                `training.loaders.compute_group_pair_weights`), used to
                restore a group's true `n_records` influence on training
                when `build_pairs` had to cap how many pairs it could sample
                for that group. A pair whose `group_id` is missing from the
                map falls back to weight 1.0. `None` (default) disables
                weighting and reproduces the unweighted mean loss.

        Raises:
            ValueError: If `valid_record_metadata` is given but missing any
                of `record_id, dataset_id, label_kind`.
        """
        if valid_record_metadata is not None:
            missing = [
                c for c in VALID_RECORD_METADATA_COLUMNS if c not in valid_record_metadata.columns
            ]
            if missing:
                raise ValueError(f"valid_record_metadata is missing required column(s): {missing}")

        self.config = config
        self.device = torch.device(config.train.device)
        self.model = model.to(self.device)
        trainable_parameters = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise ValueError("model has no trainable parameters")
        self.optimizer = torch.optim.Adam(trainable_parameters, lr=config.train.lr)
        self._trainable_parameter_names = {
            name for name, parameter in self.model.named_parameters() if parameter.requires_grad
        }

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self._valid_record_metadata = (
            None
            if valid_record_metadata is None
            else valid_record_metadata[list(VALID_RECORD_METADATA_COLUMNS)].copy()
        )

        self.output_dir = None if output_dir is None else Path(output_dir)
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_metric = early_stopping_metric
        self.embedding_metadata_hashes = dict(embedding_metadata_hashes or {})

        self.global_step = 0
        self.current_epoch = 0
        self.model_epoch = 0
        self.best_epoch: int | None = None
        self.best_metric: float | None = None
        self._best_model_state: dict[str, torch.Tensor] | None = None
        self.history: list[dict[str, float]] = []
        self.log_every_n_steps = log_every_n_steps
        self.eval_log_every_n_batches = eval_log_every_n_batches
        self._group_weights = None if group_weights is None else dict(group_weights)

    def fit(self) -> None:
        """Run the full training loop (spec §5.7 rules 3 and 6).

        For each epoch up to `config.train.epochs`: runs one pass over
        `train_dataloader` (RankNet loss + one optimizer step per batch via
        `_run_train_epoch`), then -- if `valid_dataloader` was provided --
        calls `evaluate` on it. Both the epoch's mean training loss and the
        validation group-level metrics are logged every epoch (spec §5.7
        rule 3).

        If `early_stopping_patience` is set and `valid_dataloader` is not
        `None`, training stops once `early_stopping_metric` has not strictly
        improved for `early_stopping_patience` consecutive epochs (spec §5.7
        rule 6).

        Returns:
            None. Side effects: updates `self.model`'s parameters and
            `self.global_step`; writes log lines; may write `error_context.json` under
            `output_dir` (see `_save_error_context`).

        Raises:
            RuntimeError: If any training batch produces a NaN loss (spec
                §5.7 rule 5). Training stops immediately; the model is left
                in whatever state it had after the last successful optimizer
                step.
            ValueError: If `train_dataloader` is empty, or (when early
                stopping is enabled) if `early_stopping_metric` is not a key
                of `evaluate`'s output.
        """
        epochs_without_improvement = 0

        for epoch in range(1, self.config.train.epochs + 1):
            self.current_epoch = epoch
            t_epoch = time.time()
            train_loss = self._run_train_epoch(epoch)
            self.model_epoch = epoch
            summary: dict[str, float] = {"epoch": float(epoch), "train_loss": train_loss}

            valid_metrics: dict[str, float] = {}
            if self.valid_dataloader is not None:
                valid_metrics = self.evaluate(self.valid_dataloader, epoch=epoch)
                summary.update(valid_metrics)

            epoch_time = time.time() - t_epoch
            summary["epoch_time_s"] = epoch_time
            self.history.append(summary)
            _logger.info("epoch %d  time=%s: %s", epoch, _fmt_seconds(epoch_time), summary)

            improved = False
            if self.valid_dataloader is not None:
                if self.early_stopping_metric not in valid_metrics:
                    raise ValueError(
                        f"early_stopping_metric {self.early_stopping_metric!r} is not in "
                        f"evaluate()'s output (keys: {sorted(valid_metrics)})"
                    )
                metric_value = valid_metrics[self.early_stopping_metric]
                improved = not math.isnan(metric_value) and (
                    self.best_metric is None or metric_value > self.best_metric
                )
                if improved:
                    self.best_metric = metric_value
                    self.best_epoch = epoch
                    self._best_model_state = {
                        key: value.detach().cpu().clone()
                        for key, value in self.model.state_dict().items()
                        if key in self._trainable_parameter_names
                    }
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            if self.output_dir is not None:
                latest_path = self.output_dir / "checkpoint_latest.pt"
                self.save_checkpoint(latest_path)
                _logger.info("epoch %d: latest checkpoint → %s", epoch, latest_path)
                if improved:
                    best_path = self.output_dir / "checkpoint_best.pt"
                    self.save_checkpoint(best_path)
                    _logger.info(
                        "epoch %d: best checkpoint (%s=%.6f) → %s",
                        epoch,
                        self.early_stopping_metric,
                        self.best_metric,
                        best_path,
                    )

            if self.early_stopping_patience is not None and self.valid_dataloader is not None:
                if epochs_without_improvement >= self.early_stopping_patience:
                    _logger.info(
                        "early stopping at epoch %d (no improvement in %s for %d epoch(s))",
                        epoch, self.early_stopping_metric, epochs_without_improvement,
                    )
                    break

        # The public model after fit() is the selected model, not simply the
        # state from the final epoch. checkpoint_latest.pt remains available
        # for resuming the interrupted training trajectory.
        if self._best_model_state is not None:
            selected_state = self.model.state_dict()
            selected_state.update(self._best_model_state)
            self.model.load_state_dict(selected_state)
            assert self.best_epoch is not None
            self.model_epoch = self.best_epoch

    def _batch_group_weights(
        self, group_ids: list[str], reference: torch.Tensor
    ) -> torch.Tensor | None:
        """Look up `self._group_weights` for one batch's pairs, or None.

        `group_ids` is `batch.left.group_ids` -- one entry per pair, in
        batch order (left/right share a `group_id` since pairs never cross
        groups). A `group_id` absent from `self._group_weights` (e.g. a
        group with zero sampled pairs, which can't happen, or a stale map)
        falls back to weight 1.0 rather than raising.
        """
        if self._group_weights is None:
            return None
        values = [self._group_weights.get(gid, 1.0) for gid in group_ids]
        return torch.tensor(values, dtype=reference.dtype, device=reference.device)

    def _run_train_epoch(self, epoch: int) -> float:
        """Run one training epoch over `train_dataloader`.

        Args:
            epoch: 1-based epoch number, used only for logging and the
                NaN-loss error context.

        Returns:
            Mean RankNet loss over all batches in `train_dataloader`.

        Raises:
            RuntimeError: If any batch's loss is NaN (spec §5.7 rule 5).
            ValueError: If `train_dataloader` yields no batches.
        """
        self.model.train()
        objective = self.config.model.objective.name
        if objective != "pairwise_ranknet":
            raise NotImplementedError(
                f"Trainer objective {objective!r} is not implemented yet; "
                "the current embedding integration supports pairwise_ranknet"
            )
        total_loss = 0.0
        n_batches = 0
        total_batches = _try_len(self.train_dataloader)
        t0 = time.time()

        autocast_enabled = self.device.type == "cuda"
        for batch in self.train_dataloader:
            batch = _move_pair_batch(batch, self.device)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled
            ):
                score_i = self.model(batch.left)
                score_j = self.model(batch.right)
                weight = self._batch_group_weights(batch.left.group_ids, score_i)
                loss = ranknet_loss(
                    score_i,
                    score_j,
                    batch.y_ij,
                    sigma=self.config.model.objective.sigma,
                    weight=weight,
                )

            if torch.isnan(loss):
                error_path = self._save_error_context(epoch, batch, score_i, score_j)
                message = f"NaN loss at epoch {epoch}, global_step {self.global_step}."
                if error_path is not None:
                    message += f" Error context saved to {error_path}."
                raise RuntimeError(message)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.global_step += 1
            total_loss += loss.item()
            n_batches += 1

            if n_batches % self.log_every_n_steps == 0:
                elapsed = time.time() - t0
                step_str = (
                    f"{n_batches}/{total_batches}" if total_batches else str(n_batches)
                )
                _logger.info(
                    "train epoch %d  step %s  loss=%.4f  avg=%.4f  elapsed=%s",
                    epoch, step_str, loss.item(), total_loss / n_batches,
                    _fmt_seconds(elapsed),
                )

        if n_batches == 0:
            raise ValueError("train_dataloader produced no batches")

        return total_loss / n_batches

    def _save_error_context(
        self,
        epoch: int,
        batch: PairBatch | PairEmbeddingBatch,
        score_i: torch.Tensor,
        score_j: torch.Tensor,
    ) -> Path | None:
        """Write the state surrounding a NaN loss to `output_dir` (spec §5.7 rule 5).

        Args:
            epoch: 1-based epoch number in which the NaN occurred.
            batch: The `PairBatch` (already moved to `self.device`) being
                processed when the NaN loss was computed.
            score_i: `self.model(batch.left)` output for this batch.
            score_j: `self.model(batch.right)` output for this batch.

        Returns:
            Path to the written `error_context.json`, or `None` if
            `output_dir` was not configured (nothing is written in that
            case).
        """
        if self.output_dir is None:
            return None

        context = {
            "epoch": epoch,
            "global_step": self.global_step,
            "left_record_ids": batch.left.record_ids,
            "right_record_ids": batch.right.record_ids,
            "group_ids": batch.left.group_ids,
            "score_i": score_i.detach().cpu().tolist(),
            "score_j": score_j.detach().cpu().tolist(),
            "y_ij": batch.y_ij.detach().cpu().tolist(),
        }
        path = ensure_dir(self.output_dir) / "error_context.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(context, f, indent=2)
        return path

    def evaluate(self, dataloader: DataLoader, epoch: int | None = None) -> dict[str, float]:
        """Compute group-level Spearman metrics over `dataloader` (spec §5.7 rule 3).

        Args:
            dataloader: Yields `RankBatch` (spec §5.3), one entry per record
                -- not pairs. (Spec §5.7 rule 2 means *training* consumes
                pairs; group-level evaluation needs one score per record.)

        Returns:
            Flattened metrics, combining `compute_group_spearman` and
            `summarize_group_spearman` (spec §5.6):
                - `valid_macro_spearman`, `valid_weighted_spearman`,
                  `n_valid_groups`, `n_skipped_groups`: from the `"overall"`
                  summary (spec §8 `metrics.json` minimum fields).
                - For every other `label_kind` present (e.g. `"binary"`,
                  spec §5.6 rule 4): `valid_{label_kind}_macro_spearman`,
                  `valid_{label_kind}_weighted_spearman`,
                  `valid_{label_kind}_n_valid_groups`,
                  `valid_{label_kind}_n_skipped_groups`.

        Raises:
            ValueError: If `dataloader` yields no examples, if `Trainer` was
                constructed without `valid_record_metadata` (needed for
                `dataset_id`/`label_kind`), or if `valid_record_metadata` is
                missing an entry for some `record_id` in `dataloader`.
        """
        if self._valid_record_metadata is None:
            raise ValueError(
                "Trainer.evaluate requires dataset_id/label_kind per record (spec §5.6 "
                "input fields); construct Trainer with valid_record_metadata to enable "
                "evaluate()."
            )

        self.model.eval()
        rows: list[dict[str, object]] = []
        n_eval_batches = 0
        total_eval_batches = _try_len(dataloader)
        epoch_tag = f"epoch {epoch}  " if epoch is not None else ""
        t0 = time.time()
        autocast_enabled = self.device.type == "cuda"
        with torch.no_grad():
            for batch in dataloader:
                batch = _move_record_batch(batch, self.device)
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled
                ):
                    scores = self.model(batch)
                scores = scores.float()
                for record_id, group_id, label, score in zip(
                    batch.record_ids, batch.group_ids, batch.labels.tolist(), scores.tolist()
                ):
                    rows.append({
                        "record_id": record_id,
                        "group_id": group_id,
                        "rank_label": label,
                        "score": score,
                    })
                n_eval_batches += 1
                if n_eval_batches % self.eval_log_every_n_batches == 0:
                    elapsed = time.time() - t0
                    step_str = (
                        f"{n_eval_batches}/{total_eval_batches}"
                        if total_eval_batches
                        else str(n_eval_batches)
                    )
                    _logger.info(
                        "eval %sbatch %s  elapsed=%s",
                        epoch_tag, step_str, _fmt_seconds(elapsed),
                    )

        if not rows:
            raise ValueError("dataloader produced no examples")

        predictions = pd.DataFrame(rows).merge(
            self._valid_record_metadata, on="record_id", how="left", validate="many_to_one"
        )
        missing = predictions[predictions["dataset_id"].isna() | predictions["label_kind"].isna()]
        if not missing.empty:
            raise ValueError(
                "valid_record_metadata is missing dataset_id/label_kind for record_id(s): "
                f"{missing['record_id'].tolist()}"
            )

        group_metrics = compute_group_spearman(predictions)
        if self.output_dir is not None:
            epoch_suffix = f"_epoch{epoch}" if epoch is not None else ""
            group_metrics_path = (
                ensure_dir(self.output_dir) / f"valid_group_metrics{epoch_suffix}.csv"
            )
            group_metrics.sort_values("n_records", ascending=False).to_csv(
                group_metrics_path, index=False
            )
        summary = summarize_group_spearman(group_metrics)
        return _flatten_spearman_summary(summary)

    def save_checkpoint(self, path: Path) -> None:
        """Save a training snapshot (spec §5.7 rule 4).

        Args:
            path: Destination file. Parent directories are created if
                missing.

        Returns:
            None. Writes a `torch.save` pickle containing:
                - `"model_state_dict"`: `self.model.state_dict()`.
                - `"config"`: `self.config` (the full `Config`, spec §5.1).
                - `"global_step"`: `self.global_step`.
                - optimizer and selection state needed to resume or audit.
                - `"seed"`: `self.config.data.seed`.
        """
        path = Path(path)
        ensure_dir(path.parent)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.config,
                "global_step": self.global_step,
                "epoch": self.current_epoch,
                "selected_epoch": self.model_epoch,
                "best_epoch": self.best_epoch,
                "best_metric": self.best_metric,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "seed": self.config.data.seed,
                "embedding_metadata_hashes": self.embedding_metadata_hashes,
            },
            path,
        )

    def load_checkpoint(
        self, path: Path, map_location: str | torch.device | None = None
    ) -> dict[str, object]:
        """Load a checkpoint written by `save_checkpoint` into this `Trainer`.

        Args:
            path: Checkpoint file written by `save_checkpoint`.
            map_location: Passed to `torch.load`; defaults to `self.device`.

        Returns:
            The raw checkpoint dict, after applying `model_state_dict` to
            `self.model` and restoring available training state. `self.config` is left unchanged --
            the caller is responsible for constructing a `Trainer` whose
            `model` architecture matches `checkpoint["config"].model`.

        Raises:
            FileNotFoundError: If `path` does not exist (propagated from
                `torch.load`).
        """
        try:
            checkpoint = torch.load(path, map_location=map_location or self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=map_location or self.device)
        checkpoint_hashes = dict(checkpoint.get("embedding_metadata_hashes", {}))
        if checkpoint_hashes != self.embedding_metadata_hashes:
            raise ValueError(
                "checkpoint embedding metadata hash mismatch: "
                f"{checkpoint_hashes} != {self.embedding_metadata_hashes}"
            )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.global_step = int(checkpoint["global_step"])
        self.current_epoch = int(checkpoint.get("epoch", 0))
        self.model_epoch = int(checkpoint.get("selected_epoch", self.current_epoch))
        self.best_epoch = checkpoint.get("best_epoch")
        self.best_metric = checkpoint.get("best_metric")
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        return checkpoint


def _flatten_spearman_summary(summary: dict[str, dict[str, float | int]]) -> dict[str, float]:
    """Flatten `summarize_group_spearman`'s output for `Trainer.evaluate`.

    Args:
        summary: Output of `summarize_group_spearman`: a dict keyed by
            `"overall"` plus each distinct `label_kind`.

    Returns:
        A flat `dict[str, float]`. The `"overall"` entry's `macro_spearman`,
        `weighted_spearman`, `n_valid_groups`, `n_skipped_groups` become
        `valid_macro_spearman`, `valid_weighted_spearman`, `n_valid_groups`,
        `n_skipped_groups` (spec §8 `metrics.json` minimum fields). Every
        other `label_kind` entry (e.g. `"binary"`, spec §5.6 rule 4) becomes
        `valid_{label_kind}_macro_spearman`,
        `valid_{label_kind}_weighted_spearman`,
        `valid_{label_kind}_n_valid_groups`,
        `valid_{label_kind}_n_skipped_groups`.
    """
    flat: dict[str, float] = {}
    overall = summary["overall"]
    flat["valid_macro_spearman"] = float(overall["macro_spearman"])
    flat["valid_weighted_spearman"] = float(overall["weighted_spearman"])
    flat["n_valid_groups"] = float(overall["n_valid_groups"])
    flat["n_skipped_groups"] = float(overall["n_skipped_groups"])

    for label_kind, stats in summary.items():
        if label_kind == "overall":
            continue
        flat[f"valid_{label_kind}_macro_spearman"] = float(stats["macro_spearman"])
        flat[f"valid_{label_kind}_weighted_spearman"] = float(stats["weighted_spearman"])
        flat[f"valid_{label_kind}_n_valid_groups"] = float(stats["n_valid_groups"])
        flat[f"valid_{label_kind}_n_skipped_groups"] = float(stats["n_skipped_groups"])

    return flat


# ── spec §5.7: string -> model/tokenizer mapping ────────────────────────────

#: ESM2 short names (the `ModelConfig.antibody_encoder` / `antigen_encoder`
#: convention from `config.py`) -> hidden dimension. Must match the
#: `d_model` produced by the corresponding `facebook/{name}_UR50D` checkpoint.
ESM2_D_MODEL: dict[str, int] = {
    "esm2_t6_8M": 320,
    "esm2_t12_35M": 480,
    "esm2_t30_150M": 640,
    "esm2_t33_650M": 1280,
    "esm2_t36_3B": 2560,
    "esm2_t48_15B": 5120,
}


class _Esm2EncoderWrapper(nn.Module):
    """Adapts a HuggingFace ESM2 model to the spec §5.4 encoder interface.
    note；
    wrapper v 包装，指的是把ESM2的输出转化为我们自己需要的格式
    """

    def __init__(self, esm_model: nn.Module, *, frozen: bool = False) -> None:
        super().__init__()
        self.esm_model = esm_model
        self.frozen = frozen
        if frozen:
            self.requires_grad_(False)
            super().train(False)

    def train(self, mode: bool = True) -> "_Esm2EncoderWrapper":
        """Keep a frozen base encoder in evaluation mode.

        Calling ``model.train()`` recursively toggles every child module by
        default. A frozen encoder must not reactivate dropout merely because
        the trainable scoring layers enter training mode.
        """
        return super().train(False if self.frozen else mode)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Map `(input_ids, attention_mask) -> FloatTensor[B, L, d_model]` (spec §5.4)."""
        if self.frozen:
            with torch.no_grad():
                outputs = self.esm_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask.long(),
                )
        else:
            outputs = self.esm_model(
                input_ids=input_ids,
                attention_mask=attention_mask.long(),
            )
        return torch.nan_to_num(outputs.last_hidden_state, nan=0.0)


def _resolve_esm2(name: str, role: str, d_model: int) -> str:
    """Validate one `ModelConfig` encoder short name and return its HF repo id.

    Args:
        name: Value of `model_config.antibody_encoder` or
            `model_config.antigen_encoder`.
        role: `"antibody_encoder"` or `"antigen_encoder"`, used only in error
            messages.
        d_model: `model_config.d_model`, checked against the known hidden
            size of `name`.

    Returns:
        The HuggingFace Hub repo id, `f"facebook/{name}_UR50D"`.

    Raises:
        ValueError: If `name` is not a recognized ESM2 short name (currently
            the only supported encoder family -- e.g. AbLang-2 is not yet
            implemented), or if `d_model` does not match the hidden size of
            `name`.
    """
    if name not in ESM2_D_MODEL:
        raise ValueError(
            f"Unsupported {role} {name!r}. build_model_and_tokenizers currently only "
            f"supports ESM2 short names: {sorted(ESM2_D_MODEL)}. Other encoders (e.g. "
            "AbLang-2) are not yet implemented."
        )
    expected_d_model = ESM2_D_MODEL[name]
    if d_model != expected_d_model:
        raise ValueError(
            f"model_config.d_model={d_model} does not match the hidden size of "
            f"{role}={name!r} ({expected_d_model})."
        )
    return f"facebook/{name}_UR50D"


def build_model_and_tokenizers(
    model_config: ModelConfig,
) -> tuple[AffinityRanker, Tokenizer, Tokenizer | None]:
    """Build a real `AffinityRanker` + tokenizers from a `ModelConfig` (spec §5.7).

    This is the "string -> concrete model/tokenizer object" mapping that
    v0.4 deferred from `dataloader.py`/`model/` to `trainer.py`. It is the
    only place in `affinity_transformer` that imports `transformers` or
    downloads pretrained weights, and that import happens inside this
    function body (not at module import time), per spec §9 "禁止在 import
    package 时加载大模型或读取大数据".

    Args:
        model_config: `Config.model` (spec §5.1). `antibody_encoder`, and
            `antigen_encoder` if not `None`, must be ESM2 short names from
            `ESM2_D_MODEL` (e.g. `"esm2_t12_35M"`), and `d_model` must match
            the corresponding hidden size.

    Returns:
        `(model, antibody_tokenizer, antigen_tokenizer)`:
            - `model`: `AffinityRanker` built from freshly-constructed ESM2
              encoders, with `d_model` and `use_cross_attention` from
              `model_config`.
            - `antibody_tokenizer`: HuggingFace tokenizer for
              `model_config.antibody_encoder`, satisfying the spec §5.3
              `Tokenizer` protocol.
            - `antigen_tokenizer`: HuggingFace tokenizer for
              `model_config.antigen_encoder`, or `None` if
              `model_config.antigen_encoder is None`.

    Raises:
        ValueError: If `antibody_encoder` / `antigen_encoder` is not a
            supported ESM2 short name, or `d_model` does not match its
            hidden size. This validation happens *before* any import of
            `transformers` or network access, so an unsupported config fails
            immediately (spec §1.2: other encoder families, e.g. AbLang-2,
            are not yet implemented and must fail loudly rather than silently
            falling back).
        ImportError: If the `transformers` package is not installed.

    Note:
        Tokenizing a heavy|light paired-chain string (spec §5.3 antibody
        sequence concatenation rule, the `"|"` separator) with a stock ESM2
        tokenizer has not been validated against AbLang-2's paired-chain
        convention; this is a known limitation of the current ESM2-only
        mapping, not addressed here.
    """
    antibody_config = model_config.antibody_encoder
    antigen_config = model_config.antigen_encoder
    if antibody_config.mode == "frozen_cached" or (
        antigen_config is not None and antigen_config.mode == "frozen_cached"
    ):
        raise ValueError(
            "build_model_and_tokenizers is the online path and cannot consume "
            "frozen_cached config; use model.build_ranker with validated caches"
        )
    online_configs = [("antibody_encoder", antibody_config)]
    if antigen_config is not None:
        online_configs.append(("antigen_encoder", antigen_config))
    for role, encoder_config in online_configs:
        if encoder_config.mode == "lora_online":
            raise NotImplementedError(
                f"{role}.mode='lora_online' is not implemented; refusing to "
                "silently run full fine-tuning"
            )
        if encoder_config.mode != "frozen_online":
            raise ValueError(
                f"unsupported online mode for {role}: {encoder_config.mode!r}"
            )
    antibody_repo = _resolve_esm2(
        antibody_config.name, "antibody_encoder", model_config.d_model
    )
    antigen_repo = (
        None
        if antigen_config is None
        else _resolve_esm2(antigen_config.name, "antigen_encoder", model_config.d_model)
    )

    from transformers import AutoModel, AutoTokenizer  # lazy: spec §9

    antibody_tokenizer = AutoTokenizer.from_pretrained(
        antibody_repo,
        revision=antibody_config.tokenizer_revision,
    )
    antibody_encoder = _Esm2EncoderWrapper(
        AutoModel.from_pretrained(antibody_repo, revision=antibody_config.revision),
        frozen=True,
    )

    if antigen_repo is None:
        antigen_tokenizer = None
        antigen_encoder = None
    else:
        assert antigen_config is not None
        antigen_tokenizer = AutoTokenizer.from_pretrained(
            antigen_repo,
            revision=antigen_config.tokenizer_revision,
        )
        antigen_encoder = _Esm2EncoderWrapper(
            AutoModel.from_pretrained(antigen_repo, revision=antigen_config.revision),
            frozen=True,
        )

    model = AffinityRanker(
        antibody_encoder=antibody_encoder,
        antigen_encoder=antigen_encoder,
        d_model=model_config.d_model,
        use_cross_attention=model_config.use_cross_attention,
    )
    return model, antibody_tokenizer, antigen_tokenizer
