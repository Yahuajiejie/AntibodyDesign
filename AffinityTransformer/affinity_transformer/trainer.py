"""Training loop, validation, and checkpointing (spec docs/programming_spec.md §5.7).

`Trainer` only consumes already-built objects: a constructed `AffinityRanker`
(spec §5.4) and `DataLoader`s whose `collate_fn` is already wired to
`collate_pair_batch` / `collate_rank_batch` (spec §5.3) with whatever
tokenizers the caller chose. This module does not read raw CSVs, does not
build pairs or groups (that is `dataset.py`, spec §5.2), and does not run
`compute_group_spearman` itself (that is `metrics.py`, spec §5.6) -- it only
calls it.

The "string -> concrete model/tokenizer object" mapping that v0.4 deferred
from `dataloader.py`/`model.py` lives at the bottom of this module:
`build_model_and_tokenizers` (currently ESM2-only).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import Config, ModelConfig
from .dataloader import PairBatch, RankBatch, Tokenizer
from .losses import ranknet_loss
from .metrics import compute_group_spearman, summarize_group_spearman
from .model import AffinityRanker
from .utils import ensure_dir, get_logger, set_seed

_logger = get_logger(__name__)

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


def _move_pair_batch(batch: PairBatch, device: torch.device) -> PairBatch:
    """Return a copy of `batch` with every tensor moved to `device`.

    Args:
        batch: A `PairBatch` (spec §5.3).
        device: Target device.

    Returns:
        A new `PairBatch` with `left`/`right` moved via `_move_rank_batch`
        and `y_ij` moved to `device`.
    """
    return PairBatch(
        left=_move_rank_batch(batch.left, device),
        right=_move_rank_batch(batch.right, device),
        y_ij=batch.y_ij.to(device),
    )


class Trainer:
    """Runs the RankNet training loop, validation, and checkpointing (spec §5.7).

    `Trainer` is built via dependency injection: `model` and the
    `DataLoader`s are already fully constructed (including whichever
    tokenizers their `collate_fn`s use), so this class never needs to know
    whether it is training on real ESM2 encoders or the
    `FakeEncoder`/`FakeTokenizer` test doubles.
    """

    def __init__(
        self,
        model: AffinityRanker,
        config: Config,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader | None = None,
        valid_record_metadata: pd.DataFrame | None = None,
        output_dir: Path | None = None,
        early_stopping_patience: int | None = None,
        early_stopping_metric: str = "valid_weighted_spearman",
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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.train.lr)

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

        self.global_step = 0
        self.history: list[dict[str, float]] = []

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
            `self.global_step`; reseeds global RNG state via `set_seed`;
            writes log lines; may write `error_context.json` under
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
        set_seed(self.config.data.seed)

        best_metric: float | None = None
        epochs_without_improvement = 0

        for epoch in range(1, self.config.train.epochs + 1):
            train_loss = self._run_train_epoch(epoch)
            summary: dict[str, float] = {"epoch": float(epoch), "train_loss": train_loss}

            valid_metrics: dict[str, float] = {}
            if self.valid_dataloader is not None:
                valid_metrics = self.evaluate(self.valid_dataloader)
                summary.update(valid_metrics)

            self.history.append(summary)
            _logger.info("epoch %d: %s", epoch, summary)

            if self.early_stopping_patience is not None and self.valid_dataloader is not None:
                if self.early_stopping_metric not in valid_metrics:
                    raise ValueError(
                        f"early_stopping_metric {self.early_stopping_metric!r} is not in "
                        f"evaluate()'s output (keys: {sorted(valid_metrics)})"
                    )
                metric_value = valid_metrics[self.early_stopping_metric]

                improved = not math.isnan(metric_value) and (
                    best_metric is None or metric_value > best_metric
                )
                if improved:
                    best_metric = metric_value
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= self.early_stopping_patience:
                    _logger.info(
                        "early stopping at epoch %d (no improvement in %s for %d epoch(s))",
                        epoch, self.early_stopping_metric, epochs_without_improvement,
                    )
                    break

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
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_dataloader:
            batch = _move_pair_batch(batch, self.device)
            score_i = self.model(batch.left)
            score_j = self.model(batch.right)
            loss = ranknet_loss(score_i, score_j, batch.y_ij)

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

        if n_batches == 0:
            raise ValueError("train_dataloader produced no batches")

        return total_loss / n_batches

    def _save_error_context(
        self, epoch: int, batch: PairBatch, score_i: torch.Tensor, score_j: torch.Tensor
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

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
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
        with torch.no_grad():
            for batch in dataloader:
                batch = _move_rank_batch(batch, self.device)
                scores = self.model(batch)
                for record_id, group_id, label, score in zip(
                    batch.record_ids, batch.group_ids, batch.labels.tolist(), scores.tolist()
                ):
                    rows.append({
                        "record_id": record_id,
                        "group_id": group_id,
                        "rank_label": label,
                        "score": score,
                    })

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
                - `"seed"`: `self.config.data.seed`.
        """
        path = Path(path)
        ensure_dir(path.parent)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.config,
                "global_step": self.global_step,
                "seed": self.config.data.seed,
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
            The raw checkpoint dict (`model_state_dict, config, global_step,
            seed`), after applying `model_state_dict` to `self.model` and
            restoring `self.global_step`. `self.config` is left unchanged --
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
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.global_step = int(checkpoint["global_step"])
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

    def __init__(self, esm_model: nn.Module) -> None:
        super().__init__()
        self.esm_model = esm_model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Map `(input_ids, attention_mask) -> FloatTensor[B, L, d_model]` (spec §5.4)."""
        outputs = self.esm_model(input_ids=input_ids, attention_mask=attention_mask.long())
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
    v0.4 deferred from `dataloader.py`/`model.py` to `trainer.py`. It is the
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
    antibody_repo = _resolve_esm2(model_config.antibody_encoder, "antibody_encoder", model_config.d_model)
    antigen_repo = (
        None
        if model_config.antigen_encoder is None
        else _resolve_esm2(model_config.antigen_encoder, "antigen_encoder", model_config.d_model)
    )

    from transformers import AutoModel, AutoTokenizer  # lazy: spec §9

    antibody_tokenizer = AutoTokenizer.from_pretrained(antibody_repo)
    antibody_encoder = _Esm2EncoderWrapper(AutoModel.from_pretrained(antibody_repo))

    if antigen_repo is None:
        antigen_tokenizer = None
        antigen_encoder = None
    else:
        antigen_tokenizer = AutoTokenizer.from_pretrained(antigen_repo)
        antigen_encoder = _Esm2EncoderWrapper(AutoModel.from_pretrained(antigen_repo))

    model = AffinityRanker(
        antibody_encoder=antibody_encoder,
        antigen_encoder=antigen_encoder,
        d_model=model_config.d_model,
        use_cross_attention=model_config.use_cross_attention,
    )
    return model, antibody_tokenizer, antigen_tokenizer
