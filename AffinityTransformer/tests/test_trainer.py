"""Tests for affinity_transformer.trainer (spec §5.7)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from affinity_transformer.config import Config, DataConfig, ModelConfig, TrainConfig
from affinity_transformer.dataloader import collate_pair_batch, collate_rank_batch
from affinity_transformer.dataset import (
    AffinityRecordDataset,
    PairwiseAffinityDataset,
    build_pairs,
    filter_trainable_records,
)
from affinity_transformer.model import AffinityRanker
from affinity_transformer.trainer import (
    ESM2_D_MODEL,
    Trainer,
    _resolve_esm2,
    build_model_and_tokenizers,
)

D_MODEL = 16


def _make_config(epochs: int = 1, seed: int = 0) -> Config:
    return Config(
        data=DataConfig(train_path=Path("unused.parquet"), valid_path=None, max_pairs_per_group=50, seed=seed),
        model=ModelConfig(antibody_encoder="fake", antigen_encoder=None, d_model=D_MODEL, use_cross_attention=False),
        train=TrainConfig(batch_size=4, lr=1.0e-3, epochs=epochs, device="cpu"),
    )


def _build_trainer(toy_records, antibody_tokenizer, make_fake_encoder, *, epochs=1, seed=0, **trainer_kwargs):
    """Build a Trainer + its pairs table over `toy_records` (spec §7.3 group types)."""
    trainable = filter_trainable_records(toy_records)
    pairs = build_pairs(trainable, max_pairs_per_group=50, seed=seed)

    train_dataloader = DataLoader(
        PairwiseAffinityDataset(trainable, pairs),
        batch_size=4,
        shuffle=False,
        collate_fn=lambda examples: collate_pair_batch(examples, antibody_tokenizer, None),
    )
    valid_dataloader = DataLoader(
        AffinityRecordDataset(trainable),
        batch_size=4,
        shuffle=False,
        collate_fn=lambda examples: collate_rank_batch(examples, antibody_tokenizer, None),
    )
    valid_record_metadata = trainable[["record_id", "dataset_id", "label_kind"]]

    model = AffinityRanker(
        antibody_encoder=make_fake_encoder(D_MODEL),
        antigen_encoder=None,
        d_model=D_MODEL,
        use_cross_attention=False,
    )
    config = _make_config(epochs=epochs, seed=seed)
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        valid_record_metadata=valid_record_metadata,
        **trainer_kwargs,
    )
    return trainer, pairs


# ── §7.3 minimal acceptance: one epoch on <=10 records ──────────────────────


def test_trainer_fit_runs_one_epoch_on_toy_data(toy_records, antibody_tokenizer, make_fake_encoder):
    """spec §5.7 acceptance 1: runs one epoch on toy data without error."""
    trainer, pairs = _build_trainer(toy_records, antibody_tokenizer, make_fake_encoder, epochs=1)
    assert len(pairs) > 0

    trainer.fit()

    assert trainer.global_step == len(trainer.train_dataloader)


# ── §5.7 acceptance 2: save and reload checkpoint ────────────────────────────


def test_trainer_save_and_load_checkpoint(tmp_path, toy_records, antibody_tokenizer, make_fake_encoder):
    trainer, _ = _build_trainer(toy_records, antibody_tokenizer, make_fake_encoder, epochs=1)
    trainer.fit()

    checkpoint_path = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(checkpoint_path)
    assert checkpoint_path.exists()

    new_trainer, _ = _build_trainer(toy_records, antibody_tokenizer, make_fake_encoder, epochs=1)
    # Perturb the fresh model so a successful load is verifiable.
    with torch.no_grad():
        for param in new_trainer.model.parameters():
            param.add_(1.0)
    perturbed = {k: v.clone() for k, v in new_trainer.model.state_dict().items()}

    checkpoint = new_trainer.load_checkpoint(checkpoint_path)

    assert checkpoint["global_step"] == trainer.global_step
    assert checkpoint["seed"] == trainer.config.data.seed
    assert new_trainer.global_step == trainer.global_step

    after = new_trainer.model.state_dict()
    assert any(not torch.equal(perturbed[k], after[k]) for k in perturbed)
    for key, value in trainer.model.state_dict().items():
        assert torch.equal(value, after[key])


# ── §5.7 acceptance 3: pair order is reproducible with a fixed seed ─────────


def test_pair_order_reproducible_with_fixed_seed(toy_records):
    trainable = filter_trainable_records(toy_records)

    pairs_a = build_pairs(trainable, max_pairs_per_group=50, seed=0)
    pairs_b = build_pairs(trainable, max_pairs_per_group=50, seed=0)

    assert pairs_a["pair_id"].tolist() == pairs_b["pair_id"].tolist()


# ── evaluate() / metrics integration ─────────────────────────────────────────


def test_trainer_evaluate_returns_overall_and_per_label_kind_metrics(
    toy_records, antibody_tokenizer, make_fake_encoder
):
    trainer, _ = _build_trainer(toy_records, antibody_tokenizer, make_fake_encoder, epochs=1)

    metrics = trainer.evaluate(trainer.valid_dataloader)

    for key in ("valid_macro_spearman", "valid_weighted_spearman", "n_valid_groups", "n_skipped_groups"):
        assert key in metrics
    # spec §5.6 rule 4: binary-label Spearman reported separately.
    assert "valid_binary_macro_spearman" in metrics
    assert "valid_experimental_macro_spearman" in metrics

    # toy_records has 5 groups total: 1 binary group, 4 experimental groups
    # (one of which has a single label and is always skipped). The model's
    # score here is a deterministic function of the antibody sequence alone,
    # which is constant within every toy group, so the Spearman correlation
    # may legitimately be NaN (skipped) for some/all groups -- only check
    # that valid+skipped accounts for every group.
    assert metrics["n_valid_groups"] + metrics["n_skipped_groups"] == 5.0
    assert metrics["valid_binary_n_valid_groups"] + metrics["valid_binary_n_skipped_groups"] == 1.0
    assert metrics["valid_experimental_n_valid_groups"] + metrics["valid_experimental_n_skipped_groups"] == 4.0


def test_trainer_evaluate_requires_valid_record_metadata(toy_records, antibody_tokenizer, make_fake_encoder):
    trainable = filter_trainable_records(toy_records)
    pairs = build_pairs(trainable, max_pairs_per_group=50, seed=0)

    train_dataloader = DataLoader(
        PairwiseAffinityDataset(trainable, pairs),
        batch_size=4,
        collate_fn=lambda examples: collate_pair_batch(examples, antibody_tokenizer, None),
    )
    valid_dataloader = DataLoader(
        AffinityRecordDataset(trainable),
        batch_size=4,
        collate_fn=lambda examples: collate_rank_batch(examples, antibody_tokenizer, None),
    )
    model = AffinityRanker(
        antibody_encoder=make_fake_encoder(D_MODEL), antigen_encoder=None, d_model=D_MODEL, use_cross_attention=False
    )
    trainer = Trainer(model=model, config=_make_config(epochs=1), train_dataloader=train_dataloader)

    with pytest.raises(ValueError):
        trainer.evaluate(valid_dataloader)


# ── §5.7 rule 5: NaN loss stops training and saves error context ───────────


def test_trainer_nan_loss_raises_and_saves_error_context(
    tmp_path, toy_records, antibody_tokenizer, make_fake_encoder, monkeypatch
):
    trainer, _ = _build_trainer(toy_records, antibody_tokenizer, make_fake_encoder, epochs=1, output_dir=tmp_path)
    monkeypatch.setattr(
        "affinity_transformer.trainer.ranknet_loss",
        lambda *args, **kwargs: torch.tensor(float("nan")),
    )

    with pytest.raises(RuntimeError):
        trainer.fit()

    error_path = tmp_path / "error_context.json"
    assert error_path.exists()
    context = json.loads(error_path.read_text())
    assert "left_record_ids" in context
    assert "right_record_ids" in context
    assert context["epoch"] == 1


# ── §5.7 rule 6: early stopping ──────────────────────────────────────────────


def test_trainer_early_stopping_stops_after_patience(toy_records, antibody_tokenizer, make_fake_encoder):
    trainer, _ = _build_trainer(
        toy_records, antibody_tokenizer, make_fake_encoder, epochs=5, early_stopping_patience=1
    )
    n_batches = len(trainer.train_dataloader)

    # A constant validation metric never "improves" after the first epoch.
    trainer.evaluate = lambda dataloader: {
        "valid_macro_spearman": 0.5,
        "valid_weighted_spearman": 0.5,
        "n_valid_groups": 4.0,
        "n_skipped_groups": 1.0,
    }

    trainer.fit()

    assert trainer.global_step == 2 * n_batches


# ── build_model_and_tokenizers / _resolve_esm2 (spec §5.7 mapping) ──────────


def test_resolve_esm2_returns_hf_repo_for_supported_name():
    assert _resolve_esm2("esm2_t12_35M", "antibody_encoder", ESM2_D_MODEL["esm2_t12_35M"]) == "facebook/esm2_t12_35M_UR50D"


def test_resolve_esm2_unsupported_name_raises():
    with pytest.raises(ValueError):
        _resolve_esm2("ablang2", "antibody_encoder", 480)


def test_resolve_esm2_d_model_mismatch_raises():
    with pytest.raises(ValueError):
        _resolve_esm2("esm2_t12_35M", "antibody_encoder", 999)


def test_build_model_and_tokenizers_unsupported_antibody_encoder_raises_before_network():
    model_config = ModelConfig(antibody_encoder="ablang2", antigen_encoder=None, d_model=480, use_cross_attention=False)

    with pytest.raises(ValueError):
        build_model_and_tokenizers(model_config)


def test_build_model_and_tokenizers_unsupported_antigen_encoder_raises_before_network():
    model_config = ModelConfig(
        antibody_encoder="esm2_t12_35M", antigen_encoder="ablang2", d_model=480, use_cross_attention=False
    )

    with pytest.raises(ValueError):
        build_model_and_tokenizers(model_config)


def test_build_model_and_tokenizers_d_model_mismatch_raises_before_network():
    model_config = ModelConfig(antibody_encoder="esm2_t12_35M", antigen_encoder=None, d_model=999, use_cross_attention=False)

    with pytest.raises(ValueError):
        build_model_and_tokenizers(model_config)
