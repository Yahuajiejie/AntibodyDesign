"""Tests for affinity_transformer.user_entry (spec §5.8)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from affinity_transformer.config import Config, DataConfig, ModelConfig, TrainConfig
from affinity_transformer.model import AffinityRanker
from affinity_transformer.user_entry import (
    AntibodyInput,
    load_model,
    rank_antibodies,
    score_antibodies,
)

D_MODEL = 16

_HEAVY_1 = "QVQLVQSGAEVKKPGASVKVSCKAS"
_HEAVY_2 = "QVKLEESGGGLVQAGGSLRLSCAAS"
_HEAVY_3 = "EVQLVESGGGLVQPGGSLRLSCAAS"


def _make_model(make_fake_encoder, antibody_tokenizer, antigen_tokenizer=None):
    model = AffinityRanker(
        antibody_encoder=make_fake_encoder(D_MODEL),
        antigen_encoder=None,
        d_model=D_MODEL,
        use_cross_attention=False,
    )
    model.antibody_tokenizer = antibody_tokenizer
    model.antigen_tokenizer = antigen_tokenizer
    return model


def _config():
    return Config(
        data=DataConfig(train_path=Path("unused.parquet"), valid_path=None, max_pairs_per_group=50, seed=0),
        model=ModelConfig(antibody_encoder="fake", antigen_encoder=None, d_model=D_MODEL, use_cross_attention=False),
        train=TrainConfig(batch_size=4, lr=1.0e-3, epochs=1, device="cpu"),
    )


# ── score_antibodies ─────────────────────────────────────────────────────────


def test_score_antibodies_returns_expected_columns_with_missing_antigen(antibody_tokenizer, make_fake_encoder):
    """spec §5.8 rule 4: no antigen sequence is allowed."""
    model = _make_model(make_fake_encoder, antibody_tokenizer)
    antibodies = [
        AntibodyInput("ab1", heavy_chain=_HEAVY_1, light_chain=None, single_chain_sequence=None, antibody_type="VHH"),
        AntibodyInput("ab2", heavy_chain=_HEAVY_2, light_chain=None, single_chain_sequence=None, antibody_type="VHH"),
    ]

    result = score_antibodies(None, antibodies, model)

    assert list(result.columns) == ["antibody_id", "score", "rank"]
    assert result["antibody_id"].tolist() == ["ab1", "ab2"]
    assert result["score"].notna().all()
    assert set(result["rank"].tolist()) <= {1, 2}


def test_score_antibodies_requires_nonempty_antibodies(antibody_tokenizer, make_fake_encoder):
    model = _make_model(make_fake_encoder, antibody_tokenizer)

    with pytest.raises(ValueError):
        score_antibodies(None, [], model)


def test_score_antibodies_rejects_invalid_antigen_sequence(antibody_tokenizer, make_fake_encoder):
    model = _make_model(make_fake_encoder, antibody_tokenizer)
    antibodies = [
        AntibodyInput("ab1", heavy_chain=_HEAVY_1, light_chain=None, single_chain_sequence=None, antibody_type="VHH"),
    ]

    with pytest.raises(ValueError):
        score_antibodies("NOT-A-SEQUENCE", antibodies, model)


def test_score_antibodies_requires_antibody_tokenizer_attribute(make_fake_encoder):
    model = AffinityRanker(
        antibody_encoder=make_fake_encoder(D_MODEL), antigen_encoder=None, d_model=D_MODEL, use_cross_attention=False
    )
    antibodies = [
        AntibodyInput("ab1", heavy_chain=_HEAVY_1, light_chain=None, single_chain_sequence=None, antibody_type="VHH"),
    ]

    with pytest.raises(ValueError):
        score_antibodies(None, antibodies, model)


def test_score_antibodies_rejects_unsupported_antibody_type(antibody_tokenizer, make_fake_encoder):
    model = _make_model(make_fake_encoder, antibody_tokenizer)
    antibodies = [
        AntibodyInput("ab1", heavy_chain=_HEAVY_1, light_chain=None, single_chain_sequence=None, antibody_type="IgG"),
    ]

    with pytest.raises(ValueError):
        score_antibodies(None, antibodies, model)


def test_score_antibodies_rejects_invalid_chain_sequence(antibody_tokenizer, make_fake_encoder):
    model = _make_model(make_fake_encoder, antibody_tokenizer)
    antibodies = [
        AntibodyInput("ab1", heavy_chain="QVQLVQSGX1Z", light_chain=None, single_chain_sequence=None, antibody_type="VHH"),
    ]

    with pytest.raises(ValueError):
        score_antibodies(None, antibodies, model)


def test_score_antibodies_requires_usable_antibody_sequence(antibody_tokenizer, make_fake_encoder):
    model = _make_model(make_fake_encoder, antibody_tokenizer)
    antibodies = [
        AntibodyInput("ab1", heavy_chain=None, light_chain=None, single_chain_sequence=None, antibody_type="VHH"),
    ]

    with pytest.raises(ValueError):
        score_antibodies(None, antibodies, model)


# ── rank_antibodies (spec §7.2: "returns descending ranks") ─────────────────


def test_rank_antibodies_returns_descending_ranks(antibody_tokenizer, make_fake_encoder):
    model = _make_model(make_fake_encoder, antibody_tokenizer)
    antibodies = [
        AntibodyInput("ab1", heavy_chain=_HEAVY_1, light_chain=None, single_chain_sequence=None, antibody_type="VHH"),
        AntibodyInput("ab2", heavy_chain=_HEAVY_2, light_chain=None, single_chain_sequence=None, antibody_type="VHH"),
        AntibodyInput("ab3", heavy_chain=_HEAVY_3, light_chain=None, single_chain_sequence=None, antibody_type="VHH"),
    ]

    scored = score_antibodies("MKTAYIAKQRQISFVKSHFSRQLE", antibodies, model)
    ranked = rank_antibodies("MKTAYIAKQRQISFVKSHFSRQLE", antibodies, model)

    assert list(ranked.columns) == ["antibody_id", "score", "rank"]
    assert set(ranked["antibody_id"]) == set(scored["antibody_id"])

    scores = ranked["score"].tolist()
    assert scores == sorted(scores, reverse=True)

    ranks = ranked["rank"].tolist()
    assert ranks == sorted(ranks)
    assert ranks[0] == 1


# ── load_model ────────────────────────────────────────────────────────────────


def test_load_model_attaches_tokenizers_and_loads_weights(tmp_path, antibody_tokenizer, make_fake_encoder, monkeypatch):
    trained_model = AffinityRanker(
        antibody_encoder=make_fake_encoder(D_MODEL), antigen_encoder=None, d_model=D_MODEL, use_cross_attention=False
    )
    with torch.no_grad():
        for param in trained_model.parameters():
            param.add_(1.0)

    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": trained_model.state_dict(),
            "config": _config(),
            "global_step": 5,
            "seed": 0,
        },
        checkpoint_path,
    )

    def fake_build_model_and_tokenizers(model_config):
        fresh_model = AffinityRanker(
            antibody_encoder=make_fake_encoder(model_config.d_model),
            antigen_encoder=None,
            d_model=model_config.d_model,
            use_cross_attention=model_config.use_cross_attention,
        )
        return fresh_model, antibody_tokenizer, None

    monkeypatch.setattr(
        "affinity_transformer.user_entry.build_model_and_tokenizers", fake_build_model_and_tokenizers
    )

    loaded = load_model(checkpoint_path)

    assert loaded.antibody_tokenizer is antibody_tokenizer
    assert loaded.antigen_tokenizer is None
    assert loaded.training is False
    for key, value in trained_model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])
