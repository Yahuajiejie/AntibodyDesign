"""Tests for affinity_transformer.user_entry (spec §5.8)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from affinity_transformer.config import (
    Config,
    DataConfig,
    EncoderConfig,
    InteractionConfig,
    ModelConfig,
    ObjectiveConfig,
    TrainConfig,
)
from affinity_transformer.model import AffinityRanker
from affinity_transformer.user_entry import (
    AffinityPredictor,
    AntibodyInput,
    load_model,
    rank_antibodies,
    rank_antibodies_with_predictor,
    rank_antibody_table_with_predictor,
    score_antibodies_with_predictor,
)

D_MODEL = 16

_HEAVY_1 = "QVQLVQSGAEVKKPGASVKVSCKAS"
_HEAVY_2 = "QVKLEESGGGLVQAGGSLRLSCAAS"
_HEAVY_3 = "EVQLVESGGGLVQPGGSLRLSCAAS"
_LIGHT = "DIQMTQSPSSLSASVGDRVTITC"
_ANTIGEN = "MKTAYIAKQRQISFVKSHFSRQLE"


def _config():
    return Config(
        data=DataConfig(
            train_path=Path("unused.parquet"),
            valid_path=None,
            max_pairs_per_group=50,
            seed=0,
        ),
        model=ModelConfig(
            antibody_encoder=EncoderConfig(
                name="fake", revision="main", tokenizer_revision="main",
                mode="frozen_online", embedding_layer=-1, cache_dir=None,
                max_length=None, long_sequence_strategy="error",
            ),
            antigen_encoder=None,
            interaction=InteractionConfig(
                kind="antibody_only", d_model=D_MODEL, num_layers=0,
                num_heads=1, ffn_multiplier=4.0, dropout=0.1,
                pooling="masked_mean", bidirectional=False,
            ),
            objective=ObjectiveConfig(
                name="pairwise_ranknet", temperature=1.0, sigma=1.0,
                pointwise_loss="huber",
            ),
        ),
        train=TrainConfig(batch_size=4, lr=1.0e-3, epochs=1, device="cpu"),
    )


def _predictor(make_fake_encoder, antibody_tokenizer, antigen_tokenizer=None):
    model = AffinityRanker(
        antibody_encoder=make_fake_encoder(D_MODEL),
        antigen_encoder=None,
        d_model=D_MODEL,
        use_cross_attention=False,
    )
    return AffinityPredictor(
        model_name="manual",
        model=model,
        config=_config(),
        antibody_tokenizer=antibody_tokenizer,
        antigen_tokenizer=antigen_tokenizer,
        checkpoint_path=Path("manual.pt"),
    )


def test_score_antibodies_with_predictor_accepts_igg_fab_and_unknown(
    antibody_tokenizer, make_fake_encoder
):
    predictor = _predictor(make_fake_encoder, antibody_tokenizer)
    antibodies = [
        AntibodyInput("igg", _HEAVY_1, _LIGHT, None, "IgG"),
        AntibodyInput("fab", _HEAVY_2, _LIGHT, None, "Fab"),
        AntibodyInput("unk", _HEAVY_3, None, None, "unknown"),
    ]

    result = score_antibodies_with_predictor(None, antibodies, predictor, query_id="q1")

    assert list(result.columns) == ["query_id", "antibody_id", "score", "rank", "model_name"]
    assert result["query_id"].unique().tolist() == ["q1"]
    assert set(result["antibody_id"]) == {"igg", "fab", "unk"}
    assert result["score"].notna().all()


def test_score_antibodies_with_predictor_rejects_invalid_sequence(
    antibody_tokenizer, make_fake_encoder
):
    predictor = _predictor(make_fake_encoder, antibody_tokenizer)
    antibodies = [AntibodyInput("bad", "QVQLVQSGX1Z", None, None, "VHH")]

    with pytest.raises(ValueError):
        score_antibodies_with_predictor(_ANTIGEN, antibodies, predictor)


def test_rank_antibodies_uses_model_name_loader(monkeypatch, antibody_tokenizer, make_fake_encoder):
    predictor = _predictor(make_fake_encoder, antibody_tokenizer)
    monkeypatch.setattr("affinity_transformer.user_entry.load_predictor", lambda model_name: predictor)
    antibodies = [
        AntibodyInput("ab1", _HEAVY_1, None, None, "VHH"),
        AntibodyInput("ab2", _HEAVY_2, None, None, "VHH"),
    ]

    result = rank_antibodies(_ANTIGEN, antibodies, model_name="best")

    assert result["model_name"].unique().tolist() == ["manual"]
    assert result["score"].tolist() == sorted(result["score"].tolist(), reverse=True)


def test_rank_antibody_table_ranks_within_query_id_only(antibody_tokenizer, make_fake_encoder):
    predictor = _predictor(make_fake_encoder, antibody_tokenizer)
    table = pd.DataFrame({
        "query_id": ["q1", "q1", "q2"],
        "antibody_id": ["a", "b", "a"],
        "antigen_sequence": [_ANTIGEN, _ANTIGEN, None],
        "heavy_chain": [_HEAVY_1, _HEAVY_2, _HEAVY_3],
        "light_chain": [None, None, None],
        "single_chain_sequence": [None, None, None],
        "antibody_type": ["VHH", "VHH", "IgG"],
    })

    result = rank_antibody_table_with_predictor(table, predictor)

    assert set(result["query_id"]) == {"q1", "q2"}
    assert result[result["query_id"] == "q1"]["rank"].min() == 1
    assert result[result["query_id"] == "q2"]["rank"].tolist() == [1]


def test_rank_antibody_table_rejects_inconsistent_antigen_sequence(
    antibody_tokenizer, make_fake_encoder
):
    predictor = _predictor(make_fake_encoder, antibody_tokenizer)
    table = pd.DataFrame({
        "query_id": ["q1", "q1"],
        "antibody_id": ["a", "b"],
        "antigen_sequence": [_ANTIGEN, "MSTNPKPQRKTKRNTNRRPQ"],
        "heavy_chain": [_HEAVY_1, _HEAVY_2],
        "light_chain": [None, None],
        "single_chain_sequence": [None, None],
        "antibody_type": ["VHH", "VHH"],
    })

    with pytest.raises(ValueError, match="inconsistent antigen_sequence"):
        rank_antibody_table_with_predictor(table, predictor)


def test_rank_antibodies_with_predictor_returns_descending_scores(
    antibody_tokenizer, make_fake_encoder
):
    predictor = _predictor(make_fake_encoder, antibody_tokenizer)
    antibodies = [
        AntibodyInput("ab1", _HEAVY_1, None, None, "VHH"),
        AntibodyInput("ab2", _HEAVY_2, None, None, "VHH"),
        AntibodyInput("ab3", _HEAVY_3, None, None, "VHH"),
    ]

    ranked = rank_antibodies_with_predictor(_ANTIGEN, antibodies, predictor)

    assert ranked["score"].tolist() == sorted(ranked["score"].tolist(), reverse=True)
    assert ranked["rank"].iloc[0] == 1


def test_load_model_returns_bare_model(tmp_path, antibody_tokenizer, make_fake_encoder, monkeypatch):
    trained_model = AffinityRanker(
        antibody_encoder=make_fake_encoder(D_MODEL),
        antigen_encoder=None,
        d_model=D_MODEL,
        use_cross_attention=False,
    )
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
        "affinity_transformer.user_entry.build_model_and_tokenizers",
        fake_build_model_and_tokenizers,
    )

    loaded = load_model(checkpoint_path)

    assert not hasattr(loaded, "antibody_tokenizer")
    assert loaded.training is False
    for key, value in trained_model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])
