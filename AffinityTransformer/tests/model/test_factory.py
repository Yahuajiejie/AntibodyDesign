"""Tests for the embedding-native model factory."""

from pathlib import Path

import pytest

from affinity_transformer.config import (
    EncoderConfig,
    InteractionConfig,
    ModelConfig,
    ObjectiveConfig,
)
from affinity_transformer.embeddings import CacheDescriptor
from affinity_transformer.model import EmbeddingAffinityRanker, build_ranker


def _encoder(name: str, cache_dir: Path) -> EncoderConfig:
    return EncoderConfig(
        name=name,
        revision=f"{name}-revision",
        tokenizer_revision=f"{name}-tokenizer",
        mode="frozen_cached",
        embedding_layer=-1,
        cache_dir=cache_dir,
        max_length=None,
        long_sequence_strategy="error",
    )


def _descriptor(
    cache_dir: Path,
    name: str,
    sequence_type: str,
    embedding_dim: int,
) -> CacheDescriptor:
    return CacheDescriptor(
        cache_dir=cache_dir,
        manifest_path=cache_dir / "manifest.parquet",
        metadata_path=cache_dir / "metadata.yaml",
        sequence_type=sequence_type,
        encoder_name=name,
        encoder_revision=f"{name}-revision",
        tokenizer_revision=f"{name}-tokenizer",
        embedding_dim=embedding_dim,
        dtype="float32",
        metadata_hash=f"{name}-hash",
        n_items=2,
        required_count=2,
        covered_count=2,
    )


def _model_config(tmp_path: Path, kind: str, num_layers: int) -> ModelConfig:
    antibody = _encoder("ab", tmp_path / "ab")
    antigen = None if kind == "antibody_only" else _encoder("ag", tmp_path / "ag")
    return ModelConfig(
        antibody_encoder=antibody,
        antigen_encoder=antigen,
        interaction=InteractionConfig(
            kind=kind,
            d_model=8,
            num_layers=num_layers,
            num_heads=2,
            ffn_multiplier=4.0,
            dropout=0.0,
            pooling="masked_mean",
            bidirectional=True,
        ),
        objective=ObjectiveConfig(
            name="pairwise_ranknet",
            temperature=1.0,
            sigma=1.0,
            pointwise_loss="huber",
        ),
    )


def test_build_ranker_constructs_concat_with_independent_input_dims(tmp_path: Path):
    config = _model_config(tmp_path, "concat", 0)

    model = build_ranker(
        config,
        _descriptor(tmp_path / "ab", "ab", "antibody", 5),
        _descriptor(tmp_path / "ag", "ag", "antigen", 7),
    )

    assert isinstance(model, EmbeddingAffinityRanker)
    assert model.fusion_kind == "concat"
    assert model.interaction is None
    assert not any("encoder" in name for name, _ in model.named_parameters())


@pytest.mark.parametrize("num_layers", [4, 8, 16])
def test_build_ranker_constructs_requested_deep_stack(tmp_path: Path, num_layers: int):
    config = _model_config(tmp_path, "deep_cross_attention", num_layers)

    model = build_ranker(
        config,
        _descriptor(tmp_path / "ab", "ab", "antibody", 5),
        _descriptor(tmp_path / "ag", "ag", "antigen", 7),
    )

    assert model.interaction is not None
    assert len(model.interaction.layers) == num_layers


def test_build_ranker_rejects_wrong_cache_role(tmp_path: Path):
    config = _model_config(tmp_path, "antibody_only", 0)

    with pytest.raises(ValueError, match="expected antibody cache"):
        build_ranker(
            config,
            _descriptor(tmp_path / "ab", "ab", "antigen", 5),
        )
