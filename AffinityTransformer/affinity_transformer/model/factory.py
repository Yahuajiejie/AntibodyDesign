"""Construction of trainable embedding-native rankers."""

from __future__ import annotations

from ..config import EncoderConfig, ModelConfig
from ..embeddings.validation import CacheDescriptor
from .embedding_ranker import EmbeddingAffinityRanker


def build_ranker(
    model_config: ModelConfig,
    antibody_cache: CacheDescriptor,
    antigen_cache: CacheDescriptor | None = None,
) -> EmbeddingAffinityRanker:
    """Build projection/interaction/scorer modules from validated cache facts.

    No foundation model, tokenizer, cache shard, or ``transformers`` module is
    constructed here. Callers must validate caches before invoking this
    function and before constructing the Trainer/optimizer.
    """
    _validate_descriptor(
        model_config.antibody_encoder,
        antibody_cache,
        expected_sequence_type="antibody",
    )
    interaction = model_config.interaction
    if interaction.kind == "antibody_only":
        if antigen_cache is not None:
            raise ValueError("antibody_only build_ranker does not accept an antigen cache")
        antigen_input_dim = None
    else:
        if model_config.antigen_encoder is None or antigen_cache is None:
            raise ValueError(f"{interaction.kind} build_ranker requires an antigen cache")
        _validate_descriptor(
            model_config.antigen_encoder,
            antigen_cache,
            expected_sequence_type="antigen",
        )
        antigen_input_dim = antigen_cache.embedding_dim

    return EmbeddingAffinityRanker(
        antibody_input_dim=antibody_cache.embedding_dim,
        antigen_input_dim=antigen_input_dim,
        d_model=interaction.d_model,
        fusion_kind=interaction.kind,
        num_layers=interaction.num_layers,
        num_heads=interaction.num_heads,
        ffn_multiplier=interaction.ffn_multiplier,
        dropout=interaction.dropout,
        pooling=interaction.pooling,
        bidirectional=interaction.bidirectional,
    )


def _validate_descriptor(
    encoder: EncoderConfig,
    descriptor: CacheDescriptor,
    *,
    expected_sequence_type: str,
) -> None:
    if encoder.mode != "frozen_cached":
        raise ValueError("build_ranker requires frozen_cached encoder configs")
    if encoder.cache_dir is None or descriptor.cache_dir.resolve() != encoder.cache_dir.resolve():
        raise ValueError("cache descriptor path does not match EncoderConfig.cache_dir")
    if descriptor.sequence_type != expected_sequence_type:
        raise ValueError(
            f"expected {expected_sequence_type} cache, got {descriptor.sequence_type}"
        )
    if descriptor.encoder_name != encoder.name:
        raise ValueError(
            f"cache encoder_name mismatch: {descriptor.encoder_name!r} != {encoder.name!r}"
        )
    if descriptor.encoder_revision != encoder.revision:
        raise ValueError(
            f"cache encoder_revision mismatch: "
            f"{descriptor.encoder_revision!r} != {encoder.revision!r}"
        )
    if descriptor.tokenizer_revision != encoder.tokenizer_revision:
        raise ValueError(
            f"cache tokenizer_revision mismatch: "
            f"{descriptor.tokenizer_revision!r} != {encoder.tokenizer_revision!r}"
        )
    if descriptor.embedding_dim < 1:
        raise ValueError("cache embedding_dim must be positive")
