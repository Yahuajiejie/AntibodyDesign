"""Tests for the v0.65 embedding-native affinity ranker."""

import pytest
import torch

from affinity_transformer.embeddings import EmbeddingBatch
from affinity_transformer.model import EmbeddingAffinityRanker


def _batch(*, include_antigen: bool = True) -> EmbeddingBatch:
    antibody_mask = torch.tensor([[True, True, False], [True, True, True]])
    antigen_mask = torch.tensor([[True, True, False, False], [False, False, False, False]])
    return EmbeddingBatch(
        antibody_embeddings=torch.randn(2, 3, 5),
        antibody_mask=antibody_mask,
        antigen_embeddings=torch.randn(2, 4, 7) if include_antigen else None,
        antigen_mask=antigen_mask if include_antigen else None,
        labels=torch.tensor([2.0, 1.0]),
        record_ids=["a", "b"],
        group_ids=["g", "g"],
    )


def test_antibody_only_scores_embedding_batch_without_base_encoder():
    model = EmbeddingAffinityRanker(
        antibody_input_dim=5,
        antigen_input_dim=None,
        d_model=8,
        fusion_kind="antibody_only",
        dropout=0.0,
    )

    scores = model(_batch())

    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()
    assert not any("encoder" in name for name, _ in model.named_parameters())


@pytest.mark.parametrize("pooling", ["masked_mean", "attention_pool"])
def test_concat_supports_independent_dims_pooling_and_backward(pooling):
    model = EmbeddingAffinityRanker(
        antibody_input_dim=5,
        antigen_input_dim=7,
        d_model=8,
        fusion_kind="concat",
        pooling=pooling,
        dropout=0.0,
    )

    features = model.forward_features(_batch())
    scores = model(_batch())
    scores.sum().backward()

    assert features.shape == (2, 16)
    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()
    assert model.interaction is None
    assert not any("encoder" in name for name, _ in model.named_parameters())
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_concat_uses_zero_antigen_representation_for_missing_antigen():
    model = EmbeddingAffinityRanker(
        antibody_input_dim=5,
        antigen_input_dim=7,
        d_model=8,
        fusion_kind="concat",
        pooling="masked_mean",
        dropout=0.0,
    )

    mixed_features = model.forward_features(_batch())
    all_missing_scores = model(_batch(include_antigen=False))

    torch.testing.assert_close(mixed_features[1, 8:], torch.zeros(8))
    assert all_missing_scores.shape == (2,)
    assert torch.isfinite(all_missing_scores).all()


@pytest.mark.parametrize("num_layers", [4, 8, 16])
def test_deep_cross_attention_handles_mixed_missing_antigen_rows(num_layers):
    model = EmbeddingAffinityRanker(
        antibody_input_dim=5,
        antigen_input_dim=7,
        d_model=8,
        fusion_kind="deep_cross_attention",
        num_layers=num_layers,
        num_heads=2,
        dropout=0.0,
    )

    scores = model(_batch())
    scores.sum().backward()

    assert torch.isfinite(scores).all()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_antigen_model_accepts_whole_batch_missing_antigen():
    model = EmbeddingAffinityRanker(
        antibody_input_dim=5,
        antigen_input_dim=7,
        d_model=8,
        fusion_kind="deep_cross_attention",
        num_layers=4,
        num_heads=2,
        dropout=0.0,
    )

    scores = model(_batch(include_antigen=False))

    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()


def test_missing_antigen_score_does_not_depend_on_other_batch_rows():
    torch.manual_seed(0)
    model = EmbeddingAffinityRanker(
        antibody_input_dim=5,
        antigen_input_dim=7,
        d_model=8,
        fusion_kind="deep_cross_attention",
        num_layers=4,
        num_heads=2,
        dropout=0.0,
    ).eval()
    mixed = _batch()
    missing_only = EmbeddingBatch(
        antibody_embeddings=mixed.antibody_embeddings[1:2],
        antibody_mask=mixed.antibody_mask[1:2],
        antigen_embeddings=None,
        antigen_mask=None,
        labels=mixed.labels[1:2],
        record_ids=[mixed.record_ids[1]],
        group_ids=[mixed.group_ids[1]],
    )

    mixed_score = model(mixed)[1]
    isolated_score = model(missing_only)[0]

    torch.testing.assert_close(mixed_score, isolated_score)


def test_architecture_validation_rejects_inconsistent_fusion_settings():
    with pytest.raises(ValueError, match="num_layers=0"):
        EmbeddingAffinityRanker(
            antibody_input_dim=5,
            antigen_input_dim=7,
            d_model=8,
            fusion_kind="concat",
            num_layers=2,
        )
    with pytest.raises(ValueError, match="num_layers >= 1"):
        EmbeddingAffinityRanker(
            antibody_input_dim=5,
            antigen_input_dim=7,
            d_model=8,
            fusion_kind="deep_cross_attention",
            num_layers=0,
        )
