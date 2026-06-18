"""Tests for deep mask-safe antibody-antigen interaction blocks."""

import torch

from affinity_transformer.model.blocks import InteractionBlock
from affinity_transformer.model.interaction import DeepCrossAttention


def _streams():
    antibody = torch.randn(2, 3, 8)
    antigen = torch.randn(2, 4, 8)
    antibody_mask = torch.tensor([[True, True, False], [True, True, True]])
    antigen_mask = torch.tensor([[True, True, False, False], [False, False, False, False]])
    return antibody, antigen, antibody_mask, antigen_mask


def test_interaction_block_masks_padding_and_missing_antigen_without_nan():
    block = InteractionBlock(8, 2, dropout=0.0, bidirectional=True)
    antibody, antigen, antibody_mask, antigen_mask = _streams()

    antibody_output, antigen_output = block(
        antibody, antigen, antibody_mask, antigen_mask
    )

    assert antibody_output.shape == antibody.shape
    assert antigen_output.shape == antigen.shape
    assert torch.isfinite(antibody_output).all()
    assert torch.isfinite(antigen_output).all()
    assert not antibody_output[~antibody_mask].any()
    assert not antigen_output[~antigen_mask].any()


def test_unidirectional_block_leaves_valid_antigen_stream_unchanged():
    block = InteractionBlock(8, 2, dropout=0.0, bidirectional=False)
    antibody, antigen, antibody_mask, antigen_mask = _streams()
    expected_antigen = antigen.masked_fill(~antigen_mask.unsqueeze(-1), 0.0)

    _, antigen_output = block(antibody, antigen, antibody_mask, antigen_mask)

    torch.testing.assert_close(antigen_output, expected_antigen)


def test_deep_cross_attention_constructs_requested_independent_layers():
    for num_layers in (4, 8, 16):
        interaction = DeepCrossAttention(
            d_model=8,
            num_layers=num_layers,
            num_heads=2,
            dropout=0.0,
        )
        assert len(interaction.layers) == num_layers
        assert len({id(layer) for layer in interaction.layers}) == num_layers


def test_interaction_backward_has_finite_gradients():
    interaction = DeepCrossAttention(8, 2, 2, dropout=0.0)
    antibody, antigen, antibody_mask, antigen_mask = _streams()
    antibody_output, antigen_output = interaction(
        antibody, antigen, antibody_mask, antigen_mask
    )

    (antibody_output.sum() + antigen_output.sum()).backward()

    gradients = [parameter.grad for parameter in interaction.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
