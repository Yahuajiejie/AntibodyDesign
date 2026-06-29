"""Tests for deep mask-safe antibody-antigen interaction blocks."""

import pytest
import torch
import torch.nn as nn

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
        first_parameter_ids = {id(next(layer.parameters())) for layer in interaction.layers}
        assert len(first_parameter_ids) == num_layers


class _RecordingAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.key_padding_masks = []

    def forward(self, query, key, value, key_padding_mask, need_weights):
        self.key_padding_masks.append(key_padding_mask.detach().clone())
        return torch.zeros_like(query), None


def test_every_interaction_layer_uses_the_correct_key_padding_mask():
    interaction = DeepCrossAttention(8, 4, 2, dropout=0.0, bidirectional=True)
    antibody = torch.randn(2, 3, 8)
    antigen = torch.randn(2, 4, 8)
    antibody_mask = torch.tensor([[True, True, False], [True, False, False]])
    antigen_mask = torch.tensor([[True, True, False, False], [True, False, True, False]])
    recorders = []
    for layer in interaction.layers:
        ab_queries_ag = _RecordingAttention()
        ag_queries_ab = _RecordingAttention()
        layer.antibody_to_antigen = ab_queries_ag
        layer.antigen_to_antibody = ag_queries_ab
        recorders.append((ab_queries_ag, ag_queries_ab))

    interaction(antibody, antigen, antibody_mask, antigen_mask)

    for ab_queries_ag, ag_queries_ab in recorders:
        torch.testing.assert_close(ab_queries_ag.key_padding_masks[0], ~antigen_mask)
        torch.testing.assert_close(ag_queries_ab.key_padding_masks[0], ~antibody_mask)


@pytest.mark.parametrize("num_layers", [4, 8, 16])
def test_every_layer_rezeros_padding_positions(num_layers):
    interaction = DeepCrossAttention(8, num_layers, 2, dropout=0.0)
    antibody, antigen, antibody_mask, antigen_mask = _streams()
    layer_outputs = []
    hooks = [
        layer.register_forward_hook(
            lambda module, inputs, output: layer_outputs.append(
                (output[0].detach().clone(), output[1].detach().clone())
            )
        )
        for layer in interaction.layers
    ]

    antibody_output, antigen_output = interaction(
        antibody, antigen, antibody_mask, antigen_mask
    )
    for hook in hooks:
        hook.remove()

    assert len(layer_outputs) == num_layers
    for layer_antibody, layer_antigen in layer_outputs:
        assert not layer_antibody[~antibody_mask].any()
        assert not layer_antigen[~antigen_mask].any()
        assert torch.isfinite(layer_antibody).all()
        assert torch.isfinite(layer_antigen).all()
    assert torch.isfinite(antibody_output).all()
    assert torch.isfinite(antigen_output).all()


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
