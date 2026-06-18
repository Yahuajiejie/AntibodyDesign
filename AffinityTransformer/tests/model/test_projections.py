"""Tests for embedding-to-interaction token projections."""

import pytest
import torch

from affinity_transformer.model.projections import TokenProjection


def test_projection_changes_width_and_zeros_padding():
    projection = TokenProjection(input_dim=5, output_dim=8)
    embeddings = torch.randn(2, 3, 5)
    mask = torch.tensor([[True, True, False], [True, False, False]])

    output = projection(embeddings, mask)

    assert output.shape == (2, 3, 8)
    assert torch.equal(output[~mask], torch.zeros_like(output[~mask]))


def test_projection_rejects_wrong_input_width():
    with pytest.raises(ValueError, match="dim mismatch"):
        TokenProjection(5, 8)(torch.randn(1, 2, 6), torch.ones(1, 2, dtype=torch.bool))
