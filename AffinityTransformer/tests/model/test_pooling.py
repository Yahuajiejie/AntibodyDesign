"""Tests for mask-aware model pooling."""

import torch

from affinity_transformer.model.pooling import masked_mean_pool


def test_masked_mean_pool_ignores_padding_tokens():
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
    mask = torch.tensor([[True, True, False]])

    pooled = masked_mean_pool(hidden, mask)

    torch.testing.assert_close(pooled, torch.tensor([[2.0, 3.0]]))


def test_masked_mean_pool_returns_zero_for_all_missing_row():
    hidden = torch.randn(1, 3, 4)
    mask = torch.zeros(1, 3, dtype=torch.bool)

    pooled = masked_mean_pool(hidden, mask)

    torch.testing.assert_close(pooled, torch.zeros(1, 4))
    assert torch.isfinite(pooled).all()
