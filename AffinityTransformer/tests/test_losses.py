"""Tests for affinity_transformer.losses (spec §5.5)."""

from __future__ import annotations

import math

import pytest
import torch

from affinity_transformer.losses import ranknet_loss


def test_ranknet_loss_rewards_correct_ordering():
    """spec §5.5 acceptance test."""
    correct = ranknet_loss(
        score_i=torch.tensor(2.0), score_j=torch.tensor(1.0), y_ij=torch.tensor(1.0)
    )
    incorrect = ranknet_loss(
        score_i=torch.tensor(1.0), score_j=torch.tensor(2.0), y_ij=torch.tensor(1.0)
    )

    assert correct < incorrect


def test_ranknet_loss_equal_scores_is_log2_regardless_of_label():
    equal_i, equal_j = torch.tensor(0.5), torch.tensor(0.5)

    loss_y1 = ranknet_loss(equal_i, equal_j, torch.tensor(1.0))
    loss_y0 = ranknet_loss(equal_i, equal_j, torch.tensor(0.0))

    assert loss_y1.item() == pytest.approx(math.log(2.0))
    assert loss_y0.item() == pytest.approx(math.log(2.0))


def test_ranknet_loss_sigma_sharpens_correct_prediction():
    score_i, score_j, y_ij = torch.tensor(1.0), torch.tensor(0.0), torch.tensor(1.0)

    loss_sigma1 = ranknet_loss(score_i, score_j, y_ij, sigma=1.0)
    loss_sigma2 = ranknet_loss(score_i, score_j, y_ij, sigma=2.0)

    assert loss_sigma2 < loss_sigma1


def test_ranknet_loss_swapping_sides_and_label_is_equivalent():
    score_a, score_b = torch.tensor(2.0), torch.tensor(0.5)

    loss_ab = ranknet_loss(score_a, score_b, torch.tensor(1.0))
    loss_ba = ranknet_loss(score_b, score_a, torch.tensor(0.0))

    assert loss_ab.item() == pytest.approx(loss_ba.item())


def test_ranknet_loss_supports_batches():
    score_i = torch.tensor([2.0, 1.0, 0.5])
    score_j = torch.tensor([1.0, 2.0, 0.5])
    y_ij = torch.tensor([1.0, 1.0, 0.0])

    loss = ranknet_loss(score_i, score_j, y_ij)

    assert loss.shape == ()
    assert loss.item() > 0.0
