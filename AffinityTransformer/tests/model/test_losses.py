"""Tests for affinity_transformer.model.losses (spec v0.65 §5.5)."""

from __future__ import annotations

import math

import pytest
import torch

from affinity_transformer.model.losses import (
    listnet_loss,
    pointwise_ranking_loss,
    ranknet_loss,
)


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


def test_pointwise_loss_prefers_scores_matching_rank_targets():
    targets = torch.tensor([0.0, 0.5, 1.0])
    matching = pointwise_ranking_loss(targets, targets)
    reversed_scores = pointwise_ranking_loss(torch.flip(targets, dims=(0,)), targets)

    assert matching < reversed_scores


def test_listnet_rewards_correct_group_ordering():
    labels = torch.tensor([[0.0, 1.0, 2.0]])
    mask = torch.ones_like(labels, dtype=torch.bool)

    correct = listnet_loss(torch.tensor([[0.0, 1.0, 2.0]]), labels, mask)
    reversed_scores = listnet_loss(torch.tensor([[2.0, 1.0, 0.0]]), labels, mask)

    assert correct < reversed_scores


def test_listnet_preserves_ties_and_is_jointly_permutation_invariant():
    scores = torch.tensor([[2.0, 2.0, 0.0]])
    labels = torch.tensor([[1.0, 1.0, 0.0]])
    mask = torch.ones_like(labels, dtype=torch.bool)
    permutation = torch.tensor([2, 0, 1])

    original = listnet_loss(scores, labels, mask)
    permuted = listnet_loss(scores[:, permutation], labels[:, permutation], mask[:, permutation])

    torch.testing.assert_close(original, permuted)


def test_listnet_padding_does_not_change_loss():
    base_scores = torch.tensor([[0.0, 1.0]])
    base_labels = torch.tensor([[0.0, 1.0]])
    base_mask = torch.tensor([[True, True]])
    padded_scores = torch.tensor([[0.0, 1.0, 1000.0]])
    padded_labels = torch.tensor([[0.0, 1.0, 1000.0]])
    padded_mask = torch.tensor([[True, True, False]])

    base = listnet_loss(base_scores, base_labels, base_mask)
    padded = listnet_loss(padded_scores, padded_labels, padded_mask)

    torch.testing.assert_close(base, padded)


def test_listnet_skips_constant_groups_but_requires_one_trainable_group():
    scores = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    mask = torch.ones_like(labels, dtype=torch.bool)

    loss = listnet_loss(scores, labels, mask)
    assert torch.isfinite(loss)

    with pytest.raises(ValueError, match="no group"):
        listnet_loss(scores[:1], labels[:1], mask[:1])
