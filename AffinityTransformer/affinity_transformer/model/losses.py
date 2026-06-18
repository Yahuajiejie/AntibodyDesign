"""Ranking losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pointwise_ranking_loss(
    scores: torch.Tensor,
    rank_targets: torch.Tensor,
    loss_type: str = "huber",
) -> torch.Tensor:
    """Regress group-normalized rank targets for the pointwise baseline."""
    if scores.shape != rank_targets.shape:
        raise ValueError(
            f"scores and rank_targets must have the same shape: "
            f"{tuple(scores.shape)} != {tuple(rank_targets.shape)}"
        )
    if not torch.isfinite(scores).all() or not torch.isfinite(rank_targets).all():
        raise ValueError("scores and rank_targets must be finite")
    if torch.any((rank_targets < 0) | (rank_targets > 1)):
        raise ValueError("rank_targets must be group-rank values in [0, 1]")
    targets = rank_targets.to(dtype=scores.dtype)
    if loss_type == "huber":
        return F.huber_loss(scores, targets)
    if loss_type == "mse":
        return F.mse_loss(scores, targets)
    raise ValueError(f"unsupported pointwise loss_type {loss_type!r}")


def ranknet_loss(
    score_i: torch.Tensor,
    score_j: torch.Tensor,
    y_ij: torch.Tensor,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Compute the RankNet binary cross-entropy loss from score differences.

    Args:
        score_i: Unbounded scores for the first item.
        score_j: Unbounded scores for the second item, shape-compatible with
            ``score_i``.
        y_ij: Float targets where 1 means item i should rank above item j and
            0 means the reverse.
        sigma: Positive scale applied to the score difference.

    Returns:
        Scalar mean loss.

    Invalid shapes or dtypes are reported by
    ``binary_cross_entropy_with_logits``.
    """
    logits = sigma * (score_i - score_j)
    return F.binary_cross_entropy_with_logits(logits, y_ij)


def listnet_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    member_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute tie-preserving ListNet top-one loss over valid groups.

    Label magnitudes are converted to tie-aware empirical ranks before the
    target softmax, so only within-group order and ties affect the target.
    Groups with fewer than two members or one unique label are skipped.

    Raises:
        ValueError: If shapes/masks are invalid, valid values are non-finite,
            temperature is not positive, or the batch has no trainable group.
    """
    _validate_listwise_inputs(scores, labels, member_mask, temperature)
    valid_groups = listnet_valid_group_mask(labels, member_mask)
    if not valid_groups.any():
        raise ValueError("listnet batch contains no group with at least two distinct labels")

    scores = scores[valid_groups]
    labels = labels[valid_groups]
    member_mask = member_mask[valid_groups]
    rank_targets = _tie_aware_rank_targets(labels, member_mask)
    floor = torch.finfo(scores.dtype).min
    target_logits = (rank_targets / temperature).masked_fill(~member_mask, floor)
    prediction_logits = (scores / temperature).masked_fill(~member_mask, floor)
    target_probabilities = torch.softmax(target_logits, dim=1)
    prediction_log_probabilities = torch.log_softmax(prediction_logits, dim=1)
    per_group = -(target_probabilities * prediction_log_probabilities).masked_fill(
        ~member_mask, 0.0
    ).sum(dim=1)
    return per_group.mean()


def listnet_valid_group_mask(
    labels: torch.Tensor,
    member_mask: torch.Tensor,
) -> torch.Tensor:
    """Return rows containing at least two valid, distinct labels."""
    if labels.ndim != 2 or member_mask.shape != labels.shape or member_mask.dtype != torch.bool:
        raise ValueError("labels/member_mask must be [G, M] with a boolean mask")
    counts = member_mask.sum(dim=1)
    positive_inf = torch.tensor(float("inf"), dtype=labels.dtype, device=labels.device)
    negative_inf = torch.tensor(float("-inf"), dtype=labels.dtype, device=labels.device)
    minimum = labels.masked_fill(~member_mask, positive_inf).min(dim=1).values
    maximum = labels.masked_fill(~member_mask, negative_inf).max(dim=1).values
    return (counts >= 2) & (maximum > minimum)


def _tie_aware_rank_targets(
    labels: torch.Tensor,
    member_mask: torch.Tensor,
) -> torch.Tensor:
    left = labels.unsqueeze(2)
    right = labels.unsqueeze(1)
    pair_mask = member_mask.unsqueeze(2) & member_mask.unsqueeze(1)
    lower_count = ((right < left) & pair_mask).sum(dim=2).to(labels.dtype)
    equal_count = ((right == left) & pair_mask).sum(dim=2).to(labels.dtype)
    average_rank = lower_count + (equal_count - 1.0).clamp(min=0.0) * 0.5
    denominator = (member_mask.sum(dim=1, keepdim=True) - 1).clamp(min=1).to(labels.dtype)
    return (average_rank / denominator).masked_fill(~member_mask, 0.0)


def _validate_listwise_inputs(
    scores: torch.Tensor,
    labels: torch.Tensor,
    member_mask: torch.Tensor,
    temperature: float,
) -> None:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if scores.ndim != 2 or scores.shape != labels.shape:
        raise ValueError("scores and labels must have matching shape [G, M]")
    if member_mask.dtype != torch.bool or member_mask.shape != scores.shape:
        raise ValueError("member_mask must be BoolTensor[G, M] matching scores")
    if not scores.is_floating_point() or not labels.is_floating_point():
        raise ValueError("scores and labels must be floating-point tensors")
    if not torch.isfinite(scores[member_mask]).all() or not torch.isfinite(labels[member_mask]).all():
        raise ValueError("valid scores and labels must be finite")
