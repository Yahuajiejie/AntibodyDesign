"""Ranking losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


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
