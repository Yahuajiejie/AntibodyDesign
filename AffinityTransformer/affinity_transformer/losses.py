"""Pairwise ranking loss(es).

(spec docs/programming_spec.md §5.5)

This module only implements loss functions. It does not construct pairs
(that is `dataset.build_pairs`, spec §5.2) and does not run the training
loop (that is `trainer.py`, spec §5.7).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def ranknet_loss(
    score_i: torch.Tensor,
    score_j: torch.Tensor,
    y_ij: torch.Tensor,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Compute the RankNet pairwise ranking loss.

    Args:
        score_i: Model scores for the "i" side of each pair. Any shape, but
            must broadcast with `score_j` and `y_ij`. Typically `[B]`.
        score_j: Model scores for the "j" side of each pair. Same shape
            constraints as `score_i`.
        y_ij: Target for each pair: `1.0` if `score_i` should rank above
            `score_j`, `0.0` otherwise (spec §5.2 pair-construction rule 5).
            Must be a float tensor with values in `{0.0, 1.0}`; `build_pairs`
            guarantees this by construction and it is not re-validated here.
        sigma: Scale applied to the score difference before the implicit
            sigmoid (spec §5.5 rule 2). Larger `sigma` makes the loss more
            sensitive to small score differences.

    Returns:
        Scalar tensor: the mean binary cross-entropy with logits
        `sigma * (score_i - score_j)` and targets `y_ij` (spec §5.5 rules
        1-3, via `torch.nn.functional.binary_cross_entropy_with_logits`).

    Raises:
        RuntimeError: If `score_i`, `score_j`, and `y_ij` have shapes that
            cannot be broadcast together, or if `y_ij` is not a float dtype
            (propagated from
            `torch.nn.functional.binary_cross_entropy_with_logits`).

    Example:
        >>> ranknet_loss(torch.tensor(2.0), torch.tensor(1.0), torch.tensor(1.0)) < \\
        ...     ranknet_loss(torch.tensor(1.0), torch.tensor(2.0), torch.tensor(1.0))
        tensor(True)
    """
    logits = sigma * (score_i - score_j)
    return F.binary_cross_entropy_with_logits(logits, y_ij)
