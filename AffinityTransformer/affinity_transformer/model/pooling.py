"""Mask-aware pooling functions."""

from __future__ import annotations

import torch


def masked_mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool valid token representations without producing NaN rows."""
    mask_f = mask.to(dtype=hidden.dtype).unsqueeze(-1)
    summed = (hidden * mask_f).sum(dim=1)
    counts = mask_f.sum(dim=1).clamp(min=1.0)
    return summed / counts
