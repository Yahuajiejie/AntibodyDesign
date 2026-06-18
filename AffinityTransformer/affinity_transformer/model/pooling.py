"""Mask-aware pooling functions."""

from __future__ import annotations

import torch
import torch.nn as nn


def masked_mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool valid token representations without producing NaN rows."""
    mask_f = mask.to(dtype=hidden.dtype).unsqueeze(-1)
    summed = (hidden * mask_f).sum(dim=1)
    counts = mask_f.sum(dim=1).clamp(min=1.0)
    return summed / counts


class MaskedMeanPooling(nn.Module):
    """Module wrapper around :func:`masked_mean_pool`."""

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return masked_mean_pool(hidden, mask)


class AttentionPooling(nn.Module):
    """Learned-query pooling with explicit all-missing-row handling."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        if d_model < 1:
            raise ValueError("d_model must be positive")
        self.norm = nn.LayerNorm(d_model)
        self.query = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.query, std=d_model ** -0.5)
        self.scale = d_model ** -0.5

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or mask.shape != hidden.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("attention pooling expects hidden[B,L,D] and BoolTensor mask[B,L]")
        normalized = self.norm(hidden)
        logits = torch.einsum("bld,d->bl", normalized, self.query) * self.scale
        has_tokens = mask.any(dim=1, keepdim=True)
        safe_mask = mask.clone()
        if (~has_tokens).any():
            safe_mask[~has_tokens.squeeze(1), 0] = True
        logits = logits.masked_fill(~safe_mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1)
        weights = weights * mask.to(weights.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1.0)
        return torch.einsum("bl,bld->bd", weights, hidden)


def build_pooling(name: str, d_model: int) -> nn.Module:
    """Construct a supported mask-aware pooling module."""
    if name == "masked_mean":
        return MaskedMeanPooling()
    if name == "attention_pool":
        return AttentionPooling(d_model)
    raise ValueError(f"unsupported pooling {name!r}; expected 'masked_mean' or 'attention_pool'")
