"""Attention construction helpers for the current ranker."""

from __future__ import annotations

import torch.nn as nn

_CROSS_ATTENTION_HEAD_OPTIONS = (8, 4, 2, 1)


def select_num_heads(d_model: int) -> int:
    """Return the largest supported head count that divides ``d_model``."""
    for num_heads in _CROSS_ATTENTION_HEAD_OPTIONS:
        if d_model % num_heads == 0:
            return num_heads
    return 1


def build_cross_attention(d_model: int) -> nn.MultiheadAttention:
    """Construct the legacy single cross-attention layer."""
    return nn.MultiheadAttention(
        embed_dim=d_model,
        num_heads=select_num_heads(d_model),
        batch_first=True,
    )
