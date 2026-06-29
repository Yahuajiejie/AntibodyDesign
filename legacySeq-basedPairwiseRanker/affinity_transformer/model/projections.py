"""Trainable projections from base-model embeddings to interaction space."""

from __future__ import annotations

import torch
import torch.nn as nn


class TokenProjection(nn.Module):
    """Project one encoder's token embeddings into a shared hidden width."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        if input_dim < 1 or output_dim < 1:
            raise ValueError("projection dimensions must be positive")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_norm = nn.LayerNorm(input_dim)
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return projected tokens with every invalid position reset to zero."""
        _validate_token_tensor(embeddings, mask, self.input_dim)
        # Cached embeddings are commonly float16 while trainable projection
        # parameters default to float32. Keep this boundary independent of
        # whether a caller enabled autocast.
        embeddings = embeddings.to(dtype=self.input_norm.weight.dtype)
        projected = self.projection(self.input_norm(embeddings))
        return projected.masked_fill(~mask.unsqueeze(-1), 0.0)


def _validate_token_tensor(
    embeddings: torch.Tensor,
    mask: torch.Tensor,
    expected_dim: int,
) -> None:
    if embeddings.ndim != 3:
        raise ValueError(
            f"embeddings must have shape [B, L, D], got {tuple(embeddings.shape)}"
        )
    if embeddings.shape[-1] != expected_dim:
        raise ValueError(
            f"embedding dim mismatch: expected {expected_dim}, got {embeddings.shape[-1]}"
        )
    if mask.dtype != torch.bool or mask.shape != embeddings.shape[:2]:
        raise ValueError(
            "mask must be BoolTensor[B, L] matching embeddings; "
            f"got dtype={mask.dtype}, shape={tuple(mask.shape)}"
        )
