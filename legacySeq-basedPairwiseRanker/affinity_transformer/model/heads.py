"""Scalar scoring heads."""

from __future__ import annotations

import torch
import torch.nn as nn


def build_scalar_head(input_dim: int, hidden_dim: int) -> nn.Sequential:
    """Build the current unbounded two-layer scalar scoring head."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )


class ScalarScoringHead(nn.Module):
    """Unbounded v0.65 scalar scorer shared by every ranking objective."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1:
            raise ValueError("head dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        self.layers = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        """Return one unconstrained score per representation row."""
        return self.layers(representation).squeeze(-1)
