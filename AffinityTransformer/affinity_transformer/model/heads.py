"""Scalar scoring heads."""

from __future__ import annotations

import torch.nn as nn


def build_scalar_head(input_dim: int, hidden_dim: int) -> nn.Sequential:
    """Build the current unbounded two-layer scalar scoring head."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )
