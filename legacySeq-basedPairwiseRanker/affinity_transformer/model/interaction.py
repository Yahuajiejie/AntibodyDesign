"""Stacked antibody-antigen interaction backbones."""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import InteractionBlock


class DeepCrossAttention(nn.Module):
    """Apply a configurable stack of independent interaction blocks."""

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ffn_multiplier: float = 4.0,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("deep cross-attention requires num_layers >= 1")
        self.layers = nn.ModuleList([
            InteractionBlock(
                d_model=d_model,
                num_heads=num_heads,
                ffn_multiplier=ffn_multiplier,
                dropout=dropout,
                bidirectional=bidirectional,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        antibody_tokens: torch.Tensor,
        antigen_tokens: torch.Tensor,
        antibody_mask: torch.Tensor,
        antigen_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            antibody_tokens, antigen_tokens = layer(
                antibody_tokens,
                antigen_tokens,
                antibody_mask,
                antigen_mask,
            )
        return antibody_tokens, antigen_tokens
