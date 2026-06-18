"""Legacy online-encoder affinity ranker.

This module preserves the current model behavior while the v0.65
embedding-backed architecture is implemented in later changes.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..dataloader import RankBatch
from .attention import build_cross_attention
from .heads import build_scalar_head
from .pooling import masked_mean_pool


class AffinityRanker(nn.Module):
    """Convert antibody and optional antigen token batches to scalar scores."""

    def __init__(
        self,
        antibody_encoder: nn.Module,
        antigen_encoder: nn.Module | None,
        d_model: int,
        use_cross_attention: bool,
    ) -> None:
        super().__init__()
        self.antibody_encoder = antibody_encoder
        self.antigen_encoder = antigen_encoder
        self.use_cross_attention = use_cross_attention and antigen_encoder is not None

        if self.use_cross_attention:
            self.cross_attention = build_cross_attention(d_model)

        head_input_dim = d_model * 2 if antigen_encoder is not None else d_model
        self.head = build_scalar_head(head_input_dim, d_model)

    def forward(self, batch: RankBatch) -> torch.Tensor:
        """Score each record in ``batch`` without constraining score range."""
        antibody_hidden = torch.nan_to_num(
            self.antibody_encoder(batch.antibody_tokens, batch.antibody_mask), nan=0.0
        )

        if self.antigen_encoder is None:
            antibody_repr = masked_mean_pool(antibody_hidden, batch.antibody_mask)
            return self.head(antibody_repr).squeeze(-1)

        if batch.antigen_tokens is None:
            antibody_repr = masked_mean_pool(antibody_hidden, batch.antibody_mask)
            antigen_repr = torch.zeros_like(antibody_repr)
        else:
            antigen_hidden = torch.nan_to_num(
                self.antigen_encoder(batch.antigen_tokens, batch.antigen_mask), nan=0.0
            )

            if self.use_cross_attention:
                key_padding_mask = ~batch.antigen_mask
                has_antigen = batch.antigen_mask.any(dim=1)
                if (~has_antigen).any():
                    key_padding_mask = key_padding_mask.clone()
                    key_padding_mask[~has_antigen, 0] = False

                attended, _ = self.cross_attention(
                    query=antibody_hidden,
                    key=antigen_hidden,
                    value=antigen_hidden,
                    key_padding_mask=key_padding_mask,
                )
                attended = torch.nan_to_num(attended, nan=0.0)
                use_attended = has_antigen.view(-1, 1, 1)
                antibody_for_pool = torch.where(use_attended, attended, antibody_hidden)
                antibody_repr = masked_mean_pool(antibody_for_pool, batch.antibody_mask)
            else:
                antibody_repr = masked_mean_pool(antibody_hidden, batch.antibody_mask)

            antigen_repr = masked_mean_pool(antigen_hidden, batch.antigen_mask)

        combined = torch.cat([antibody_repr, antigen_repr], dim=-1)
        return self.head(combined).squeeze(-1)
