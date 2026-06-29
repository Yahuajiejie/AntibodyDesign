"""Embedding-native affinity ranker specified by programming spec v0.65."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from ..embeddings.collate import EmbeddingBatch
from .heads import ScalarScoringHead
from .interaction import DeepCrossAttention
from .pooling import build_pooling
from .projections import TokenProjection

FusionKind = Literal["antibody_only", "concat", "deep_cross_attention"]


class EmbeddingAffinityRanker(nn.Module):
    """Score cached antibody/antigen token embeddings without base encoders."""

    def __init__(
        self,
        *,
        antibody_input_dim: int,
        antigen_input_dim: int | None,
        d_model: int,
        fusion_kind: FusionKind,
        num_layers: int = 0,
        num_heads: int = 8,
        ffn_multiplier: float = 4.0,
        dropout: float = 0.1,
        pooling: str = "masked_mean",
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        _validate_architecture(
            antigen_input_dim=antigen_input_dim,
            d_model=d_model,
            fusion_kind=fusion_kind,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        self.fusion_kind = fusion_kind
        self.d_model = d_model
        self.antibody_projection = TokenProjection(antibody_input_dim, d_model)
        self.antibody_pooling = build_pooling(pooling, d_model)

        if fusion_kind == "antibody_only":
            self.antigen_projection = None
            self.antigen_pooling = None
            self.interaction = None
            head_input_dim = d_model
        else:
            assert antigen_input_dim is not None
            self.antigen_projection = TokenProjection(antigen_input_dim, d_model)
            self.antigen_pooling = build_pooling(pooling, d_model)
            self.interaction = (
                DeepCrossAttention(
                    d_model=d_model,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    ffn_multiplier=ffn_multiplier,
                    dropout=dropout,
                    bidirectional=bidirectional,
                )
                if fusion_kind == "deep_cross_attention"
                else None
            )
            head_input_dim = d_model * 2

        self.scoring_head = ScalarScoringHead(
            input_dim=head_input_dim,
            hidden_dim=d_model,
            dropout=dropout,
        )

    def forward(self, batch: EmbeddingBatch) -> torch.Tensor:
        """Return one unbounded affinity-ranking score per record."""
        representation = self.forward_features(batch)
        return self.scoring_head(representation)

    def forward_features(self, batch: EmbeddingBatch) -> torch.Tensor:
        """Return the pooled representation consumed by the scalar head."""
        antibody_tokens = self.antibody_projection(
            batch.antibody_embeddings,
            batch.antibody_mask,
        )
        if self.fusion_kind == "antibody_only":
            return self.antibody_pooling(antibody_tokens, batch.antibody_mask)

        antibody_repr: torch.Tensor
        if batch.antigen_embeddings is None:
            antibody_repr = self.antibody_pooling(antibody_tokens, batch.antibody_mask)
            antigen_repr = torch.zeros_like(antibody_repr)
        else:
            if batch.antigen_mask is None:
                raise ValueError("antigen_mask is required when antigen_embeddings are present")
            assert self.antigen_projection is not None
            assert self.antigen_pooling is not None
            antigen_tokens = self.antigen_projection(
                batch.antigen_embeddings,
                batch.antigen_mask,
            )
            if self.interaction is not None:
                antibody_tokens, antigen_tokens = self.interaction(
                    antibody_tokens,
                    antigen_tokens,
                    batch.antibody_mask,
                    batch.antigen_mask,
                )
            antibody_repr = self.antibody_pooling(antibody_tokens, batch.antibody_mask)
            antigen_repr = self.antigen_pooling(antigen_tokens, batch.antigen_mask)

        return torch.cat([antibody_repr, antigen_repr], dim=-1)


def _validate_architecture(
    *,
    antigen_input_dim: int | None,
    d_model: int,
    fusion_kind: str,
    num_layers: int,
    num_heads: int,
) -> None:
    supported = {"antibody_only", "concat", "deep_cross_attention"}
    if fusion_kind not in supported:
        raise ValueError(f"unsupported fusion_kind {fusion_kind!r}; expected {sorted(supported)}")
    if d_model < 1:
        raise ValueError("d_model must be positive")
    if fusion_kind == "antibody_only":
        if antigen_input_dim is not None:
            raise ValueError("antibody_only requires antigen_input_dim=None")
        if num_layers != 0:
            raise ValueError("antibody_only requires num_layers=0")
    elif antigen_input_dim is None:
        raise ValueError(f"{fusion_kind} requires antigen_input_dim")
    elif fusion_kind == "concat" and num_layers != 0:
        raise ValueError("concat requires num_layers=0")
    elif fusion_kind == "deep_cross_attention" and num_layers < 1:
        raise ValueError("deep_cross_attention requires num_layers >= 1")
    elif fusion_kind == "deep_cross_attention" and (
        num_heads < 1 or d_model % num_heads != 0
    ):
        raise ValueError("deep_cross_attention requires d_model divisible by num_heads")
