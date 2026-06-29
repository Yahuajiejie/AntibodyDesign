"""Mask-safe antibody-antigen interaction Transformer blocks."""

from __future__ import annotations

import torch
import torch.nn as nn


class InteractionBlock(nn.Module):
    """Pre-norm cross-attention and FFN updates for two token streams."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_multiplier: float = 4.0,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        if d_model < 1 or num_heads < 1 or d_model % num_heads != 0:
            raise ValueError("d_model must be positive and divisible by num_heads")
        if ffn_multiplier <= 0:
            raise ValueError("ffn_multiplier must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")

        self.bidirectional = bidirectional
        self.antibody_cross_norm = nn.LayerNorm(d_model)
        self.antigen_cross_norm = nn.LayerNorm(d_model)
        self.antibody_to_antigen = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        if bidirectional:
            self.antigen_to_antibody = nn.MultiheadAttention(
                d_model,
                num_heads,
                dropout=dropout,
                batch_first=True,
            )

        hidden_dim = max(1, round(d_model * ffn_multiplier))
        self.antibody_ffn_norm = nn.LayerNorm(d_model)
        self.antibody_ffn = _feed_forward(d_model, hidden_dim, dropout)
        if bidirectional:
            self.antigen_ffn_norm = nn.LayerNorm(d_model)
            self.antigen_ffn = _feed_forward(d_model, hidden_dim, dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(
        self,
        antibody_tokens: torch.Tensor,
        antigen_tokens: torch.Tensor,
        antibody_mask: torch.Tensor,
        antigen_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update token streams while preserving zeros at invalid positions."""
        _validate_streams(antibody_tokens, antigen_tokens, antibody_mask, antigen_mask)
        antibody_tokens = _zero_invalid(antibody_tokens, antibody_mask)
        antigen_tokens = _zero_invalid(antigen_tokens, antigen_mask)
        original_antibody = antibody_tokens
        has_antigen = antigen_mask.any(dim=1).view(-1, 1, 1)

        normalized_antibody = _zero_invalid(
            self.antibody_cross_norm(antibody_tokens), antibody_mask
        )
        normalized_antigen = _zero_invalid(
            self.antigen_cross_norm(antigen_tokens), antigen_mask
        )

        antibody_delta = _safe_cross_attention(
            self.antibody_to_antigen,
            query=normalized_antibody,
            key_value=normalized_antigen,
            query_mask=antibody_mask,
            key_value_mask=antigen_mask,
        )
        updated_antibody = _zero_invalid(
            antibody_tokens + self.residual_dropout(antibody_delta),
            antibody_mask,
        )

        if self.bidirectional:
            antigen_delta = _safe_cross_attention(
                self.antigen_to_antibody,
                query=normalized_antigen,
                key_value=normalized_antibody,
                query_mask=antigen_mask,
                key_value_mask=antibody_mask,
            )
            updated_antigen = _zero_invalid(
                antigen_tokens + self.residual_dropout(antigen_delta),
                antigen_mask,
            )
        else:
            updated_antigen = antigen_tokens

        antibody_ffn_delta = self.antibody_ffn(
            _zero_invalid(self.antibody_ffn_norm(updated_antibody), antibody_mask)
        )
        updated_antibody = _zero_invalid(
            updated_antibody + self.residual_dropout(antibody_ffn_delta),
            antibody_mask,
        )

        if self.bidirectional:
            antigen_ffn_delta = self.antigen_ffn(
                _zero_invalid(self.antigen_ffn_norm(updated_antigen), antigen_mask)
            )
            updated_antigen = _zero_invalid(
                updated_antigen + self.residual_dropout(antigen_ffn_delta),
                antigen_mask,
            )
        # Missing-antigen rows bypass the entire interaction block. This
        # keeps their result identical whether they appear alone (the ranker
        # skips the stack) or beside records that do have antigen context.
        updated_antibody = torch.where(has_antigen, updated_antibody, original_antibody)
        return updated_antibody, updated_antigen


def _feed_forward(d_model: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_model, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, d_model),
    )


def _safe_cross_attention(
    attention: nn.MultiheadAttention,
    *,
    query: torch.Tensor,
    key_value: torch.Tensor,
    query_mask: torch.Tensor,
    key_value_mask: torch.Tensor,
) -> torch.Tensor:
    key_padding_mask = ~key_value_mask
    has_context = key_value_mask.any(dim=1)
    safe_key_value = key_value
    if (~has_context).any():
        key_padding_mask = key_padding_mask.clone()
        key_padding_mask[~has_context, 0] = False
        safe_key_value = key_value.clone()
        safe_key_value[~has_context, 0] = 0.0
    attended, _ = attention(
        query=query,
        key=safe_key_value,
        value=safe_key_value,
        key_padding_mask=key_padding_mask,
        need_weights=False,
    )
    valid_output = query_mask & has_context.unsqueeze(1)
    return torch.nan_to_num(attended, nan=0.0).masked_fill(
        ~valid_output.unsqueeze(-1), 0.0
    )


def _zero_invalid(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return tokens.masked_fill(~mask.unsqueeze(-1), 0.0)


def _validate_streams(
    antibody_tokens: torch.Tensor,
    antigen_tokens: torch.Tensor,
    antibody_mask: torch.Tensor,
    antigen_mask: torch.Tensor,
) -> None:
    if antibody_tokens.ndim != 3 or antigen_tokens.ndim != 3:
        raise ValueError("interaction tokens must have shape [B, L, D]")
    if antibody_tokens.shape[0] != antigen_tokens.shape[0]:
        raise ValueError("antibody and antigen batch sizes must match")
    if antibody_tokens.shape[2] != antigen_tokens.shape[2]:
        raise ValueError("antibody and antigen interaction dimensions must match")
    if antibody_mask.dtype != torch.bool or antibody_mask.shape != antibody_tokens.shape[:2]:
        raise ValueError("antibody_mask must be BoolTensor[B, L_ab]")
    if antigen_mask.dtype != torch.bool or antigen_mask.shape != antigen_tokens.shape[:2]:
        raise ValueError("antigen_mask must be BoolTensor[B, L_ag]")
