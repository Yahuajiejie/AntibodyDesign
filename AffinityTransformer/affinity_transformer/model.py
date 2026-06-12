"""Ranking model (spec docs/programming_spec.md §5.4).

`AffinityRanker` turns a `RankBatch` (spec §5.3) into one score per example.
It does not construct any encoder itself: `antibody_encoder` /
`antigen_encoder` are pretrained-or-otherwise `nn.Module`s built and supplied
by the caller (eventually `trainer.py`, spec §5.7, not yet implemented), each
satisfying::

    forward(input_ids: LongTensor[B, L], attention_mask: BoolTensor[B, L])
        -> FloatTensor[B, L, d_model]

with no `NaN` in the output, even for rows where `attention_mask` is entirely
`False`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .dataloader import RankBatch

_HEAD_HIDDEN_DIVISOR = 1  # head hidden width == d_model (no extra widening)
_CROSS_ATTENTION_HEAD_OPTIONS = (8, 4, 2, 1)


def _select_num_heads(d_model: int) -> int:
    """Pick a number of attention heads that evenly divides `d_model`.

    Args:
        d_model: Shared hidden dimension.

    Returns:
        The largest value in `(8, 4, 2, 1)` that divides `d_model`. `1`
        always divides any positive integer, so this always returns a valid
        value.
    """
    for num_heads in _CROSS_ATTENTION_HEAD_OPTIONS:
        if d_model % num_heads == 0:
            return num_heads
    return 1


def _masked_mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token representations over valid positions.

    Args:
        hidden: `FloatTensor[B, L, d_model]` per-token representations.
        mask: `BoolTensor[B, L]`. `True` = valid token, `False` =
            padding/missing (spec §5.3 mask convention).

    Returns:
        `FloatTensor[B, d_model]`. Rows where `mask` is entirely `False`
        return an all-zero vector (no division by zero, no `NaN`).
    """
    mask_f = mask.to(dtype=hidden.dtype).unsqueeze(-1)  # [B, L, 1]
    summed = (hidden * mask_f).sum(dim=1)  # [B, d_model]
    counts = mask_f.sum(dim=1).clamp(min=1.0)  # [B, 1]
    return summed / counts


class AffinityRanker(nn.Module):
    """Antibody (+ optional antigen) -> scalar affinity-ranking score.

    Attributes:
        antibody_encoder: `nn.Module` mapping
            `(antibody_tokens, antibody_mask) -> FloatTensor[B, L, d_model]`.
        antigen_encoder: `nn.Module` with the same interface for antigen
            tokens, or `None` for an antibody-only model.
        use_cross_attention: Whether to run antibody-query / antigen-key-value
            cross-attention before pooling the antibody representation.
            Ignored (treated as `False`) when `antigen_encoder is None`.
    """

    def __init__(
        self,
        antibody_encoder: nn.Module,
        antigen_encoder: nn.Module | None,
        d_model: int,
        use_cross_attention: bool,
    ) -> None:
        """Build the ranker on top of already-constructed encoders.

        Args:
            antibody_encoder: Antibody sequence encoder (spec §5.4 encoder
                interface). Constructed by the caller.
            antigen_encoder: Antigen sequence encoder with the same
                interface, or `None` to run in antibody-only mode.
            d_model: Hidden dimension produced by both encoders.
            use_cross_attention: Whether to apply antibody-antigen
                cross-attention (spec §0: a feature-interaction module, not a
                structural simulation). Has no effect if `antigen_encoder is
                None`.
        """
        super().__init__()
        self.antibody_encoder = antibody_encoder
        self.antigen_encoder = antigen_encoder
        self.use_cross_attention = use_cross_attention and antigen_encoder is not None

        if self.use_cross_attention:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=_select_num_heads(d_model),
                batch_first=True,
            )

        head_input_dim = d_model * 2 if antigen_encoder is not None else d_model
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, _HEAD_HIDDEN_DIVISOR),
        )

    def forward(self, batch: RankBatch) -> torch.Tensor:
        """Score every example in `batch`.

        Args:
            batch: A `RankBatch` (spec §5.3). Only tensor fields
                (`antibody_tokens`, `antibody_mask`, `antigen_tokens`,
                `antigen_mask`) are used -- `record_ids`/`group_ids` are
                ignored (spec §5.4 rule 4).

        Returns:
            `score: FloatTensor[B]`, unbounded (no sigmoid/softmax/clamp,
            spec §5.4 rule 1). Never contains `NaN`, even if
            `batch.antigen_tokens is None` or some row of `batch.antigen_mask`
            is entirely `False` (spec §5.4 rules 2-3).
        """
        antibody_hidden = torch.nan_to_num(
            self.antibody_encoder(batch.antibody_tokens, batch.antibody_mask), nan=0.0
        )

        if self.antigen_encoder is None:
            antibody_repr = _masked_mean_pool(antibody_hidden, batch.antibody_mask)
            return self.head(antibody_repr).squeeze(-1)

        if batch.antigen_tokens is None:
            antibody_repr = _masked_mean_pool(antibody_hidden, batch.antibody_mask)
            antigen_repr = torch.zeros_like(antibody_repr)
        else:
            antigen_hidden = torch.nan_to_num(
                self.antigen_encoder(batch.antigen_tokens, batch.antigen_mask), nan=0.0
            )

            if self.use_cross_attention:
                key_padding_mask = ~batch.antigen_mask  # True = ignore (spec mask is True=valid)

                # A row with an entirely-True key_padding_mask makes
                # nn.MultiheadAttention's softmax divide 0/0 (NaN), which then
                # poisons gradients for *all* rows via the shared projection
                # weights -- even though that row's output is discarded below.
                # Unmask one dummy position for such rows so attention itself
                # stays finite; the row's contribution is replaced by
                # antibody_hidden (antibody-only baseline) regardless.
                has_antigen = batch.antigen_mask.any(dim=1)  # [B]
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

                # Rows with no valid antigen token skip cross-attention
                # entirely (spec §5.4 rule 2: antibody-only baseline).
                use_attended = has_antigen.view(-1, 1, 1)
                antibody_for_pool = torch.where(use_attended, attended, antibody_hidden)
                antibody_repr = _masked_mean_pool(antibody_for_pool, batch.antibody_mask)
            else:
                antibody_repr = _masked_mean_pool(antibody_hidden, batch.antibody_mask)

            antigen_repr = _masked_mean_pool(antigen_hidden, batch.antigen_mask)

        combined = torch.cat([antibody_repr, antigen_repr], dim=-1)
        return self.head(combined).squeeze(-1)
