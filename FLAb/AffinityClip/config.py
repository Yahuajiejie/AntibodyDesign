"""
config.py - AffinityClip(v4) independent configuration.

This file does not import v1/v2/v3 config.  The dimensions mirror the
feature layout produced by AffinityTransformer(v3), but v4 keeps its own
defaults so that ablation experiments can be reported independently.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AffinityClipConfig:
    """
    Configuration for the v4 two-tower model.

    Parameters:
      heavy_dim:            heavy-chain embedding dimension.
      light_dim:            light-chain embedding dimension.
      antigen_single_dim:   single antigen sequence embedding dimension.
      antigen_msa_dim:      MSA-aware antigen embedding dimension.
      antigen_flag_dim:     antigen type one-hot + availability flags dimension.
      token_dim:            hidden size inside each Transformer tower.
      projection_dim:       final shared retrieval embedding dimension.
      num_layers:           TransformerEncoder layer count per tower.
      num_heads:            attention head count per tower.
      feedforward_dim:      Transformer feed-forward hidden dimension.
      dropout:              dropout used in projections and Transformer layers.
      logit_temperature:    initial temperature for cosine-similarity logits.
      logit_scale_max:      upper clamp for exp(logit_scale), following CLIP-style
                            practice to avoid numerically unstable logits.
      clip_weight:          weight of group-aware soft CLIP loss.
      ranknet_weight:       weight of within-group RankNet loss.
      label_temperature:    softness of affinity-label targets in CLIP loss.
      min_label_diff:       ignore pairs whose label gap is too small.
    """

    heavy_dim: int = 1280
    light_dim: int = 1280
    antigen_single_dim: int = 1280
    antigen_msa_dim: int = 768
    antigen_flag_dim: int = 20

    token_dim: int = 256
    projection_dim: int = 256
    num_layers: int = 2
    num_heads: int = 8
    feedforward_dim: int = 512
    dropout: float = 0.2

    logit_temperature: float = 0.07
    logit_scale_max: float = 100.0

    clip_weight: float = 1.0
    ranknet_weight: float = 1.0
    label_temperature: float = 0.25
    min_label_diff: float = 0.0

    seed: int = 42
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 100

    @property
    def antibody_feature_dims(self) -> tuple[int, int]:
        """Return the default antibody token dimensions: heavy, light."""
        return (self.heavy_dim, self.light_dim)

    @property
    def antigen_feature_dims(self) -> tuple[int, int, int]:
        """Return the default antigen token dimensions: single, MSA, flags."""
        return (
            self.antigen_single_dim,
            self.antigen_msa_dim,
            self.antigen_flag_dim,
        )

    @property
    def v3_concat_dim(self) -> int:
        """Return expected v3 concat feature dimension for chain_concat input."""
        return (
            self.heavy_dim
            + self.light_dim
            + self.antigen_single_dim
            + self.antigen_msa_dim
            + self.antigen_flag_dim
        )


cfg = AffinityClipConfig()
