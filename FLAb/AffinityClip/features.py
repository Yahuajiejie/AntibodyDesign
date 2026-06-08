"""
features.py - feature slicing helpers for AffinityClip(v4).

v4 can reuse v3 cached features, but the model wants separate modality
tokens instead of one long vector.  This module performs only deterministic
slicing; it does not generate embeddings and does not read data files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import cfg


@dataclass(frozen=True)
class AffinityClipFeatureLayout:
    """
    Layout of a v3-style concatenated feature matrix.

    Parameters:
      heavy_dim:          dimension of heavy-chain embedding.
      light_dim:          dimension of light-chain embedding.
      antigen_single_dim: dimension of single antigen embedding.
      antigen_msa_dim:    dimension of MSA-aware antigen embedding.
      antigen_flag_dim:   dimension of antigen type/availability flags.

    Returns:
      This dataclass itself returns no value; it stores the slicing contract.
    """

    heavy_dim: int = cfg.heavy_dim
    light_dim: int = cfg.light_dim
    antigen_single_dim: int = cfg.antigen_single_dim
    antigen_msa_dim: int = cfg.antigen_msa_dim
    antigen_flag_dim: int = cfg.antigen_flag_dim

    @property
    def total_dim(self) -> int:
        """Return the expected full feature dimension."""
        return (
            self.heavy_dim
            + self.light_dim
            + self.antigen_single_dim
            + self.antigen_msa_dim
            + self.antigen_flag_dim
        )


def _shape_dim(features: Any) -> int:
    """Return features.shape[1] for numpy arrays or torch tensors."""
    if not hasattr(features, "shape") or len(features.shape) != 2:
        raise ValueError("features 必须是二维矩阵：[batch, dim]")
    return int(features.shape[1])


def _slice(features: Any, start: int, dim: int) -> Any:
    """
    Slice a matrix without changing its backend.

    Works for numpy.ndarray, torch.Tensor, and pandas/numpy-like objects that
    support [:, start:end] indexing.
    """
    return features[:, start:start + dim]


def split_v3_concat_features(
    features: Any,
    layout: AffinityClipFeatureLayout = AffinityClipFeatureLayout(),
) -> dict[str, list[Any]]:
    """
    Split a v3-style concat matrix into v4 tower inputs.

    Parameters:
      features: 2D matrix with column order:
                heavy, light, antigen_single, antigen_msa, antigen_flags.
      layout:   dimension contract used to slice columns.

    Returns:
      dict with:
        antibody_features: [heavy, light]
        antigen_features:  [antigen_single, antigen_msa, antigen_flags]

    Implementation:
      The function checks the total dimension first, then returns views/slices.
      It does not copy data unless the backend itself decides to copy.
    """
    dim = _shape_dim(features)
    if dim != layout.total_dim:
        raise ValueError(
            f"features dim={dim}, expected {layout.total_dim}. "
            "请确认 v3 特征顺序是否为 heavy/light/single/MSA/flags。"
        )

    start = 0
    heavy = _slice(features, start, layout.heavy_dim)
    start += layout.heavy_dim
    light = _slice(features, start, layout.light_dim)
    start += layout.light_dim
    single = _slice(features, start, layout.antigen_single_dim)
    start += layout.antigen_single_dim
    msa = _slice(features, start, layout.antigen_msa_dim)
    start += layout.antigen_msa_dim
    flags = _slice(features, start, layout.antigen_flag_dim)

    return {
        "antibody_features": [heavy, light],
        "antigen_features": [single, msa, flags],
    }
