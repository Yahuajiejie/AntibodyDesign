"""
AffinityClip(v4) package.

v4 is intentionally separate from AffinityMLPSimplified(v1),
AffinityMLP(v2), and AffinityTransformer(v3).  It implements a
DrugCLIP-inspired two-tower ranking model:

  antigen context -> query tower
  antibody        -> key tower
  similarity      -> RankNet / soft CLIP-style loss
"""

from .config import AffinityClipConfig, cfg
from .features import AffinityClipFeatureLayout, split_v3_concat_features
from .losses import affinity_clip_loss, group_soft_clip_loss, ranknet_loss_from_scores
from .model import AffinityCLIP, TransformerTower

__all__ = [
    "AffinityCLIP",
    "AffinityClipConfig",
    "AffinityClipFeatureLayout",
    "TransformerTower",
    "affinity_clip_loss",
    "cfg",
    "group_soft_clip_loss",
    "ranknet_loss_from_scores",
    "split_v3_concat_features",
]
