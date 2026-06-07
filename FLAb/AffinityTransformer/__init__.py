"""
AffinityTransformer — v3 antigen-aware affinity modeling package.

这个包与 AffinityMLP(v1/v2) 隔离。v3 可以复制 v1/v2 的思想，但不应 import
或修改 AffinityMLP 的实现。
"""

from .config import V3Config, cfg

__all__ = ["V3Config", "cfg"]

