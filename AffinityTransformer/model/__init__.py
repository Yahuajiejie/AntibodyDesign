"""Readable split mirror of the AffinityTransformer core components.

The production training code still lives under ``affinity_transformer/``.
Files under ``model/`` are split copies intended for code review, onboarding,
and future refactoring. Keep behavior aligned with the original package when
copying changes across.
"""

from .dataset import (
    AffinityExample,
    AffinityGroupExample,
    AffinityPairExample,
    AffinityRecordDataset,
    ListwiseAffinityDataset,
    PairwiseAffinityDataset,
    build_groups,
    build_pairs,
    filter_trainable_records,
    load_records,
)

__all__ = [
    "AffinityExample",
    "AffinityGroupExample",
    "AffinityPairExample",
    "AffinityRecordDataset",
    "ListwiseAffinityDataset",
    "PairwiseAffinityDataset",
    "build_groups",
    "build_pairs",
    "filter_trainable_records",
    "load_records",
]
