"""Dataset package for standard affinity records and ranking views.

This package replaces the former single-file ``affinity_transformer.dataset``
module while keeping the same import path for callers:

- ``schema``: standard table columns and sampler defaults.
- ``examples``: dataclasses used by downstream datasets.
- ``records``: table loading and trainable-record filtering.
- ``pairs``: pairwise ranking pair construction.
- ``pair_sampling``: large-group, two-label, and block samplers.
- ``groups``: listwise group construction.
- ``datasets``: torch ``Dataset`` wrappers.
"""

from .datasets import AffinityRecordDataset, ListwiseAffinityDataset, PairwiseAffinityDataset
from .examples import AffinityExample, AffinityGroupExample, AffinityPairExample
from .groups import build_groups
from .pairs import build_pairs
from .records import filter_trainable_records, load_records
from .schema import GROUP_COLUMNS, PAIR_COLUMNS, REQUIRED_COLUMNS

__all__ = [
    "AffinityExample",
    "AffinityGroupExample",
    "AffinityPairExample",
    "AffinityRecordDataset",
    "ListwiseAffinityDataset",
    "PairwiseAffinityDataset",
    "GROUP_COLUMNS",
    "PAIR_COLUMNS",
    "REQUIRED_COLUMNS",
    "build_groups",
    "build_pairs",
    "filter_trainable_records",
    "load_records",
]
