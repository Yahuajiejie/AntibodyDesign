"""Split dataset module.

This package mirrors ``affinity_transformer.dataset`` but divides the original
large file by responsibility:

- ``schema``: standard table columns and sampler defaults.
- ``examples``: dataclasses used by downstream datasets.
- ``records``: table loading and trainable-record filtering.
- ``pairs``: pairwise ranking pair construction.
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
