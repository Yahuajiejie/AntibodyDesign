"""Private helpers behind ``affinity_transformer.dataset.pairs``."""

from .common import (
    _candidate_pair_count,
    _canonical_pair,
    _pair_row,
    _pair_sample_count,
    _should_enumerate_pairs,
    _weighted_choice,
)
from .large_group import _sample_large_group_pairs
from .validation import _validate_pair_sampling

__all__ = [
    "_candidate_pair_count",
    "_canonical_pair",
    "_pair_row",
    "_pair_sample_count",
    "_sample_large_group_pairs",
    "_should_enumerate_pairs",
    "_validate_pair_sampling",
    "_weighted_choice",
]
