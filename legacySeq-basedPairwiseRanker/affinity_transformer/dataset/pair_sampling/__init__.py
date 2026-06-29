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
from .noise_aware_multiscale import _noise_aware_multiscale_pairs
from .randomized_tree import _randomized_bst_pairs
from .tau_registry import resolve_tau_for_group
from .tree import _balanced_tree_pairs
from .validation import _validate_pair_sampling

__all__ = [
    "_balanced_tree_pairs",
    "_candidate_pair_count",
    "_canonical_pair",
    "_noise_aware_multiscale_pairs",
    "_pair_row",
    "_pair_sample_count",
    "_randomized_bst_pairs",
    "_sample_large_group_pairs",
    "_should_enumerate_pairs",
    "_validate_pair_sampling",
    "_weighted_choice",
    "resolve_tau_for_group",
]
