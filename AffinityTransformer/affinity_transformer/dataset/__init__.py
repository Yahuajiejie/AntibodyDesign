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
from . import pairs as _pairs_module
from .pair_sampling import blocks as _blocks_module
from .pair_sampling import large_group as _large_group_module
from .pairs import _candidate_pairs
from .pair_sampling.blocks import _build_label_blocks
from .records import filter_trainable_records, load_records
from .schema import GROUP_COLUMNS, PAIR_COLUMNS, REQUIRED_COLUMNS


def build_pairs(*args, **kwargs):
    """Build pairwise examples, preserving the historical module-level hook.

    A few tests monkeypatch ``affinity_transformer.dataset._candidate_pairs``
    and ``_build_label_blocks`` directly. The implementation now lives in
    submodules, so this thin wrapper syncs those package-level attributes into
    the actual worker modules before dispatching.
    """
    _pairs_module._candidate_pairs = globals()["_candidate_pairs"]
    _blocks_module._build_label_blocks = globals()["_build_label_blocks"]
    _large_group_module._build_label_blocks = globals()["_build_label_blocks"]
    return _pairs_module.build_pairs(*args, **kwargs)


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
    "_build_label_blocks",
    "_candidate_pairs",
]
