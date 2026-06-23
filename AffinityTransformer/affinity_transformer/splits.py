"""Compatibility facade for the split-protocol package.

The implementation now lives in ``affinity_transformer.splitting`` (one module
per concern). This module re-exports the public API so existing imports such as
``from affinity_transformer.splits import build_splits`` keep working
unchanged. There is no logic here -- exactly one implementation of each
behavior, in the ``splitting`` subpackage.
"""

from __future__ import annotations

from .splitting.audits import (
    ELIGIBILITY_COLUMNS,
    ENTITY_UNIT_COLUMNS,
    LEAKAGE_COLUMNS,
    PINNED_GROUPS_COLUMNS,
    SPLIT_COLUMNS,
)
from .splitting.common import COLD_START_IDENTITY_COLUMNS, frame_hash
from .splitting.dispatch import (
    AUXILIARY_STRATEGIES,
    VALID_STRATEGIES,
    build_splits,
)
from .splitting.entity_cold_start import (
    ENTITY_COLD_START_STRATEGIES,
    _select_protocol_eligible_records,
    build_antibody_cold_start_kfolds,
    build_antibody_cold_start_split,
    build_antigen_cold_start_kfolds,
    build_antigen_cold_start_split,
)
from .splitting.group import build_group_kfolds
from .splitting.results import (
    EntityColdStartFold,
    EntityColdStartSplitResult,
    GroupFold,
    SplitResult,
    WithinAntigenSplitResult,
    build_antibody_cold_start_manifest,
    build_antigen_cold_start_manifest,
    write_antibody_cold_start_split,
    write_antigen_cold_start_split,
    write_entity_cold_start_split,
    write_splits,
    write_within_antigen_split,
)
from .splitting.within_antigen import build_within_antigen_split

__all__ = [
    # strategy registries / column constants
    "VALID_STRATEGIES",
    "AUXILIARY_STRATEGIES",
    "ENTITY_COLD_START_STRATEGIES",
    "COLD_START_IDENTITY_COLUMNS",
    "SPLIT_COLUMNS",
    "LEAKAGE_COLUMNS",
    "PINNED_GROUPS_COLUMNS",
    "ENTITY_UNIT_COLUMNS",
    "ELIGIBILITY_COLUMNS",
    # result dataclasses
    "SplitResult",
    "WithinAntigenSplitResult",
    "GroupFold",
    "EntityColdStartSplitResult",
    "EntityColdStartFold",
    # generic fixed splits + group K-fold
    "build_splits",
    "write_splits",
    "build_group_kfolds",
    # within-antigen auxiliary split
    "build_within_antigen_split",
    "write_within_antigen_split",
    # entity cold-start (antibody + antigen)
    "build_antibody_cold_start_split",
    "build_antigen_cold_start_split",
    "build_antibody_cold_start_kfolds",
    "build_antigen_cold_start_kfolds",
    "write_entity_cold_start_split",
    "write_antibody_cold_start_split",
    "write_antigen_cold_start_split",
    "build_antibody_cold_start_manifest",
    "build_antigen_cold_start_manifest",
    # shared helpers re-exported for existing call sites
    "frame_hash",
    "_select_protocol_eligible_records",
]
