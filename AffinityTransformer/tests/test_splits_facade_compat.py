"""Backward-compatibility guard for the splits package refactor.

After splitting ``affinity_transformer/splits.py`` into the
``affinity_transformer.splitting`` package, the original module must remain a
compatibility facade: every previously public symbol stays importable from
``affinity_transformer.splits`` and resolves to the implementation that now
lives in the subpackage.

Note: ``group_holdout_split`` is intentionally NOT tested as an importable
symbol -- it has never been a public function, only a strategy string passed
to ``build_splits``. The refactor preserves today's surface exactly and does
not add it.
"""

from __future__ import annotations

import affinity_transformer.splits as splits

# The exact imports called out in the refactor brief.
def test_documented_imports_resolve():
    from affinity_transformer.splits import build_splits  # noqa: F401
    from affinity_transformer.splits import build_group_kfolds  # noqa: F401
    from affinity_transformer.splits import build_antibody_cold_start_split  # noqa: F401
    from affinity_transformer.splits import build_within_antigen_split  # noqa: F401


PUBLIC_SURFACE = (
    "VALID_STRATEGIES", "AUXILIARY_STRATEGIES", "ENTITY_COLD_START_STRATEGIES",
    "COLD_START_IDENTITY_COLUMNS", "SPLIT_COLUMNS", "LEAKAGE_COLUMNS",
    "PINNED_GROUPS_COLUMNS", "ENTITY_UNIT_COLUMNS", "ELIGIBILITY_COLUMNS",
    "SplitResult", "WithinAntigenSplitResult", "GroupFold",
    "EntityColdStartSplitResult", "EntityColdStartFold",
    "build_splits", "write_splits", "build_group_kfolds",
    "build_within_antigen_split", "write_within_antigen_split",
    "build_antibody_cold_start_split", "build_antigen_cold_start_split",
    "build_antibody_cold_start_kfolds", "build_antigen_cold_start_kfolds",
    "write_entity_cold_start_split", "write_antibody_cold_start_split",
    "build_antibody_cold_start_manifest", "frame_hash",
    "_select_protocol_eligible_records",
)


def test_full_public_surface_present():
    missing = [name for name in PUBLIC_SURFACE if not hasattr(splits, name)]
    assert not missing, f"facade missing symbols: {missing}"


def test_facade_delegates_to_splitting_package():
    # The facade itself defines no implementation; symbols come from splitting.*.
    assert splits.build_splits.__module__ == "affinity_transformer.splitting.dispatch"
    assert (
        splits.build_group_kfolds.__module__
        == "affinity_transformer.splitting.group"
    )
    assert (
        splits.build_within_antigen_split.__module__
        == "affinity_transformer.splitting.within_antigen"
    )
    assert (
        splits.build_antibody_cold_start_split.__module__
        == "affinity_transformer.splitting.entity_cold_start"
    )
    assert (
        splits.build_antigen_cold_start_split.__module__
        == "affinity_transformer.splitting.entity_cold_start"
    )
