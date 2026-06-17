"""Dataclasses shared by record, pairwise, and listwise datasets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AffinityExample:
    """One trainable antibody-antigen record."""

    record_id: str
    dataset_id: str
    heavy_chain: str | None
    light_chain: str | None
    single_chain_sequence: str | None
    antibody_type: str
    antigen_sequence: str | None
    antigen_key: str | None
    rank_label: float
    label_kind: str
    group_id: str


@dataclass(frozen=True)
class AffinityPairExample:
    """One pairwise ranking example."""

    pair_id: str
    group_id: str
    left: AffinityExample
    right: AffinityExample
    y_ij: float


@dataclass(frozen=True)
class AffinityGroupExample:
    """One listwise ranking example: every surviving record of one group."""

    group_id: str
    label_kind: str
    examples: tuple[AffinityExample, ...]


@dataclass(frozen=True)
class _LabelBlock:
    """Rank-label quantile block used by the large-group pair sampler."""

    index: int
    items: tuple[tuple[str, float], ...]
    label_to_ids: dict[float, tuple[str, ...]]
    n_records: int
