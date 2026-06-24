"""Torch Dataset wrappers for processed records, pairs, and groups."""

from __future__ import annotations

import math

import pandas as pd
from torch.utils.data import Dataset

from .examples import AffinityExample, AffinityGroupExample, AffinityPairExample
from .schema import GROUP_COLUMNS, PAIR_COLUMNS, _EXAMPLE_COLUMNS


def _optional_str(value: object) -> str | None:
    """Convert a table cell to ``str``, mapping missing values to ``None``."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value)


def _row_to_example(row: pd.Series) -> AffinityExample:
    """Convert one processed-table row into an ``AffinityExample``."""
    return AffinityExample(
        record_id=str(row["record_id"]),
        dataset_id=str(row["dataset_id"]),
        heavy_chain=_optional_str(row["heavy_chain"]),
        light_chain=_optional_str(row["light_chain"]),
        single_chain_sequence=_optional_str(row["single_chain_sequence"]),
        antibody_type=str(row["antibody_type"]),
        antigen_sequence=_optional_str(row["antigen_sequence"]),
        antigen_key=_optional_str(row["antigen_key"]),
        rank_label=float(row["rank_label"]),
        label_kind=str(row["label_kind"]),
        group_id=str(row["group_id"]),
    )


class AffinityRecordDataset(Dataset):
    """Expose processed-table rows as ``AffinityExample`` items."""

    def __init__(self, records: pd.DataFrame) -> None:
        missing = [c for c in _EXAMPLE_COLUMNS if c not in records.columns]
        if missing:
            raise ValueError(f"records is missing required column(s): {missing}")
        self._records = records.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> AffinityExample:
        return _row_to_example(self._records.iloc[index])


class PairwiseAffinityDataset(Dataset):
    """Expose a ``build_pairs`` table as ``AffinityPairExample`` items."""

    def __init__(self, records: pd.DataFrame, pairs: pd.DataFrame) -> None:
        missing_records = [c for c in _EXAMPLE_COLUMNS if c not in records.columns]
        if missing_records:
            raise ValueError(f"records is missing required column(s): {missing_records}")
        missing_pairs = [c for c in PAIR_COLUMNS if c not in pairs.columns]
        if missing_pairs:
            raise ValueError(f"pairs is missing required column(s): {missing_pairs}")

        self._records = records.set_index("record_id", drop=False)
        self._pairs = pairs.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> AffinityPairExample:
        pair = self._pairs.iloc[index]
        left = _row_to_example(self._records.loc[pair["record_id_i"]])
        right = _row_to_example(self._records.loc[pair["record_id_j"]])
        return AffinityPairExample(
            pair_id=str(pair["pair_id"]),
            group_id=str(pair["group_id"]),
            left=left,
            right=right,
            y_ij=float(pair["y_ij"]),
        )


class ListwiseAffinityDataset(Dataset):
    """Expose a ``build_groups`` table as ``AffinityGroupExample`` items."""

    def __init__(self, records: pd.DataFrame, groups: pd.DataFrame) -> None:
        missing_records = [c for c in _EXAMPLE_COLUMNS if c not in records.columns]
        if missing_records:
            raise ValueError(f"records is missing required column(s): {missing_records}")
        missing_groups = [c for c in GROUP_COLUMNS if c not in groups.columns]
        if missing_groups:
            raise ValueError(f"groups is missing required column(s): {missing_groups}")

        self._records = records.set_index("record_id", drop=False)
        groups = groups.reset_index(drop=True)
        self._groups = groups

        self._group_ids: list[str] = []
        self._group_row_indices: dict[str, list[int]] = {}
        for group_id, index in groups.groupby("group_id", sort=False).groups.items():
            group_id = str(group_id)
            self._group_ids.append(group_id)
            self._group_row_indices[group_id] = list(index)

    def __len__(self) -> int:
        return len(self._group_ids)

    def __getitem__(self, index: int) -> AffinityGroupExample:
        group_id = self._group_ids[index]
        rows = self._groups.iloc[self._group_row_indices[group_id]]
        examples = tuple(
            _row_to_example(self._records.loc[record_id]) for record_id in rows["record_id"]
        )
        label_kind = str(rows.iloc[0]["label_kind"])
        return AffinityGroupExample(group_id=group_id, label_kind=label_kind, examples=examples)
