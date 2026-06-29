"""Reproducible training samplers with embedding-cache locality."""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Sampler


class GroupShuffleSampler(Sampler[int]):
    """Shuffle group blocks and their pair rows on every iteration.

    A fully global pair shuffle destroys locality in the sharded embedding
    cache.  Keeping each group contiguous retains repeated antigen/antibody
    reads while randomizing both group order and within-group pair order on
    every epoch.
    """

    def __init__(self, pairs: pd.DataFrame, seed: int) -> None:
        if "group_id" not in pairs.columns:
            raise ValueError("pairs must contain group_id")
        if pairs.empty:
            raise ValueError("pairs must be non-empty")
        self._groups = [
            list(indices)
            for _, indices in pairs.groupby("group_id", sort=True).groups.items()
        ]
        self._length = len(pairs)
        self._generator = torch.Generator().manual_seed(seed)

    def __iter__(self):
        group_order = torch.randperm(
            len(self._groups), generator=self._generator
        ).tolist()
        for group_index in group_order:
            indices = self._groups[group_index]
            within_group = torch.randperm(
                len(indices), generator=self._generator
            ).tolist()
            for position in within_group:
                yield indices[position]

    def __len__(self) -> int:
        return self._length
