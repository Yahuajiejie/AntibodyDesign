"""Listwise ranking group construction."""

from __future__ import annotations

import random

import pandas as pd

from .records import filter_trainable_records
from .schema import GROUP_COLUMNS


def _group_member_ids(group: pd.DataFrame) -> list[str]:
    """Return trainable record IDs of one rankable group, else ``[]``."""
    labels = group["rank_label"].astype(float)
    if labels.nunique() < 2:
        return []
    return sorted(group["record_id"].astype(str))


def build_groups(
    records: pd.DataFrame,
    max_group_size: int | None,
    seed: int,
) -> pd.DataFrame:
    """Build listwise ranking groups."""
    required = ("record_id", "group_id", "rank_label", "label_kind", "keep_for_training")
    missing = [c for c in required if c not in records.columns]
    if missing:
        raise ValueError(f"records is missing required column(s): {missing}")
    if max_group_size is not None and max_group_size < 2:
        raise ValueError(f"max_group_size must be None or >= 2, got {max_group_size}")

    trainable = filter_trainable_records(records)

    rows: list[dict[str, object]] = []
    for group_id, group in trainable.groupby("group_id", sort=True):
        member_ids = _group_member_ids(group)
        if not member_ids:
            continue

        if max_group_size is not None and len(member_ids) > max_group_size:
            rng = random.Random(f"{seed}:{group_id}")
            member_ids = sorted(rng.sample(member_ids, max_group_size))

        group_indexed_by_record_id = group.set_index(group["record_id"].astype(str))
        for record_id in member_ids:
            row = group_indexed_by_record_id.loc[record_id]
            rows.append(
                dict(
                    group_id=group_id,
                    record_id=record_id,
                    rank_label=float(row["rank_label"]),
                    label_kind=str(row["label_kind"]),
                )
            )

    if not rows:
        return pd.DataFrame(columns=GROUP_COLUMNS)
    return pd.DataFrame(rows, columns=GROUP_COLUMNS)
