"""debug_record_split strategy (random record-level split)."""
from __future__ import annotations

import pandas as pd

from .common import _partition_units, _rows_for_values


def _split_by_record(
    records: pd.DataFrame,
    valid_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    record_ids = sorted(records["record_id"].astype(str).tolist())
    train_ids, valid_ids, test_ids = _partition_units(record_ids, valid_fraction, test_fraction, seed)
    return (
        _rows_for_values(records, "record_id", train_ids),
        _rows_for_values(records, "record_id", valid_ids),
        _rows_for_values(records, "record_id", test_ids),
    )
