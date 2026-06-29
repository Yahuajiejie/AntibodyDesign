"""Load processed records and filter the trainable subset."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from .schema import REQUIRED_COLUMNS


def load_records(path: Path) -> pd.DataFrame:
    """Load a standard processed table and check required columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Processed table not found: {path}")

    if path.suffix == ".parquet":
        records = pd.read_parquet(path)
    elif path.suffix == ".csv":
        records = pd.read_csv(path, low_memory=False)
    else:
        raise ValueError(f"Unsupported processed table extension: {path.suffix!r} ({path})")

    missing = [c for c in REQUIRED_COLUMNS if c not in records.columns]
    if missing:
        raise ValueError(f"Processed table {path} is missing required column(s): {missing}")

    return records


def _is_finite_number(value: object) -> bool:
    """Return True if ``value`` can be interpreted as a finite float."""
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _parse_bool(value: object) -> bool:
    """Parse a standard-table boolean cell strictly."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, float) and value in (0.0, 1.0):
        return bool(int(value))
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "t", "1", "yes", "y"}:
            return True
        if normalized in {"false", "f", "0", "no", "n"}:
            return False
    raise ValueError(f"Invalid boolean value for keep_for_training: {value!r}")


def filter_trainable_records(records: pd.DataFrame) -> pd.DataFrame:
    """Keep only records with ``keep_for_training=True`` and finite label."""
    required = ("keep_for_training", "rank_label")
    missing = [c for c in required if c not in records.columns]
    if missing:
        raise ValueError(f"records is missing required column(s): {missing}")

    keep_mask = records["keep_for_training"].apply(_parse_bool)
    finite_mask = records["rank_label"].apply(_is_finite_number)
    return records[keep_mask & finite_mask].reset_index(drop=True)
