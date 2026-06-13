"""Train/validation/test splitting for standard processed records.

This module consumes only processed tables that already follow the standard
schema. It does not read raw CSVs, does not derive labels, and does not build
pairs. Its job is to create split files and explicit leakage reports before
training starts.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import pandas as pd

from .utils import ensure_dir

SPLIT_COLUMNS = ("split", "strategy", "n_records", "n_trainable_records", "n_groups",
                 "n_trainable_groups", "n_spearman_eligible_groups",
                 "label_kind_counts", "antigen_source_counts")
LEAKAGE_COLUMNS = ("check_name", "status", "n_violations", "details")

VALID_STRATEGIES = {"debug_record_split", "group_holdout_split"}


@dataclass
class SplitResult:
    """DataFrames and QC reports produced by `build_splits`."""

    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    summary: pd.DataFrame
    leakage_report: pd.DataFrame


def build_splits(
    records: pd.DataFrame,
    strategy: str,
    valid_fraction: float,
    test_fraction: float,
    seed: int,
) -> SplitResult:
    """Build train/valid/test splits from one merged processed table.

    Args:
        records: Standard processed records.
        strategy: "debug_record_split" or "group_holdout_split".
        valid_fraction: Fraction of units reserved for validation.
        test_fraction: Fraction of units reserved for test.
        seed: Random seed controlling unit shuffling.

    Returns:
        `SplitResult` containing train/valid/test records and two QC reports.

    Raises:
        ValueError: If required columns are missing, the strategy/fractions
            are invalid, too few units exist, or a leakage check fails.
    """
    _validate_inputs(records, strategy, valid_fraction, test_fraction)

    if strategy == "debug_record_split":
        train, valid, test = _split_by_record(records, valid_fraction, test_fraction, seed)
    elif strategy == "group_holdout_split":
        train, valid, test = _split_by_group(records, valid_fraction, test_fraction, seed)
    else:  # guarded by _validate_inputs
        raise ValueError(f"Unsupported split strategy: {strategy!r}")

    summary = _build_summary(strategy, train=train, valid=valid, test=test)
    leakage_report = _build_leakage_report(strategy, train=train, valid=valid, test=test)
    failed = leakage_report[leakage_report["status"] != "PASS"]
    if not failed.empty:
        raise ValueError(f"Split leakage check failed: {failed.to_dict(orient='records')}")

    return SplitResult(train=train, valid=valid, test=test, summary=summary,
                       leakage_report=leakage_report)


def write_splits(result: SplitResult, output_dir: Path) -> None:
    """Write split records and QC reports to `output_dir`."""
    output_dir = ensure_dir(Path(output_dir))
    result.train.to_parquet(output_dir / "train.parquet", index=False)
    result.valid.to_parquet(output_dir / "valid.parquet", index=False)
    result.test.to_parquet(output_dir / "test.parquet", index=False)
    result.summary.to_csv(output_dir / "split_summary.csv", index=False)
    result.leakage_report.to_csv(output_dir / "leakage_report.csv", index=False)


def _validate_inputs(
    records: pd.DataFrame,
    strategy: str,
    valid_fraction: float,
    test_fraction: float,
) -> None:
    required = ("record_id", "group_id", "keep_for_training", "rank_label")
    missing = [column for column in required if column not in records.columns]
    if missing:
        raise ValueError(f"records is missing required column(s): {missing}")
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"strategy must be one of {sorted(VALID_STRATEGIES)}, got {strategy!r}")
    if valid_fraction < 0 or test_fraction < 0:
        raise ValueError("valid_fraction and test_fraction must be non-negative")
    if not (0.0 < valid_fraction + test_fraction < 1.0):
        raise ValueError("valid_fraction + test_fraction must be > 0 and < 1")
    if records["record_id"].isna().any():
        raise ValueError("records contains null record_id values")
    if records["record_id"].astype(str).duplicated().any():
        duplicated = records.loc[records["record_id"].astype(str).duplicated(), "record_id"].tolist()
        raise ValueError(f"records contains duplicate record_id values: {duplicated[:10]}")


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


def _split_by_group(
    records: pd.DataFrame,
    valid_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_ids = sorted(records["group_id"].astype(str).unique().tolist())
    train_groups, valid_groups, test_groups = _partition_units(
        group_ids, valid_fraction, test_fraction, seed
    )
    return (
        _rows_for_values(records, "group_id", train_groups),
        _rows_for_values(records, "group_id", valid_groups),
        _rows_for_values(records, "group_id", test_groups),
    )


def _partition_units(
    units: list[str],
    valid_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[set[str], set[str], set[str]]:
    if len(units) < 3:
        raise ValueError(
            f"At least 3 split units are required to create train/valid/test, got {len(units)}"
        )

    shuffled = list(units)
    random.Random(seed).shuffle(shuffled)

    n_units = len(shuffled)
    n_valid = _fraction_count(n_units, valid_fraction)
    n_test = _fraction_count(n_units, test_fraction)
    while n_valid + n_test >= n_units:
        if n_test >= n_valid and n_test > 0:
            n_test -= 1
        elif n_valid > 0:
            n_valid -= 1
        else:
            break

    if n_valid == 0 and valid_fraction > 0:
        n_valid = 1
    if n_test == 0 and test_fraction > 0 and n_valid + n_test < n_units - 1:
        n_test = 1
    while n_valid + n_test >= n_units:
        if n_test >= n_valid and n_test > 0:
            n_test -= 1
        else:
            n_valid -= 1

    test_units = set(shuffled[:n_test])
    valid_units = set(shuffled[n_test:n_test + n_valid])
    train_units = set(shuffled[n_test + n_valid:])
    if not train_units or not valid_units or not test_units:
        raise ValueError(
            "Split fractions produced an empty train, valid, or test split; "
            f"n_units={n_units}, n_valid={n_valid}, n_test={n_test}"
        )
    return train_units, valid_units, test_units


def _fraction_count(n_units: int, fraction: float) -> int:
    if fraction <= 0:
        return 0
    return max(1, int(round(n_units * fraction)))


def _rows_for_values(records: pd.DataFrame, column: str, values: set[str]) -> pd.DataFrame:
    mask = records[column].astype(str).isin(values)
    return records.loc[mask].sort_values(["dataset_id", "record_id"]).reset_index(drop=True)


def _build_summary(strategy: str, **splits: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split_name, split_records in splits.items():
        trainable = _trainable_records(split_records)
        rows.append({
            "split": split_name,
            "strategy": strategy,
            "n_records": len(split_records),
            "n_trainable_records": len(trainable),
            "n_groups": int(split_records["group_id"].nunique()) if "group_id" in split_records else 0,
            "n_trainable_groups": int(trainable["group_id"].nunique()) if not trainable.empty else 0,
            "n_spearman_eligible_groups": _count_spearman_eligible_groups(trainable),
            "label_kind_counts": _value_counts_json(split_records, "label_kind"),
            "antigen_source_counts": _value_counts_json(split_records, "antigen_source"),
        })
    return pd.DataFrame(rows, columns=SPLIT_COLUMNS)


def _build_leakage_report(strategy: str, **splits: pd.DataFrame) -> pd.DataFrame:
    rows = [_overlap_report("record_id_overlap", splits, "record_id")]
    if strategy == "group_holdout_split":
        rows.append(_overlap_report("group_id_overlap", splits, "group_id"))
    return pd.DataFrame(rows, columns=LEAKAGE_COLUMNS)


def _overlap_report(check_name: str, splits: dict[str, pd.DataFrame], column: str) -> dict[str, object]:
    violations: list[str] = []
    for left_name, right_name in combinations(splits, 2):
        left = set(splits[left_name][column].astype(str))
        right = set(splits[right_name][column].astype(str))
        overlap = sorted(left & right)
        if overlap:
            violations.append(f"{left_name}/{right_name}: {overlap[:10]}")

    return {
        "check_name": check_name,
        "status": "PASS" if not violations else "FAIL",
        "n_violations": len(violations),
        "details": "; ".join(violations),
    }


def _trainable_records(records: pd.DataFrame) -> pd.DataFrame:
    keep = records["keep_for_training"].apply(_parse_bool)
    labels = pd.to_numeric(records["rank_label"], errors="coerce")
    finite = labels.apply(lambda value: pd.notna(value) and value not in (float("inf"), float("-inf")))
    return records.loc[keep & finite].copy()


def _parse_bool(value: object) -> bool:
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


def _count_spearman_eligible_groups(records: pd.DataFrame) -> int:
    if records.empty:
        return 0
    count = 0
    for _, group in records.groupby("group_id"):
        if pd.to_numeric(group["rank_label"], errors="coerce").nunique() >= 2:
            count += 1
    return count


def _value_counts_json(records: pd.DataFrame, column: str) -> str:
    if column not in records:
        return "{}"
    counts = records[column].fillna("<NA>").astype(str).value_counts().sort_index().to_dict()
    return json.dumps(counts, sort_keys=True)
