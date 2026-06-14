#!/usr/bin/env python3
"""Write QC summaries for a standard processed records table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from affinity_transformer.dataset import load_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input records.parquet/csv")
    parser.add_argument("--output", required=True, type=Path, help="Single-row QC summary CSV")
    parser.add_argument(
        "--by-dataset",
        type=Path,
        default=None,
        help="Optional dataset-level QC summary CSV.",
    )
    args = parser.parse_args()

    records = load_records(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build_overall_summary(records).to_csv(args.output, index=False)
    print(f"overall QC -> {args.output}")

    if args.by_dataset is not None:
        args.by_dataset.parent.mkdir(parents=True, exist_ok=True)
        build_dataset_summary(records).to_csv(args.by_dataset, index=False)
        print(f"dataset QC -> {args.by_dataset}")


def build_overall_summary(records: pd.DataFrame) -> pd.DataFrame:
    trainable = _trainable(records)
    row = {
        "n_records": len(records),
        "n_trainable_records": len(trainable),
        "n_groups": _nunique(records, "group_id"),
        "n_trainable_groups": _nunique(trainable, "group_id"),
        "n_spearman_eligible_groups": _n_spearman_eligible_groups(trainable),
        "n_dataset_ids": _nunique(records, "dataset_id"),
        "n_study_ids": _nunique(records, "study_id"),
        "n_antigen_keys": _nunique(records, "antigen_key"),
        "n_antibody_ids": _nunique(records, "antibody_id"),
        "n_missing_antigen_sequence": _n_missing(records, "antigen_sequence"),
        "label_kind_counts": _value_counts_json(records, "label_kind"),
        "antigen_source_counts": _value_counts_json(records, "antigen_source"),
        "antibody_type_counts": _value_counts_json(records, "antibody_type"),
    }
    return pd.DataFrame([row])


def build_dataset_summary(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_id, dataset_records in records.groupby("dataset_id", sort=True):
        trainable = _trainable(dataset_records)
        rows.append({
            "dataset_id": dataset_id,
            "n_records": len(dataset_records),
            "n_trainable_records": len(trainable),
            "n_groups": _nunique(dataset_records, "group_id"),
            "n_trainable_groups": _nunique(trainable, "group_id"),
            "n_spearman_eligible_groups": _n_spearman_eligible_groups(trainable),
            "n_antigen_keys": _nunique(dataset_records, "antigen_key"),
            "n_missing_antigen_sequence": _n_missing(dataset_records, "antigen_sequence"),
            "label_kind_counts": _value_counts_json(dataset_records, "label_kind"),
            "antigen_source_counts": _value_counts_json(dataset_records, "antigen_source"),
            "antibody_type_counts": _value_counts_json(dataset_records, "antibody_type"),
        })
    return pd.DataFrame(rows)


def _trainable(records: pd.DataFrame) -> pd.DataFrame:
    keep = records["keep_for_training"].map(_parse_bool)
    labels = pd.to_numeric(records["rank_label"], errors="coerce")
    finite = labels.notna() & labels.map(lambda value: value not in (float("inf"), float("-inf")))
    return records.loc[keep & finite]


def _n_spearman_eligible_groups(records: pd.DataFrame) -> int:
    if records.empty:
        return 0
    labels = records.copy()
    labels["rank_label_numeric"] = pd.to_numeric(labels["rank_label"], errors="coerce")
    stats = labels.groupby("group_id")["rank_label_numeric"].agg(["size", "nunique"])
    return int(((stats["size"] >= 2) & (stats["nunique"] >= 2)).sum())


def _nunique(records: pd.DataFrame, column: str) -> int:
    if column not in records:
        return 0
    return int(records[column].nunique(dropna=True))


def _n_missing(records: pd.DataFrame, column: str) -> int:
    if column not in records:
        return 0
    values = records[column].map(_nullable_text)
    return int(values.isin({"", "nan", "none", "null"}).sum())


def _value_counts_json(records: pd.DataFrame, column: str) -> str:
    if column not in records:
        return "{}"
    counts = records[column].fillna("<NA>").astype(str).value_counts().sort_index().to_dict()
    return json.dumps(counts, sort_keys=True)


def _nullable_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower()


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


if __name__ == "__main__":
    main()
