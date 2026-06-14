#!/usr/bin/env python3
"""Merge ready binding `records.parquet` files into one all-records table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from affinity_transformer.dataset import load_records  # noqa: E402


def merge_binding_records(
    manifest_path: Path,
    processed_root: Path,
    output_path: Path,
    summary_path: Path,
) -> pd.DataFrame:
    """Merge all manifest rows with `status == "ready"`.

    Args:
        manifest_path: `scripts/prepare/binding/manifest.csv`.
        processed_root: Root containing `{study_id}/{table_id}/records.parquet`.
        output_path: Destination for the merged parquet table.
        summary_path: Destination for the per-dataset CSV summary.

    Returns:
        The merged records DataFrame.

    Raises:
        FileNotFoundError: If the manifest or any ready records file is missing.
        ValueError: If manifest columns are missing, no rows are ready, or
            merged `record_id` values are not globally unique.
    """
    manifest_path = Path(manifest_path)
    processed_root = Path(processed_root)
    output_path = Path(output_path)
    summary_path = Path(summary_path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    required_manifest = ("study_id", "table_id", "status")
    missing = [column for column in required_manifest if column not in manifest.columns]
    if missing:
        raise ValueError(f"Manifest is missing required column(s): {missing}")

    ready = manifest[manifest["status"].astype(str) == "ready"].copy()
    if ready.empty:
        raise ValueError("Manifest contains no status='ready' rows")

    tables: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for row in ready.sort_values(["study_id", "table_id"]).itertuples(index=False):
        study_id = str(getattr(row, "study_id"))
        table_id = str(getattr(row, "table_id"))
        records_path = _records_path(processed_root, study_id, table_id)
        records = load_records(records_path)

        expected_dataset_id = f"{study_id}/{table_id}"
        if "dataset_id" in records and not set(records["dataset_id"].astype(str)) <= {expected_dataset_id}:
            raise ValueError(
                f"{records_path} contains dataset_id values other than {expected_dataset_id!r}"
            )

        tables.append(records)
        summary_rows.append(_summary_row(study_id, table_id, records_path, records))

    merged = pd.concat(tables, ignore_index=True)
    duplicated = merged.loc[merged["record_id"].astype(str).duplicated(), "record_id"].tolist()
    if duplicated:
        raise ValueError(f"Duplicate record_id values across ready datasets: {duplicated[:20]}")

    merged = merged.sort_values(["dataset_id", "record_id"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    pd.DataFrame(summary_rows).sort_values(["study_id", "table_id"]).to_csv(
        summary_path, index=False
    )
    return merged


def _records_path(processed_root: Path, study_id: str, table_id: str) -> Path:
    parquet_path = processed_root / study_id / table_id / "records.parquet"
    if parquet_path.exists():
        return parquet_path
    csv_path = processed_root / study_id / table_id / "records.csv"
    if csv_path.exists():
        return csv_path
    raise FileNotFoundError(
        f"No records.parquet or records.csv found for {study_id}/{table_id} under {processed_root}"
    )


def _summary_row(study_id: str, table_id: str, records_path: Path, records: pd.DataFrame) -> dict[str, object]:
    trainable = _trainable_records(records)
    return {
        "dataset_id": f"{study_id}/{table_id}",
        "study_id": study_id,
        "table_id": table_id,
        "n_records": len(records),
        "n_trainable_records": len(trainable),
        "n_groups": int(records["group_id"].nunique()),
        "n_trainable_groups": int(trainable["group_id"].nunique()) if not trainable.empty else 0,
        "label_kind_counts": _value_counts_json(records, "label_kind"),
        "antigen_source_counts": _value_counts_json(records, "antigen_source"),
        "records_path": str(records_path),
    }


def _trainable_records(records: pd.DataFrame) -> pd.DataFrame:
    keep = records["keep_for_training"].apply(_parse_bool)
    labels = pd.to_numeric(records["rank_label"], errors="coerce")
    finite = labels.apply(lambda value: pd.notna(value) and value not in (float("inf"), float("-inf")))
    return records.loc[keep & finite]


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


def _value_counts_json(records: pd.DataFrame, column: str) -> str:
    counts = records[column].fillna("<NA>").astype(str).value_counts().sort_index().to_dict()
    return json.dumps(counts, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("scripts/prepare/binding/manifest.csv"))
    parser.add_argument("--processed-root", type=Path, default=Path("processed/binding"))
    parser.add_argument("--output", type=Path, default=Path("processed/binding/all_records.parquet"))
    parser.add_argument("--summary", type=Path, default=Path("processed/binding/all_records_summary.csv"))
    args = parser.parse_args()

    merged = merge_binding_records(args.manifest, args.processed_root, args.output, args.summary)
    print(f"merged rows={len(merged)} -> {args.output}")
    print(f"summary -> {args.summary}")


if __name__ == "__main__":
    main()
