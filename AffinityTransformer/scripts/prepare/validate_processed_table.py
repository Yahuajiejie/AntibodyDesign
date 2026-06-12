#!/usr/bin/env python3
"""
Validate a processed binding table against the standard schema (§3).

Usage:
    python3 scripts/prepare/validate_processed_table.py \
        processed/binding/phillips2021binding/cr6261_h1_kd/records.parquet

Exit 0 = PASS, 1 = FAIL.
"""
import math
import sys
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "record_id", "dataset_id", "study_id", "table_id",
    "source_file", "source_row",
    "antibody_id", "antibody_type",
    "heavy_chain", "light_chain", "single_chain_sequence",
    "antigen_key", "antigen_name", "antigen_sequence", "antigen_source",
    "assay_name", "assay_type",
    "metric_name", "metric_value_raw", "metric_value_numeric",
    "metric_unit", "metric_direction", "transform_rule",
    "rank_label", "label_kind",
    "group_id", "keep_for_training", "drop_reason",
]

_VALID_ANTIBODY_TYPE  = {"Fv", "scFv", "VHH", "Fab", "IgG", "unknown"}
_VALID_ANTIGEN_SOURCE = {"provided", "retrieved", "missing"}
_VALID_ASSAY_TYPE     = {"binding", "neutralization", "fitness", "expression", "unknown"}
_VALID_METRIC_DIR     = {"higher_is_better", "lower_is_better", "unknown"}
_VALID_LABEL_KIND     = {"experimental", "predicted", "binary", "unknown"}
_VALID_AA             = frozenset("ACDEFGHIKLMNPQRSTVWYX")  # X = any AA (IUPAC), handled by ESMC


def _validate(path: Path) -> list[str]:
    errors: list[str] = []

    # ── load ──────────────────────────────────────────────────────────────────
    try:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix == ".csv":
            df = pd.read_csv(path, low_memory=False)
        else:
            return [f"Unknown extension: {path.suffix}"]
    except Exception as exc:
        return [f"Failed to load file: {exc}"]

    # 1. Required columns present
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return errors  # can't proceed without schema

    n = len(df)
    keep_mask = df["keep_for_training"].astype(bool)
    keep = df[keep_mask]

    # 2. record_id: non-null, unique
    if df["record_id"].isna().any():
        errors.append("record_id has null values")
    dups = df["record_id"].duplicated().sum()
    if dups:
        errors.append(f"record_id has {int(dups)} duplicate(s)")

    # 3. group_id: non-null, non-empty
    bad_gid = (df["group_id"].isna() | (df["group_id"].astype(str).str.strip() == "")).sum()
    if bad_gid:
        errors.append(f"group_id null/empty in {int(bad_gid)} row(s)")

    # 4. rank_label finite for keep_for_training rows
    def _bad_rl(x):
        if x is None or (isinstance(x, float) and not math.isfinite(x)):
            return True
        try:
            return not math.isfinite(float(x))
        except Exception:
            return True

    bad_rl = keep["rank_label"].apply(_bad_rl).sum()
    if bad_rl:
        errors.append(
            f"{int(bad_rl)} keep_for_training=True record(s) have null/non-finite rank_label"
        )

    # 5. keep_for_training rows must have heavy_chain
    no_heavy = keep["heavy_chain"].isna().sum()
    if no_heavy:
        errors.append(
            f"{int(no_heavy)} keep_for_training=True record(s) missing heavy_chain"
        )

    # 6. Amino acid validity in sequences
    for col in ("heavy_chain", "light_chain", "single_chain_sequence"):
        for idx, val in df[col].dropna().items():
            bad_aa = [c for c in str(val).upper() if c not in _VALID_AA]
            if bad_aa:
                rid = df.at[idx, "record_id"]
                errors.append(
                    f"Non-standard AA {set(bad_aa)} in {col} for record {rid!r}"
                )
                break  # report first violation per column

    # 7. Enum columns
    enum_checks = {
        "antibody_type":  _VALID_ANTIBODY_TYPE,
        "antigen_source": _VALID_ANTIGEN_SOURCE,
        "assay_type":     _VALID_ASSAY_TYPE,
        "metric_direction": _VALID_METRIC_DIR,
        "label_kind":     _VALID_LABEL_KIND,
    }
    for col, valid in enum_checks.items():
        bad_vals = df[~df[col].isin(valid)][col].dropna().unique()
        if len(bad_vals):
            errors.append(f"Invalid {col} value(s): {list(bad_vals)}")

    # 8. source_row is positive integer
    try:
        bad_sr = (df["source_row"].astype(int) < 2).sum()
        if bad_sr:
            errors.append(f"{int(bad_sr)} source_row value(s) < 2")
    except Exception:
        errors.append("source_row is not integer-castable")

    # ── summary ───────────────────────────────────────────────────────────────
    n_keep = int(keep_mask.sum())
    n_drop = n - n_keep
    if not errors:
        print(f"  rows={n}  keep={n_keep}  drop={n_drop}")

    return errors


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: validate_processed_table.py <records.parquet|records.csv>",
            file=sys.stderr,
        )
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    errors = _validate(path)
    if errors:
        print(f"FAIL  {path}")
        for e in errors:
            print(f"  ✗  {e}")
        sys.exit(1)
    else:
        print(f"PASS  {path}")
        sys.exit(0)


if __name__ == "__main__":
    main()
