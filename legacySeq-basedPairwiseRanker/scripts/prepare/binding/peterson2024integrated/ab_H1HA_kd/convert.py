#!/usr/bin/env python3
"""Convert raw data to standard training table.

Usage (from repo root):
    python3 {script_rel} <source_file> <output_dir>

Or via prepare.sh (recommended).
"""
import math, sys, datetime
from pathlib import Path
import pandas as pd

_VALID_AA = frozenset("ACDEFGHIKLMNPQRSTVWYX")  # X = any AA (IUPAC), handled by ESMC


def _seq(val) -> "str | None":
    if val is None:
        return None
    s = str(val).strip().upper()
    if not s or s in ("NAN", "NONE", ""):
        return None
    return s if all(c in _VALID_AA for c in s) else None


# ── CONFIG ───────────────────────────────────────────────────────────────────
# NOTE: Column is named KD but values range 2.68–2490.81, consistent with nM.
# Treating as nM despite the column name; see retrieval_notes in antigen_missing_summary.csv
STUDY_ID       = "peterson2024integrated"
TABLE_ID       = "ab_H1HA_kd"
SOURCE_FILE    = "data/binding/peterson2024integrated_ab_H1HA_kd.csv"
ANTIBODY_TYPE  = "IgG"
ANTIGEN_KEY    = "H1_HA"
ANTIGEN_NAME   = "Influenza A H1 hemagglutinin"
ANTIGEN_SEQ    = None
ANTIGEN_SOURCE = "missing"
ASSAY_NAME     = "SPR"
ASSAY_TYPE     = "binding"
METRIC_NAME    = "neg_log10_kd_M"
METRIC_UNIT    = "-log10(KD/M)"
METRIC_DIRECTION = "higher_is_better"
TRANSFORM_RULE = "rank_label = -log10(KD_nM * 1e-9)  [column values treated as nM]"
LABEL_KIND     = "experimental"
GROUP_ID       = "peterson2024integrated/ab_H1HA_kd/H1_HA/neg_log10_kd_M/experimental"


def _rl(raw) -> "float | None":
    # Column KD values (2.68–2490.81) are consistent with nM despite label.
    try:
        v = -math.log10(float(raw) * 1e-9)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _write_outputs(records: list, out: Path, study_id: str, table_id: str,
                   source_file: str) -> None:
    df = pd.DataFrame(records)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "records.parquet", index=False)
    df.to_csv(out / "records.csv", index=False)

    keep_mask = df["keep_for_training"].astype(bool)
    n_keep = int(keep_mask.sum())
    n_drop = int((~keep_mask).sum())
    n_groups = int(df.loc[keep_mask, "group_id"].nunique()) if n_keep else 0

    qc = pd.DataFrame([{
        "study_id": study_id, "table_id": table_id,
        "source_file": source_file,
        "total": len(df), "keep": n_keep, "drop": n_drop,
        "n_groups": n_groups,
        "generated_at": datetime.datetime.now().isoformat(),
    }])
    qc.to_csv(out / "qc_summary.csv", index=False)

    drop_cols = ["record_id", "source_file", "source_row",
                 "heavy_chain", "metric_value_raw", "drop_reason"]
    dropped = df[~keep_mask][drop_cols]
    dropped.to_csv(out / "dropped_records.csv", index=False)

    print(f"[{study_id}/{table_id}]  total={len(df)}  keep={n_keep}  drop={n_drop}")
    print(f"  -> {out}/records.parquet")



def convert(src: Path, out: Path) -> None:
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Converting {src} ...")
    df = pd.read_csv(src, low_memory=False)

    records = []
    for i, row in df.iterrows():
        source_row = i + 2
        heavy = _seq(row.get("heavy"))
        light = _seq(row.get("light"))
        raw_v = row.get("fitness")
        rl    = _rl(raw_v)

        drops = []
        if heavy is None:
            drops.append("missing_or_invalid_heavy_chain")
        if rl is None:
            drops.append("missing_rank_label")

        records.append({
            "record_id":          f"{STUDY_ID}/{TABLE_ID}/{source_row}",
            "dataset_id":         f"{STUDY_ID}/{TABLE_ID}",
            "study_id":           STUDY_ID,
            "table_id":           TABLE_ID,
            "source_file":        SOURCE_FILE,
            "source_row":         source_row,
            "antibody_id":        None,
            "antibody_type":      ANTIBODY_TYPE,
            "heavy_chain":        heavy,
            "light_chain":        light,
            "single_chain_sequence": None,
            "antigen_key":        ANTIGEN_KEY,
            "antigen_name":       ANTIGEN_NAME,
            "antigen_sequence":   ANTIGEN_SEQ,
            "antigen_source":     ANTIGEN_SOURCE,
            "assay_name":         ASSAY_NAME,
            "assay_type":         ASSAY_TYPE,
            "metric_name":        METRIC_NAME,
            "metric_value_raw":   str(raw_v),
            "metric_value_numeric": (float(raw_v) if raw_v is not None else None),
            "metric_unit":        METRIC_UNIT,
            "metric_direction":   METRIC_DIRECTION,
            "transform_rule":     TRANSFORM_RULE,
            "rank_label":         rl,
            "label_kind":         LABEL_KIND,
            "group_id":           GROUP_ID,
            "keep_for_training":  len(drops) == 0,
            "drop_reason":        "; ".join(drops) if drops else None,
        })

    _write_outputs(records, out, STUDY_ID, TABLE_ID, SOURCE_FILE)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {Path(sys.argv[0]).name} <source_file> <output_dir>",
              file=sys.stderr)
        sys.exit(1)
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
