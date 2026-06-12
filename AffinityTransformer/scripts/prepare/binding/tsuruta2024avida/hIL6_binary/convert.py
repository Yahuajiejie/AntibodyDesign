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


import zipfile as _zf, io as _io, re as _re

# ── CONFIG ───────────────────────────────────────────────────────────────────
STUDY_ID       = "tsuruta2024avida"
TABLE_ID       = "hIL6_binary"
SOURCE_FILE    = "data/binding/tsuruta2024avida-hIL6_binary.csv.zip"
ANTIBODY_TYPE  = "VHH"
ANTIGEN_KEY    = "IL6"
ANTIGEN_NAME   = "Interleukin-6"
ANTIGEN_SEQ    = None
ANTIGEN_SOURCE = "missing"
ASSAY_NAME     = "AVIDa display"
ASSAY_TYPE     = "binding"
METRIC_NAME    = "binary"
METRIC_UNIT    = "0/1"
METRIC_DIRECTION = "higher_is_better"
TRANSFORM_RULE = "rank_label = float(fitness)  [0=no bind, 1=bind]"
LABEL_KIND     = "binary"
GROUP_ID       = "tsuruta2024avida/hIL6_binary/IL6/binary/binary"

_SP  = "MKYLLPTAAAGLLLLAAQPAMA"
_HIS = _re.compile(r"H{4,}$")


def _strip_vhh(seq: "str | None") -> "str | None":
    if not seq:
        return None
    s = str(seq).strip().upper()
    if s.startswith(_SP):
        s = s[len(_SP):]
    s = _HIS.sub("", s)
    return s.strip() if s else None


def _rl(raw) -> "float | None":
    try:
        v = float(raw)
        return v if v in (0.0, 1.0) else None
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
    with _zf.ZipFile(src) as zf:
        csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
        with zf.open(csv_name) as fh:
            raw_bytes = fh.read()
    df = pd.read_csv(_io.BytesIO(raw_bytes), low_memory=False)

    records = []
    for i, row in df.iterrows():
        source_row = i + 2
        raw_vhh  = str(row.get("heavy", "")).strip()
        vhh_str  = _strip_vhh(raw_vhh)
        heavy    = _seq(vhh_str)
        raw_v    = row.get("fitness")
        rl       = _rl(raw_v)

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
            "light_chain":        None,
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
