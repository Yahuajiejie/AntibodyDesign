#!/usr/bin/env python3
"""Convert hie2023efficient_CoV2_S309_Kd.csv -> standard training table.

Study:      hie2023efficient
Table:      CoV2_S309_Kd
Antigen:    SARS-CoV-2 WT Spike S6P
rank_label: fitness = neg_log_Kd; S309 lineage vs CoV2 WT S6P
"""
# ── CONFIG ────────────────────────────────────────────────────────────────────
STUDY_ID       = "hie2023efficient"
TABLE_ID       = "CoV2_S309_Kd"
SOURCE_FILE    = "data/binding/hie2023efficient_CoV2_S309_Kd.csv"
ANTIBODY_TYPE  = "IgG"   # Fv | Fab | IgG | scFv | VHH | unknown | maybe wrong because the table doesn't provide anything
ANTIGEN_KEY    = "CoV2_WT_RBD"
ANTIGEN_NAME   = "SARS-CoV-2 WT spike receptor-binding domain"
ANTIGEN_SEQ    = "RVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF"
ANTIGEN_SOURCE = "retrieved"
ANTIGEN_SOURCE_NOTE = "retrieved: UniProt P0DTC2 aa319-541 (SARS-CoV-2 Wuhan-Hu-1 Spike RBD); local competition README lists WT-S6P assay antigen, S309 is an RBD antibody"
ASSAY_NAME     = "SPR"
METRIC_NAME    = "neg_log10_kd_M"
METRIC_UNIT    = "-log10(KD/M)"
TRANSFORM_RULE = "fitness = neg_log_Kd; S309 lineage vs CoV2 WT S6P"
LABEL_KIND     = "experimental"
FITNESS_COL    = "fitness"
# ─────────────────────────────────────────────────────────────────────────────
import argparse, math
from pathlib import Path
import pandas as pd

DATASET_ID = f"{STUDY_ID}/{TABLE_ID}"
GROUP_ID   = f"{STUDY_ID}/{TABLE_ID}/{ANTIGEN_KEY}/{METRIC_NAME}/{LABEL_KIND}"
_VALID_AA  = frozenset("ACDEFGHIKLMNPQRSTVWYX")  # X = any AA (IUPAC), handled by ESMC

def _rl(raw) -> float:
    return float(raw)


def _seq(val) -> "str | None":
    """Return uppercased sequence if all AA are standard, else None.

    Non-standard characters (e.g. X) cause the sequence to be nulled.
    Records are NOT dropped solely because the light chain is nulled.
    Only missing/invalid heavy chain or missing rank_label cause drops.
    """
    if val is None:
        return None
    s = str(val).strip().upper()
    if not s or s == "":
        return None
    return s if all(c in _VALID_AA for c in s) else None


def _flt(v):
    try:
        return float(v)
    except Exception:
        return None


def convert(src: Path, out: Path) -> None:
    df = pd.read_csv(src, dtype=str)
    recs = []
    for i, row in df.iterrows():
        sr = int(i) + 2   # 1-indexed row; +1 for header

        heavy = _seq(row.get("heavy"))
        light = _seq(row.get("light"))
        raw = row.get(FITNESS_COL)

        try:
            rl = _rl(raw)
            rl = rl if math.isfinite(rl) else None
        except Exception:
            rl = None

        drops = []
        if heavy is None:
            drops.append("missing_or_invalid_heavy_chain")
        if rl is None:
            drops.append("missing_rank_label")

        recs.append(dict(
            record_id             = f"{DATASET_ID}/{sr}",
            dataset_id            = DATASET_ID,
            study_id              = STUDY_ID,
            table_id              = TABLE_ID,
            source_file           = SOURCE_FILE,
            source_row            = sr,
            antibody_id           = None,
            antibody_type         = ANTIBODY_TYPE,
            heavy_chain           = heavy,
            light_chain           = light,
            single_chain_sequence = None,
            antigen_key           = ANTIGEN_KEY,
            antigen_name          = ANTIGEN_NAME,
            antigen_sequence      = ANTIGEN_SEQ,
            antigen_source        = ANTIGEN_SOURCE,
            assay_name            = ASSAY_NAME,
            assay_type            = "binding",
            metric_name           = METRIC_NAME,
            metric_value_raw      = str(raw),
            metric_value_numeric  = _flt(raw),
            metric_unit           = METRIC_UNIT,
            metric_direction      = "higher_is_better",
            transform_rule        = TRANSFORM_RULE,
            rank_label            = rl,
            label_kind            = LABEL_KIND,
            group_id              = GROUP_ID,
            keep_for_training     = not drops,
            drop_reason           = ("; ".join(drops) or None),
        ))

    out.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(recs)
    result.to_parquet(out / "records.parquet", index=False)
    result.to_csv(out / "records.csv", index=False)
    n, k = len(result), int(result["keep_for_training"].sum())
    print(f"[{STUDY_ID}/{TABLE_ID}]  total={n}  keep={k}  drop={n - k}")
    print(f"  -> {out / 'records.parquet'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Convert binding CSV to standard training table."
    )
    ap.add_argument("input",      type=Path, help="Source CSV path")
    ap.add_argument("output_dir", type=Path, help="Output directory")
    a = ap.parse_args()
    convert(a.input, a.output_dir)
