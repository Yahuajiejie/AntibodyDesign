#!/usr/bin/env python3
"""Convert AbRank dataset to standard training table.

Usage (from repo root):
    python3 scripts/prepare/binding/AbRank/dataset/convert.py <src> <outdir>

AbRank contains antibodies vs. many antigens. Ag_seq is provided in the CSV.
Two metrics per row (where not NA): Affinity_Kd [nM] and IC50 [ug/mL].
Each metric yields a separate record; group_id encodes metric_name + Ag_name.

~283 MB uncompressed. Uses chunked streaming + incremental parquet writes.
"""
import math, sys, datetime, zipfile, re
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_VALID_AA = frozenset("ACDEFGHIKLMNPQRSTVWYX")  # X = any AA (IUPAC), handled by ESMC
_STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
_SARS_COV_2_RBD = (
    "RVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTK"
    "LNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLY"
    "RLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHA"
    "PATVCGPKKSTNLVKNKCVNF"
)
_SARS_COV_2_RBD_UNIPROT_START = 319
_SARS_COV_2_ABRANK_OFFSET = 1
_SARS_COV_2_MUTANT_RE = re.compile(r"^SARS_CoV_2_([A-Z])(\d+)([A-Z])$")

def _seq(val):
    if val is None: return None
    if isinstance(val, float) and not math.isfinite(val): return None
    s = str(val).strip().upper()
    if not s or s == "": return None
    return s if all(c in _VALID_AA for c in s) else None

def _sanitize(name):
    return re.sub(r"[^A-Za-z0-9_]", "_", str(name).strip())

def _derive_sars_cov_2_rbd_mutant(ag_key):
    """Derive AbRank RBD single-mutant antigens from Wuhan-Hu-1 RBD.

    AbRank's RBD-escape entries use numbering that is one residue lower than
    UniProt P0DTC2. For example, AbRank E483 maps to UniProt E484.
    """
    m = _SARS_COV_2_MUTANT_RE.match(ag_key)
    if not m:
        return None

    ref, abrank_pos, alt = m.group(1), int(m.group(2)), m.group(3)
    if ref not in _STANDARD_AA or alt not in _STANDARD_AA:
        return None

    uniprot_pos = abrank_pos + _SARS_COV_2_ABRANK_OFFSET
    idx = uniprot_pos - _SARS_COV_2_RBD_UNIPROT_START
    if idx < 0 or idx >= len(_SARS_COV_2_RBD):
        return None
    if _SARS_COV_2_RBD[idx] != ref:
        return None

    seq = list(_SARS_COV_2_RBD)
    seq[idx] = alt
    return "".join(seq)

def _antigen_sequence_and_source(row, ag_key):
    ag_seq = _seq(row.get("Ag_seq"))
    if ag_seq:
        return ag_seq, "provided"

    derived = _derive_sars_cov_2_rbd_mutant(ag_key)
    if derived:
        return derived, "retrieved"

    return None, "missing"

STUDY_ID="AbRank"; TABLE_ID="dataset"
SOURCE_FILE="data/binding/AbRank_dataset.csv.zip"
ANTIBODY_TYPE="Fv"; ASSAY_TYPE="binding"
CHUNKSIZE=10_000   # smaller chunks because rows are wide

def _rl_kd(raw):
    try:
        v = -math.log10(float(raw) * 1e-9)
        return v if math.isfinite(v) else None
    except: return None

def _rl_ic50(raw):
    try:
        v = -math.log10(float(raw))
        return v if math.isfinite(v) else None
    except: return None

def _make_base(source_row, row, ag_key, ag_name, ag_seq, ag_src):
    return dict(
        dataset_id=f"{STUDY_ID}/{TABLE_ID}",
        study_id=STUDY_ID, table_id=TABLE_ID,
        source_file=SOURCE_FILE, source_row=source_row,
        antibody_id=str(row.get("Ab_name","")).strip() or None,
        antibody_type=ANTIBODY_TYPE,
        heavy_chain=_seq(row.get("Ab_heavy_chain_seq")),
        light_chain=_seq(row.get("Ab_light_chain_seq")),
        single_chain_sequence=None,
        antigen_key=ag_key, antigen_name=ag_name,
        antigen_sequence=ag_seq, antigen_source=ag_src,
        assay_type=ASSAY_TYPE,
    )

def _build_records(source_row, row):
    ag_raw  = str(row.get("Ag_name","")).strip()
    ag_key  = _sanitize(ag_raw) if ag_raw else "unknown"
    ag_seq, ag_src = _antigen_sequence_and_source(row, ag_key)
    base    = _make_base(source_row, row, ag_key, ag_raw, ag_seq, ag_src)
    heavy   = base["heavy_chain"]
    records = []

    def _is_censored(s): return s.startswith("<") or s.startswith(">")
    def _safe_float(s):
        try: return float(s)
        except: return None

    # Kd record
    kd_raw = row.get("Affinity_Kd [nM]")
    kd_str = str(kd_raw).strip()
    if kd_str not in ("nan","NA","None",""):
        censored = _is_censored(kd_str)
        kd_rl  = None if censored else _rl_kd(kd_raw)
        drops  = []
        if heavy is None: drops.append("missing_or_invalid_heavy_chain")
        if censored: drops.append("censored_measurement")
        elif kd_rl is None: drops.append("missing_rank_label")
        records.append({**base,
            "record_id": f"AbRank/dataset/kd/{source_row}",
            "assay_name": "SPR Kd",
            "metric_name": "neg_log10_kd_M",
            "metric_value_raw": kd_str,
            "metric_value_numeric": _safe_float(kd_str.lstrip("<>")),
            "metric_unit": "-log10(KD/M)",
            "metric_direction": "higher_is_better",
            "transform_rule": "rank_label = -log10(Kd_nM * 1e-9)",
            "rank_label": kd_rl, "label_kind": "experimental",
            "group_id": f"AbRank/dataset/{ag_key}/neg_log10_kd_M/experimental",
            "keep_for_training": len(drops)==0,
            "drop_reason": "; ".join(drops) if drops else None,
        })

    # IC50 record
    ic_raw = row.get("IC50 [ug/mL]")
    ic_str = str(ic_raw).strip()
    if ic_str not in ("nan","NA","None",""):
        censored = _is_censored(ic_str)
        ic_rl  = None if censored else _rl_ic50(ic_raw)
        drops  = []
        if heavy is None: drops.append("missing_or_invalid_heavy_chain")
        if censored: drops.append("censored_measurement")
        elif ic_rl is None: drops.append("missing_rank_label")
        records.append({**base,
            "record_id": f"AbRank/dataset/ic50/{source_row}",
            "assay_name": "IC50",
            "metric_name": "neg_log10_ic50_ugml",
            "metric_value_raw": ic_str,
            "metric_value_numeric": _safe_float(ic_str.lstrip("<>")),
            "metric_unit": "-log10(IC50 ug/mL)",
            "metric_direction": "higher_is_better",
            "transform_rule": "rank_label = -log10(IC50_ugml); within-group rank preserved",
            "rank_label": ic_rl, "label_kind": "experimental",
            "group_id": f"AbRank/dataset/{ag_key}/neg_log10_ic50_ugml/experimental",
            "keep_for_training": len(drops)==0,
            "drop_reason": "; ".join(drops) if drops else None,
        })

    return records

def convert(src: Path, out: Path):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Streaming {src} ...")
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src) as zf:
        fn=[n for n in zf.namelist()
            if n.endswith(".csv") and not n.startswith("__")][0]
        writer=None; total=0; kept=0; dropped=[]
        with zf.open(fn) as fh:
            for chunk in pd.read_csv(fh, chunksize=CHUNKSIZE, low_memory=False):
                batch=[]
                for _,row in chunk.iterrows():
                    total+=1; source_row=total+1
                    for r in _build_records(source_row, row):
                        batch.append(r)
                        if r["keep_for_training"]: kept+=1
                        else: dropped.append({k:r[k] for k in
                            ("record_id","source_file","source_row",
                             "heavy_chain","metric_value_raw","drop_reason")})
                if batch:
                    df_batch=pd.DataFrame(batch)
                    # Ensure nullable string cols aren't inferred as null-type
                    for c in ["drop_reason","heavy_chain","light_chain",
                              "single_chain_sequence","antigen_sequence","antibody_id"]:
                        if c in df_batch.columns:
                            df_batch[c]=df_batch[c].astype(object)
                    tbl=pa.Table.from_pandas(df_batch,preserve_index=False)
                    # Cast any remaining null-typed columns to string
                    new_fields=[]; new_cols=[]
                    for i,field in enumerate(tbl.schema):
                        col=tbl.column(i)
                        if pa.types.is_null(field.type):
                            col=col.cast(pa.string())
                            field=pa.field(field.name,pa.string())
                        new_fields.append(field); new_cols.append(col)
                    tbl=pa.table({f.name:c for f,c in zip(new_fields,new_cols)},
                                 schema=pa.schema(new_fields))
                    if writer is None: writer=pq.ParquetWriter(out/"records.parquet",tbl.schema)
                    writer.write_table(tbl)
        if writer: writer.close()
    n_records=kept+(len(dropped))
    drop_count=len(dropped)
    pd.DataFrame([dict(study_id=STUDY_ID,table_id=TABLE_ID,source_file=SOURCE_FILE,
        total=n_records,keep=kept,drop=drop_count,n_groups="see group_id",
        generated_at=datetime.datetime.now().isoformat())
    ]).to_csv(out/"qc_summary.csv",index=False)
    pd.DataFrame(dropped).to_csv(out/"dropped_records.csv",index=False)
    # records.csv omitted for large files
    print(f"[{STUDY_ID}/{TABLE_ID}]  source_rows={total}  records={n_records}  keep={kept}  drop={drop_count}")
    print(f"  -> {out}/records.parquet")

if __name__=="__main__":
    if len(sys.argv)<3: print(f"Usage: {Path(sys.argv[0]).name} <src> <outdir>",file=sys.stderr); sys.exit(1)
    convert(Path(sys.argv[1]),Path(sys.argv[2]))
