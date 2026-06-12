#!/usr/bin/env python3
"""Convert raw data to standard training table — li2023machine/affinity2.

~700 MB uncompressed. Uses chunked streaming + incremental parquet writes.
Usage (from repo root):
    python3 scripts/prepare/binding/li2023machine/affinity2/convert.py <src> <outdir>
"""
import math, sys, datetime, zipfile
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_VALID_AA = frozenset("ACDEFGHIKLMNPQRSTVWYX")  # X = any AA (IUPAC), handled by ESMC

def _seq(val):
    if val is None: return None
    if isinstance(val, float) and not math.isfinite(val): return None
    s = str(val).strip().upper()
    if not s: return None
    return s if all(c in _VALID_AA for c in s) else None

STUDY_ID="li2023machine"; TABLE_ID="affinity2"
SOURCE_FILE="data/binding/li2023machine_scFv-SARS-CoV-2_affinity2.csv.zip"
ANTIBODY_TYPE="scFv"; ANTIGEN_KEY="CoV2_RBD"
ANTIGEN_NAME="SARS-CoV-2 RBD"; ANTIGEN_SEQ=None; ANTIGEN_SOURCE="missing"
ASSAY_NAME="ML predicted affinity"; ASSAY_TYPE="binding"
METRIC_NAME="pred_affinity"; METRIC_UNIT="predicted score (dimensionless)"
METRIC_DIRECTION="higher_is_better"; LABEL_KIND="predicted"
TRANSFORM_RULE="rank_label = Pred_affinity"
GROUP_ID="li2023machine/affinity2/CoV2_RBD/pred_affinity/predicted"
CHUNKSIZE=50_000; SKIP_ROWS=6

def _rl(raw):
    try:
        v=float(raw); return v if math.isfinite(v) else None
    except: return None

def _rec(n, row):
    heavy=_seq(row.get("HC")); light=_seq(row.get("LC"))
    raw_v=row.get("Pred_affinity"); rl=_rl(raw_v)
    drops=[]
    if heavy is None: drops.append("missing_or_invalid_heavy_chain")
    if rl is None: drops.append("missing_rank_label")
    return dict(record_id=f"{STUDY_ID}/{TABLE_ID}/{n}",
        dataset_id=f"{STUDY_ID}/{TABLE_ID}", study_id=STUDY_ID, table_id=TABLE_ID,
        source_file=SOURCE_FILE, source_row=n,
        antibody_id=str(row.get("POI","")) or None, antibody_type=ANTIBODY_TYPE,
        heavy_chain=heavy, light_chain=light, single_chain_sequence=None,
        antigen_key=ANTIGEN_KEY, antigen_name=ANTIGEN_NAME,
        antigen_sequence=ANTIGEN_SEQ, antigen_source=ANTIGEN_SOURCE,
        assay_name=ASSAY_NAME, assay_type=ASSAY_TYPE,
        metric_name=METRIC_NAME, metric_value_raw=str(raw_v),
        metric_value_numeric=(float(raw_v) if raw_v is not None else None),
        metric_unit=METRIC_UNIT, metric_direction=METRIC_DIRECTION,
        transform_rule=TRANSFORM_RULE, rank_label=rl,
        label_kind=LABEL_KIND, group_id=GROUP_ID,
        keep_for_training=len(drops)==0,
        drop_reason="; ".join(drops) if drops else None)

def convert(src: Path, out: Path):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Streaming {src} (skip 6 header rows) ...")
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src) as zf:
        fn=[n for n in zf.namelist() if n.endswith(".csv") and not n.startswith("__")][0]
        writer=None; total=0; kept=0; dropped=[]
        with zf.open(fn) as fh:
            skip=list(range(0,SKIP_ROWS)) if SKIP_ROWS else None
            for chunk in pd.read_csv(fh,chunksize=CHUNKSIZE,skiprows=skip,low_memory=False):
                batch=[]
                for _,row in chunk.iterrows():
                    total+=1; n=total+1+SKIP_ROWS; r=_rec(n,row)
                    batch.append(r)
                    if r["keep_for_training"]: kept+=1
                    else: dropped.append({k:r[k] for k in
                        ("record_id","source_file","source_row","heavy_chain","metric_value_raw","drop_reason")})
                tbl=pa.Table.from_pandas(pd.DataFrame(batch),preserve_index=False)
                if writer is None: writer=pq.ParquetWriter(out/"records.parquet",tbl.schema)
                writer.write_table(tbl)
        if writer: writer.close()
    drop=total-kept
    pd.DataFrame([dict(study_id=STUDY_ID,table_id=TABLE_ID,source_file=SOURCE_FILE,
        total=total,keep=kept,drop=drop,n_groups=1,
        generated_at=datetime.datetime.now().isoformat())
    ]).to_csv(out/"qc_summary.csv",index=False)
    pd.DataFrame(dropped).to_csv(out/"dropped_records.csv",index=False)
    # records.csv omitted for large files
    print(f"[{STUDY_ID}/{TABLE_ID}]  total={total}  keep={kept}  drop={drop}")
    print(f"  -> {out}/records.parquet")

if __name__=="__main__":
    if len(sys.argv)<3: print(f"Usage: {Path(sys.argv[0]).name} <src> <outdir>",file=sys.stderr); sys.exit(1)
    convert(Path(sys.argv[1]),Path(sys.argv[2]))
