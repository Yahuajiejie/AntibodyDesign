"""

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

_VALID_AA = frozenset("ACDEFGHIKLMNPQRSTVWYX")  # X = any AA (IUPAC), handled by ESMC    BJOUZ不能用

def _seq(val):
    if val is None: return None
    if isinstance(val, float) and not math.isfinite(val): return None
    s = str(val).strip().upper()
    if not s or s == "": return None
    return s if all(c in _VALID_AA for c in s) else None

def _sanitize(name):
    return re.sub(r"[^A-Za-z0-9_]", "_", str(name).strip())

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
    # source row 所在行数
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
    ag_seq  = _seq(row.get("Ag_seq"))
    ag_src  = "provided" if ag_seq else "missing"
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
    from rich import print as rprint
    # ------------------------ 测试seq()，实现思路： ------------------------ 
    # 1.空值与初步清洗：输入不能是none，以及要全大写
    # 2.不能包含非蛋白质氨基酸的字母
    s1 = " ACDEFGHIKL "
    s2 = "ABABABAB"
    s3 = ""
    s4 = "nan"
    s5 = "NAN" # Asn - Ala -Asn
    print(_seq(s1),_seq(s2),_seq(s3),_seq(s4),_seq(s5))

    #  ------------------------ 测试_rl_kd():v=−log 10(raw×10^−9) ------------------------ 
    i1 = "100"
    i2 = "1000000"
    i3 = "   100000  "
    i4 = " 1000 00 0"
    i5 = 50

    print(_rl_kd(i1),_rl_kd(i2),_rl_kd(i3),_rl_kd(i4),_rl_kd(i5))

    #  ------------------------ 测试_make_base ------------------------ 
    # 这个函数的功能，只是原封不动的依照input和全局变量构造结构体
    # 因此，构造案例时，只需要观察Ag_seq这一列的数据分布即可，因为他是函数的一个输入
    a1 = {
        "Ab_name": "Sab-4i18_HL",
        "Ab_heavy_chain_seq": "EVQLVESGGGLVQPGGSL",
        "Ab_light_chain_seq": "DIQMTQSPSSLSASVGDR",
        "Ag_name": "Spike_protein",
        "Ag_seq": "MNGT",
        "Affinity_Kd [nM]": "1.0",
        "IC50 [ug/mL]": "0.05"
    }
    a1 = pd.Series(a1)
    a2 = {    
        "Ab_name":"AbCoV-C1170",
        "Ag_name":"SARS-CoV-2",
        "Ag_name_details":"NA",
        "IC50 [ug/mL]":"2.38E-02",
        "Affinity_Kd [nM]":"NA",
        "Ag_epitope_restrictions":"RBD",
        "Ab_heavy_chain_seq":"QVQLVQSGAEVKKPGASVKVSCKASGYTFSSYFIHWVRQAPGQGLEWMGIINPGGASRSSAQKFQGRVTMTSDTSTSTVYMELSSLRSEDTAVYYCAREHGGNSYFDQWGQGTLVTVSS+I253",
        "Ab_light_chain_seq":"DIQLTQSPSFLSASVGDRVTITCRASQGISGYLAWYQQKPGEAPKVLIYAASTLQSGVPSRFSGSGSGTEFTLTISSLQPEDFATYYCQHLNNYPVAFGQGTKVEIK",
        "Ag_seq":"MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYF",
        "Ab_structure_method":"NA",
        "Ag_structure_method":"Crystalized",
        "bound_AbAg_structure_method":"NA",
        "Ab_PDB_ID":"NA",
        "Ag_PDB_ID":"NA",
        "bound_AbAg_PDB_ID":"NA",
        "Ab_Lev3_cluster":"3048" ,#p
        "Source":"AbCov",
    }
    a3 = {    
        "Ab_name":"AbCoV-C1170",
        "Ag_name":"SARS-CoV-2",
        "Ag_name_details":"NA",
        "IC50 [ug/mL]":"2.38E-02",
        "Affinity_Kd [nM]":"NA",
        "Ag_epitope_restrictions":"RBD",
        "Ab_heavy_chain_seq":"QVQLVQSGAEVKKPGASVKVSCKASGYTFSSYFIHWVRQAPGQGLEWMGIINPGGASRSSAQKFQGRVTMTSDTSTSTVYMELSSLRSEDTAVYYCAREHGGNSYFDQWGQGTLVTVSS",
        "Ab_light_chain_seq":"DIQLTQSPSFLSASVGDRVTITCRASQGISGYLAWYQQKPGEAPKVLIYAASTLQSGVPSRFSGSGSGTEFTLTISSLQPEDFATYYCQHLNNYPVAFGQGTKVEIK",
        "Ag_seq":"",
        "Ab_structure_method":"NA",
        "Ag_structure_method":"Crystalized",
        "bound_AbAg_structure_method":"NA",
        "Ab_PDB_ID":"NA",
        "Ag_PDB_ID":"NA",
        "bound_AbAg_PDB_ID":"NA",
        "Ab_Lev3_cluster":"3048" ,#p
        "Source":"AbCov",
    }
    a4 = {    
        "Ab_name":"AbCoV-C1170",
        "Ag_name":"SARS-CoV-2",
        "Ag_name_details":"NA",
        "IC50 [ug/mL]":"2.38E-02",
        "Affinity_Kd [nM]":"NA",
        "Ag_epitope_restrictions":"RBD",
        "Ab_heavy_chain_seq":"QVQLVQSGAEVKKPGASVKVSCKASGYTFSSYFIHWVRQAPGQGLEWMGIINPGGASRSSAQKFQGRVTMTSDTSTSTVYMELSSLRSEDTAVYYCAREHGGNSYFDQWGQGTLVTVSS",
        "Ab_light_chain_seq":"DIQLTQSPSFLSASVGDRVTITCRASQGISGYLAWYQQKPGEAPKVLIYAASTLQSGVPSRFSGSGSGTEFTLTISSLQPEDFATYYCQHLNNYPVAFGQGTKVEIK",
        "Ag_seq":"SARS-CoV2_(V502K)",
        "Ab_structure_method":"NA",
        "Ag_structure_method":"Crystalized",
        "bound_AbAg_structure_method":"NA",
        "Ab_PDB_ID":"NA",
        "Ag_PDB_ID":"NA",
        "bound_AbAg_PDB_ID":"NA",
        "Ab_Lev3_cluster":"3048" ,#p
        "Source":"AbCov",
    }
    a2 = pd.Series(a2)
    print(_make_base(1,a1,_sanitize(str(a1.get("Ag_name","")).strip()),str(a1.get("Ag_name","")).strip(),str(a1.get("Ag_seq","")).strip(),"provided"))
    print(_make_base(2232,a2,_sanitize(str(a2.get("Ag_name","")).strip()),str(a2.get("Ag_name","")).strip(),str(a2.get("Ag_seq","")).strip(),"provided"))
    print(_make_base(2232,a3,_sanitize(str(a3.get("Ag_name","")).strip()),str(a3.get("Ag_name","")).strip(),str(a3.get("Ag_seq","")).strip(),"missing"))
    print(_make_base(182232,a4,_sanitize(str(a4.get("Ag_name","")).strip()),str(a4.get("Ag_name","")).strip(),str(a4.get("Ag_seq","")).strip(),"missing"))
    # 正常来说，a4 的 Ag-seq直接输出Sars-Cov2 才是对的

    #  ------------------------ 测试_make_base ------------------------ 
    # 这个函数的功能，需要检查边缘案例
    # 因此，构造案例时，检查它对奇怪输出的兼容性，并且尽可能构造多的奇怪案例

    a5 = {
        "Ab_name": "Test-Censored-Kd",
        "Ab_heavy_chain_seq": "EVQLVES", "Ab_light_chain_seq": "DIQMTQS",
        "Ag_name": "Target", "Ag_seq": "MNGT",
        "Affinity_Kd [nM]": "<0.1",       # 带有小于号的截断数据
        "IC50 [ug/mL]": ">100.0"          # 带有大于号的截断数据
    }
    # 预期结果：
    # - 成功生成记录
    # - keep_for_training = False
    # - drop_reason 包含 "censored_measurement"
    # - metric_value_numeric 成功解析为 0.1 和 100.0
    # - rank_label = None

    a6 = {
        "Ab_name": "Test-Zero-Kd",
        "Ab_heavy_chain_seq": "EVQLVES", "Ab_light_chain_seq": "DIQMTQS",
        "Ag_name": "Target", "Ag_seq": "MNGT",
        "Affinity_Kd [nM]": "0",          # 0 浓度在 log10 计算时会出问题
        "IC50 [ug/mL]": "-5.5"            # 负数在 log10 计算时会出问题
    }
    # 预期结果：
    # - 内部 log 计算会进入 except 并令 kd_rl/ic_rl = None
    # - keep_for_training = False
    # - drop_reason 包含 "missing_rank_label"
    a7 = {
    "Ab_name": "Test-Mixed-Nan",
    "Ab_heavy_chain_seq": "EVQLVES", "Ab_light_chain_seq": "DIQMTQS",
    "Ag_name": "Target", "Ag_seq": "MNGT",
    "Affinity_Kd [nM]": "  nAn ",      # 带有空格和大小写混合的 NaN
    "IC50 [ug/mL]": "None"            # 字符串形式的 None
    }
    # 预期结果：
    # - 该行代码直接跳过两个 if 判断，返回一个空列表 []
    a8 = {
    "Ab_name": "Test-Missing-Heavy",
    "Ab_heavy_chain_seq": "",        # 缺失重链
    "Ab_light_chain_seq": "DIQMTQS",   # 有轻链（类似纳米抗体的反向情况，或单纯的数据缺失）
    "Ag_name": "Target", "Ag_seq": "MNGT",
    "Affinity_Kd [nM]": "1.5",
    "IC50 [ug/mL]": "NA"
    }
    # 预期结果：
    # - 成功生成 Kd 记录
    # - heavy 被 _seq 解析为 None
    # - keep_for_training = False
    # - drop_reason 包含 "missing_or_invalid_heavy_chain"
    a9 = {
    "Ab_name": "Test-No-Ag",
    "Ab_heavy_chain_seq": "EVQLVES", "Ab_light_chain_seq": "DIQMTQS",
    "Ag_name": "",                     # 留空
    "Ag_seq": "MNGT",
    "Affinity_Kd [nM]": "1.5",
    "IC50 [ug/mL]": "NA"
    }
    # 预期结果：
    # - ag_key 应该被优雅地降级赋值为 "unknown"
    # - group_id 会变成 "AbRank/dataset/unknown/neg_log10_kd_M/experimental"
    # - 程序顺利运行不报错

    rprint("a1---\n",_build_records(1,a1))
    rprint("a2---\n",_build_records(2,a2))
    rprint("a3---\n",_build_records(3,a3))
    rprint("a4---\n",_build_records(4,a4))
    rprint("a5---\n",_build_records(5,a5))
    rprint("a6---\n",_build_records(6,a6))
    rprint("a7---\n",_build_records(7,a7))
    rprint("a8---\n",_build_records(8,a8))
    rprint("a9---\n",_build_records(9,a9))

    parquet_path = "/Users/yahuagege/Desktop/antibody/AffinityTransformer/processed/binding/AbRank/dataset/records.parquet"
    df = pd.read_parquet(parquet_path)

    # 2. 检查维度（行数、列数）
    print(f"=== 数据形状 ===")
    print(f"行数: {df.shape[0]}, 列数: {df.shape[1]}\n")

    # 3. 检查 Schema（列名、数据类型、缺失值）
    print("=== 数据结构与类型 ===")
    print(df.info()) 
    print("\n")

    # 4. 可视化查看前 5 行（在 Jupyter 中会自动渲染成漂亮的表格）
    print("=== 数据样例 (前5行) ===")
    # 如果列很多，可以设置展示全部列
    pd.set_option('display.max_columns', None)
    print(df.head())

    # 5. 快速统计摘要（检查数值范围、唯一值数量等）
    print("\n=== 数据统计摘要 ===")
    print(df.describe(include='all'))