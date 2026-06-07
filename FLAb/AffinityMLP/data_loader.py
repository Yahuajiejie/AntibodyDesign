"""
data_loader.py — FLAb binding 数据集加载与标签标准化

当前版本只为“通用亲和力排序模型”准备数据，核心原则是：

  1. 默认只纳入真实 Kd 类数据，跳过 predicted Kd、bind/no bind、
     IC50、EC50、ADCC 等语义不同的测量。
  2. 每个 CSV 文件保留为一个 compatible_group，不再为了凑测试集
     自动合并不同实验或不同测量方法。
  3. 统一标签方向：label 越大表示亲和力越强。
  4. MSE 不回归原始物理量，而使用组内 z-score label_z，避免 Kd、
     -logKd、不同单位和不同动态范围混在一起。
"""

from __future__ import annotations

import os
import re
import zipfile

import numpy as np
import pandas as pd

from .config import cfg


# ── 列名别名映射 ────────────────────────────────────────────────────────────────
# 不同数据集对同一信息使用了不同的列名，统一映射成 heavy / light。
# fitness/label 列不在这里强行重命名，因为不同文件里的测量列语义不同，
# 需要结合 metadata 和列名单独判断。
COLUMN_ALIASES = {
    "Ab_heavy_chain_seq": "heavy",   # AbRank 数据集
    "Ab_light_chain_seq": "light",   # AbRank 数据集
    "VHH_sequence": "heavy",         # COGNANO 纳米抗体数据集
}


def canonical_filename(path_or_name: str) -> str:
    """
    将本地文件名转为 metadata 中使用的标准文件名。

    用法说明：
      os.path.basename(path) 取路径最后一级文件名；
      FLAb 的本地大文件常以 .csv.zip 保存，而 metadata/README 中通常写 .csv，
      所以这里会去掉最后的 .zip，方便做映射。
    """
    name = os.path.basename(path_or_name)
    if name.endswith(".zip"):
        name = name[:-4]
    return name


def dataset_name_from_file(path_or_name: str) -> str:
    """
    从文件名得到无后缀的数据集名，用于日志、分组和模型评估表。
    """
    name = canonical_filename(path_or_name)
    return name.replace(".csv", "")


def load_metadata(metadata_path: str = cfg.METADATA_PATH) -> dict[str, dict]:
    """
    读取 FLAb metadata 表，返回 filename -> row dict。

    metadata 提供 assay/units、关键词、论文等信息。这里主要用它判断：
      - 当前文件是不是 binding 类 Kd 数据；
      - 是否为 predicted/binary/IC50/EC50 等需要跳过的数据；
      - 日志和文档中可追踪的实验语义。
    """
    if not os.path.exists(metadata_path) and not os.path.isabs(metadata_path):
        flab_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        package_relative_path = os.path.join(flab_root, metadata_path)
        if os.path.exists(package_relative_path):
            metadata_path = package_relative_path

    if not os.path.exists(metadata_path):
        print(f"[metadata] 未找到 {metadata_path}，将仅根据文件名和列名判断")
        return {}

    meta = pd.read_csv(metadata_path, low_memory=False)
    # low_memory=False 让 pandas 一次性看更多数据后再推断列类型。
    # 这样不容易把同一列在不同分块里推断成不同类型。
    rows = {}
    for _, row in meta.iterrows():
        filename = canonical_filename(str(row.get("filename", "")))
        if filename:
            rows[filename] = row.to_dict()
    print(f"[metadata] 加载 {len(rows)} 条 metadata")
    return rows


def _read_csv_or_zip(filepath: str) -> pd.DataFrame | None:
    """
    读取 .csv 或 .csv.zip 文件。

    zipfile.ZipFile 是 Python 标准库的 zip 读取器；namelist() 返回压缩包内
    文件列表，open(name) 返回一个类文件对象，可以直接交给 pandas.read_csv。
    """
    try:
        if filepath.endswith(".csv.zip"):
            with zipfile.ZipFile(filepath) as z:
                csv_names = [n for n in z.namelist() if n.endswith(".csv")]
                if not csv_names:
                    print("  [SKIP] zip 中没有 csv 文件")
                    return None
                with z.open(csv_names[0]) as f:
                    # 这段代码可能只会得到zip压缩包里的第一个csv文件，但是在这一般没问题
                    return pd.read_csv(f, low_memory=False)
        return pd.read_csv(filepath, low_memory=False)
    except Exception as exc:
        print(f"  [ERROR] 读取失败: {exc}")
        return None


def classify_assay(name: str, metadata_row: dict | None) -> tuple[str, str]:
    """
    将一个数据集归入粗粒度 assay family。

    返回：
      (assay_family, reason)

    重要逻辑：
      - kd：真实 Kd / KD / SPR Kd / BLI Kd / Octet Kd 等；
      - predicted_kd：预测标签，不作为真实监督信号；
      - binary：bind/no bind；
      - ic50/ec50/adcc：功能性或细胞实验读数，不和 Kd 混训；
      - other：其它 binding/enrichment 读数，默认不进入主亲和力模型。
    """
    row = metadata_row or {}
    # metadata_row 格式如下：
    text = " ".join([
        name,
        str(row.get("assay/units_raw", "")),
        str(row.get("assay/units", "")),
        str(row.get("key_words", "")),
    ]).lower()

    # 不用 \bkd\b，因为下划线在正则里算“单词字符”：
    # garbinski2023_kd / rawat2022abcov_kd 这类名字会被 \b 漏掉。
    has_kd = bool(re.search(r"(^|[^a-z0-9])kd([^a-z0-9]|$)", text) or "k d" in text)
    has_ic50 = "ic50" in text
    has_ec50 = "ec50" in text

    if "predicted" in text:
        return "predicted_kd", "metadata/文件名包含 predicted，跳过预测标签"
    if "bind/no bind" in text or "binary" in text:
        return "binary", "二分类结合标签，不是连续亲和力"
    if "adcc" in text:
        return "adcc", "ADCC 是细胞功能读数，不等同 Kd"
    if has_kd and (has_ic50 or has_ec50):
        return "mixed_affinity_functional", "同时包含 Kd 与 IC50/EC50，默认不自动拆分"
    if has_ic50:
        return "ic50", "IC50 是功能性抑制读数，不等同 Kd"
    if has_ec50:
        return "ec50", "EC50 是效应浓度读数，不等同 Kd"
    if has_kd:
        return "kd", "真实 Kd 类亲和力数据"

    return "other", "无法确认是 Kd 类亲和力数据"


def _numeric_fraction(series: pd.Series) -> float:
    """
    计算一列能被转成数值的比例，用于挑选真实标签列。
    """
    converted = pd.to_numeric(series, errors="coerce")
    return float(converted.notna().mean())


def choose_label_column(df: pd.DataFrame, assay_family: str) -> str | None:
    """
    为当前数据集选择亲和力标签列。

    优先级：
      1. Kd 数据集中先寻找列名包含 kd 的数值列；
      2. 显式 -log / neg log / neg_log Kd 列优先；
      3. fitness 是泛名列，优先级低于显式 log Kd；
      4. 避开 ka、kdis、counts、sequence 等非亲和力列。

    这样可以处理两类文件：
      - 已有标准 fitness 列的 FLAb 文件；
      - 只有 'Kd [M]'、'Kd avg (M)' 等原始列的文件。
    """
    if assay_family != "kd":
        if "fitness" in df.columns and _numeric_fraction(df["fitness"]) >= 0.8:
            return "fitness"
        return None

    candidates: list[tuple[int, str]] = []
    for col in df.columns:
        # 对于表格的所有表头(df.columns)，检查哪些表头包含Kd
        col_norm = col.lower().strip()
        if "kd" not in col_norm:
            continue
        if any(bad in col_norm for bad in ["kdis", "ka ", "ka(", "counts"]):
            continue
        if _numeric_fraction(df[col]) < 0.8:
            continue

        # 分数越高优先级越高：显式 -log/neg log 优先，其次 fitness，再次普通 Kd。
        score = 1
        if "-log" in col_norm or "neg log" in col_norm or "neg_log" in col_norm:
            score += 10
        if "fitness" in col_norm:
            score += 4
        if col_norm in {"kd", "kd [m]", "kd [nm]"}:
            score += 2
        candidates.append((score, col))

    if not candidates:
        if "fitness" in df.columns and _numeric_fraction(df["fitness"]) >= 0.8:
            return "fitness"
        return None
    candidates.sort(reverse=True) # 取优先级最高的表头和列，
    return candidates[0][1] # 返回的是表头名字


def _is_log_label(label_col: str, metadata_row: dict | None) -> bool:
    """
    判断标签是否已经是 -logKd / neg log Kd。

    如果已经是 log 尺度，值越大通常越强，直接使用；否则把原始 Kd
    转成 -log10(raw_value)。单位缩放不会影响组内排序，MSE 又会做组内
    z-score，因此这里的重点是方向正确。
    """
    col_text = label_col.lower().strip()
    if any(token in col_text for token in ["-log", "neg log", "neg_log"]):
        return True

    # 对 KD (nM)、Kd avg (M) 这类明确列名，优先相信实际列名；
    generic_cols = {"fitness", "label", "score"}
    if col_text not in generic_cols:
        return False

    # 如果表格的 label 无法给出判断，那就在 metadata_row 中做出判断
    row = metadata_row or {}
    meta_text = " ".join([
        str(row.get("assay/units_raw", "")),
        str(row.get("assay/units", "")),
    ]).lower()
    return any(token in meta_text for token in ["-log", "neg log", "neg_log"])


def normalize_label(
    df: pd.DataFrame,
    label_col: str,
    assay_family: str,
    metadata_row: dict | None,
) -> pd.DataFrame:
    """
    生成 label / label_z / label_rank 三种训练标签。

    label：
      方向统一后的连续值，越大表示亲和力越强。Ranking loss 使用它构造 pair。

    label_z：
      每个 compatible_group 内的 z-score。MSE 使用它，避免不同实验体系
      的绝对量纲混用。

    label_rank：
      组内百分位 rank，取值大约在 [0, 1]，方便后续做稳健性消融。
    """
    out = df.copy()
    raw = pd.to_numeric(out[label_col], errors="coerce")
    out["label_raw"] = raw

    if assay_family == "kd" and not _is_log_label(label_col, metadata_row):
        positive = raw > 0
        if not positive.all():
            out = out.loc[positive].copy()
            raw = out["label_raw"]
        out["label"] = -np.log10(raw.astype(float))
        out["label_transform"] = "neg_log10_raw_kd"
    else:
        out["label"] = raw.astype(float)
        out["label_transform"] = "as_is_higher_is_better"

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["label"]).reset_index(drop=True)

    mu = out["label"].mean()
    sigma = out["label"].std(ddof=0)
    if sigma > 0:
        out["label_z"] = (out["label"] - mu) / sigma
    else:
        out["label_z"] = 0.0

    # pct=True 返回百分位秩；method='average' 让 ties 共享平均名次。
    out["label_rank"] = out["label"].rank(method="average", pct=True)
    return out


def _standardize_sequences(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    统一 heavy/light 列并拼接成模型输入 sequence。
    """
    rename_map = {k: v for k, v in COLUMN_ALIASES.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    if "heavy" not in df.columns:
        print("  [SKIP] 缺少 heavy 重链序列列")
        return None

    df = df.dropna(subset=["heavy"]).reset_index(drop=True)
    missing_tokens = {"", "nan", "none", "null", "na"}
    heavy_seq = df["heavy"].astype(str).str.strip()
    valid_heavy = ~heavy_seq.str.lower().isin(missing_tokens)
    if not valid_heavy.all():
        df = df.loc[valid_heavy].reset_index(drop=True)
        heavy_seq = df["heavy"].astype(str).str.strip()

    if "light" in df.columns:
        light_seq = df["light"].astype(str).str.strip()
        has_light = df["light"].notna() & ~light_seq.str.lower().isin(missing_tokens)
        df["sequence"] = heavy_seq
        df.loc[has_light, "sequence"] = (
            heavy_seq.loc[has_light] + cfg.LINKER + light_seq.loc[has_light]
        )
    else:
        df["sequence"] = heavy_seq

    return df


def load_one_dataset(
    filepath: str,
    metadata: dict[str, dict] | None = None,
) -> pd.DataFrame | None:
    """
    加载单个 binding 数据集，返回标准化 DataFrame。

    返回 None 表示跳过。跳过原因会打印到日志，方便后续质检。
    """
    metadata = metadata or {}
    filename = canonical_filename(filepath)
    name = dataset_name_from_file(filepath)
    meta_row = metadata.get(filename)

    assay_family, reason = classify_assay(name, meta_row)
    if assay_family not in cfg.ALLOWED_ASSAY_FAMILIES:
        print(f"  [SKIP] assay_family={assay_family}: {reason}")
        return None

    df = _read_csv_or_zip(filepath)
    if df is None:
        return None

    df = _standardize_sequences(df)
    if df is None:
        return None

    label_col = choose_label_column(df, assay_family)
    if label_col is None:
        print("  [SKIP] 找不到可靠的亲和力数值列")
        return None

    df = normalize_label(df, label_col, assay_family, meta_row)

    if len(df) > cfg.MAX_DATASET_SIZE:
        print(
            f"  [SKIP] 数据集过大（{len(df):,} 条 > {cfg.MAX_DATASET_SIZE:,}），"
            "可在 config.py 调高上限后重新运行"
        )
        return None
    if len(df) < cfg.MIN_GROUP_SIZE:
        print(f"  [SKIP] 数据量不足（{len(df)} 条 < {cfg.MIN_GROUP_SIZE}）")
        return None
    if df["label"].nunique() < 2:
        print("  [SKIP] label 全部相同，无法学习排序")
        return None

    row = meta_row or {}
    df["dataset"] = name
    df["source_file"] = filename
    df["assay_family"] = assay_family
    df["assay_units_raw"] = row.get("assay/units_raw", "")
    df["assay_units"] = row.get("assay/units", "")
    df["key_words"] = row.get("key_words", "")
    df["publication_title"] = row.get("publication_title", "")

    # 关键：每个 CSV 作为一个可比较组。这样 ranking pair 不会跨 assay/抗原构造。
    df["compatible_group"] = name

    print(
        f"  → {len(df)} 条，assay={assay_family}, label_col='{label_col}', "
        f"transform={df['label_transform'].iloc[0]}"
    )
    return df


def load_all_datasets(
    data_dir: str = cfg.DATA_DIR,
    metadata_path: str = cfg.METADATA_PATH,
) -> dict[str, pd.DataFrame]:
    """
    扫描 data_dir 下全部 CSV，加载可用于通用亲和力模型的数据集。

    与旧版本不同：
      - 不做自动合并；
      - 不按每个 benchmark 单独训练；
      - 返回的每个 DataFrame 都带有 compatible_group、label、label_z。
    """
    metadata = load_metadata(metadata_path)

    all_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith(".csv") or f.endswith(".csv.zip")
    ])
    print(f"\n[数据] 扫描 {len(all_files)} 个 binding 文件...")

    datasets: dict[str, pd.DataFrame] = {}
    skipped = 0

    for fname in all_files:
        name = dataset_name_from_file(fname)
        fpath = os.path.join(data_dir, fname)
        print(f"\n  [{name}]")
        df = load_one_dataset(fpath, metadata=metadata)
        if df is None:
            skipped += 1
            continue
        datasets[name] = df

    total_rows = sum(len(df) for df in datasets.values())
    print(
        f"\n[数据] 加载完成：{len(datasets)} 个可比较组，"
        f"{total_rows:,} 条序列；跳过 {skipped} 个文件"
    )
    return datasets
