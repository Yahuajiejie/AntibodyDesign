"""
data_loader.py — FLAb binding 数据集加载

负责：
  - 读取 .csv 和 .csv.zip 文件
  - 统一各数据集的列名差异（不同论文用不同命名）
  - 拼接 heavy + linker + light 作为模型输入序列
  - 过滤规模不合适的数据集
"""

import os
import zipfile
import pandas as pd

from .config import cfg


# ── 列名别名映射 ────────────────────────────────────────────────────────────────
# 不同数据集对同一信息使用了不同的列名，统一映射成 heavy / light / fitness
# key = 原始列名，value = 标准列名
COLUMN_ALIASES = {
    "Ab_heavy_chain_seq": "heavy",   # AbRank 数据集
    "Ab_light_chain_seq": "light",   # AbRank 数据集
    "VHH_sequence":       "heavy",   # COGNANO 纳米抗体数据集（VHH 只有重链）
}


def load_one_dataset(filepath: str) -> pd.DataFrame | None:
    """
    加载单个 benchmark 文件，返回标准化后的 DataFrame。

    标准化步骤：
      1. 读取文件（支持 .csv 和 .csv.zip）
      2. 统一列名
      3. 检查必要列
      4. 删除缺失值行
      5. 过滤规模异常的数据集
      6. 拼接输入序列（heavy + linker + light）

    返回 None 表示该数据集不可用（跳过）。
    """

    # ── 1. 读取文件 ─────────────────────────────────────────────────────────────
    try:
        if filepath.endswith(".csv.zip"):
            # 解压 zip，找到其中的第一个 csv 文件
            with zipfile.ZipFile(filepath) as z:
                csv_names = [n for n in z.namelist() if n.endswith(".csv")]
                if not csv_names:
                    return None
                with z.open(csv_names[0]) as f:
                    df = pd.read_csv(f, low_memory=False)
        else:
            df = pd.read_csv(filepath, low_memory=False)
    except Exception as e:
        print(f"  [ERROR] 读取失败: {e}")
        return None

    # ── 2. 统一列名 ─────────────────────────────────────────────────────────────
    # 只重命名在当前 df 中实际存在的列，避免 KeyError
    rename_map = {k: v for k, v in COLUMN_ALIASES.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # ── 3. 检查必要列 ───────────────────────────────────────────────────────────
    if "heavy" not in df.columns:
        print(f"  [SKIP] 缺少重链序列列")
        return None
    if "fitness" not in df.columns:
        print(f"  [SKIP] 缺少 fitness 列")
        return None

    # ── 4. 删除缺失值 ───────────────────────────────────────────────────────────
    # 至少 heavy 和 fitness 不能为空；如果有 light 列，light 也不能为空
    required_cols = ["heavy", "fitness"] + (["light"] if "light" in df.columns else [])
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    # fitness 列转为 float（防止字符串型数字）
    df["fitness"] = pd.to_numeric(df["fitness"], errors="coerce")
    df = df.dropna(subset=["fitness"]).reset_index(drop=True)

    # ── 5. 规模过滤 ─────────────────────────────────────────────────────────────
    if len(df) > cfg.MAX_DATASET_SIZE:
        # 超大数据集（如 li2023 百万级）多为预测值，不是实验 Kd，跳过
        print(f"  [SKIP] 数据集过大（{len(df):,} 条 > 上限 {cfg.MAX_DATASET_SIZE:,}）")
        return None
    if len(df) < cfg.MIN_DATASET_SIZE:
        # 太小的数据集无法做有意义的 train/val/test 划分
        print(f"  [SKIP] 数据量不足（{len(df)} 条）")
        return None

    # ── 6. 拼接输入序列 ─────────────────────────────────────────────────────────
    if "light" in df.columns:
        # 双链抗体（Fv）：heavy + GS linker + light，模拟 scFv 结构
        df["sequence"] = df["heavy"] + cfg.LINKER + df["light"]
    else:
        # 纳米抗体（VHH）：只有重链，直接使用
        df["sequence"] = df["heavy"]

    return df


def load_all_datasets(data_dir: str = cfg.DATA_DIR) -> dict[str, pd.DataFrame]:
    """
    扫描 data_dir 下所有文件，加载全部合法的 binding 数据集。

    返回：
      dict，key = 数据集名（文件名去掉后缀），value = 标准化 DataFrame
    """
    # 找出目录下所有 csv 文件
    all_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith(".csv") or f.endswith(".csv.zip")
    ])

    print(f"\n[数据] 扫描 {len(all_files)} 个文件...")
    datasets = {}

    for fname in all_files:
        # 数据集名：去掉 .csv 或 .csv.zip 后缀
        name = fname.replace(".csv.zip", "").replace(".csv", "")
        fpath = os.path.join(data_dir, fname)

        print(f"\n  [{name}]")
        df = load_one_dataset(fpath)

        if df is not None:
            chain_type = "双链" if "light" in df.columns else "单链(纳米抗体)"
            print(f"  → {len(df)} 条序列，{chain_type}")
            datasets[name] = df

    print(f"\n[数据] 共加载 {len(datasets)} 个有效数据集")
    return datasets
