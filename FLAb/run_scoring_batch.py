"""
FLAb Batch Scoring Runner
用法:
    python run_scoring_batch.py --model esm2_650M
    python run_scoring_batch.py --model esm2_650M esm2_3B
    python run_scoring_batch.py --model all

从 FLAb/ 根目录运行。结果保存在 score_batch/<model_name>/
    - <dataset>_perseq.csv   每条序列的 perplexity 分数
    - summary.csv            所有数据集的 Spearman 汇总
"""

import os
import sys
import argparse
import zipfile
import io
import pandas as pd
import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import EsmTokenizer, EsmForMaskedLM

# ── 设备 ──────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── 模型注册表 ─────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "esm2_8M":   "facebook/esm2_t6_8M_UR50D",
    "esm2_35M":  "facebook/esm2_t12_35M_UR50D",
    "esm2_150M": "facebook/esm2_t30_150M_UR50D",
    "esm2_650M": "facebook/esm2_t33_650M_UR50D",
    "esm2_3B":   "facebook/esm2_t36_3B_UR50D",
    "esm2_15B":  "facebook/esm2_t48_15B_UR50D",
}

# ── 模型加载（懒加载，避免重复初始化）────────────────────────────────────────────
_loaded_models = {}

def get_model(model_name):
    if model_name not in _loaded_models:
        hf_name = MODEL_REGISTRY[model_name]
        print(f"Loading {hf_name} ...")
        tokenizer = EsmTokenizer.from_pretrained(hf_name)
        model = EsmForMaskedLM.from_pretrained(hf_name).to(device)
        model.eval()
        _loaded_models[model_name] = (tokenizer, model)
    return _loaded_models[model_name]

# ── 单条序列打分 ───────────────────────────────────────────────────────────────
def pseudo_perplexity(seq: str, model_name: str) -> float:
    """计算序列的 pseudo-perplexity（masked language model loss）"""
    tokenizer, model = get_model(model_name)
    tensor_input = tokenizer.encode(seq, return_tensors="pt")
    seq_len = tensor_input.size(-1) - 2  # 去掉 [CLS] 和 [SEP]

    if seq_len <= 0:
        return float("nan")

    repeat_input = tensor_input.repeat(seq_len, 1)
    mask = torch.ones(seq_len + 1).diag(1)[:-2]  # 每行 mask 一个位置
    masked_input = repeat_input.masked_fill(mask.bool(), tokenizer.mask_token_id)
    labels = repeat_input.masked_fill(masked_input != tokenizer.mask_token_id, -100)

    with torch.no_grad():
        loss = model(
            masked_input.to(device),
            labels=labels.to(device)
        ).loss

    return float(np.exp(loss.item()))

# ── 读取数据集（支持 .csv 和 .csv.zip）────────────────────────────────────────
def load_dataset(filepath: str, max_size: int = 5000) -> pd.DataFrame | None:
    try:
        if filepath.endswith(".csv.zip"):
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

    # AbRank 列名映射
    if "Ab_heavy_chain_seq" in df.columns:
        df = df.rename(columns={
            "Ab_heavy_chain_seq": "heavy",
            "Ab_light_chain_seq": "light",
        })

    # 必须有 heavy 和 fitness 列
    if "heavy" not in df.columns or "fitness" not in df.columns:
        print(f"  [SKIP] 缺少 heavy 或 fitness 列")
        return None

    # 跳过超大数据集（大多是预测值或 binary label，实验价值低）
    if len(df) > max_size:
        print(f"  [SKIP] 数据集过大（{len(df):,} 条 > 上限 {max_size:,} 条），跳过")
        return None

    drop_cols = ["heavy", "fitness"] + (["light"] if "light" in df.columns else [])
    df = df.dropna(subset=drop_cols).reset_index(drop=True)

    if len(df) < 5:
        print(f"  [SKIP] 有效行数不足 5 条（实际 {len(df)} 条）")
        return None

    return df

# ── 对一个数据集打分 ───────────────────────────────────────────────────────────
def score_dataset(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    has_light = "light" in df.columns

    print(f"  打分中，共 {len(df)} 条序列 {'(重链+轻链)' if has_light else '(仅重链)'} ...")

    df = df.copy()
    df["heavy_perplexity"] = df["heavy"].apply(
        lambda seq: pseudo_perplexity(seq, model_name)
    )

    if has_light:
        df["light_perplexity"] = df["light"].apply(
            lambda seq: pseudo_perplexity(seq, model_name)
        )
        df["average_perplexity"] = (df["heavy_perplexity"] + df["light_perplexity"]) / 2
    else:
        df["average_perplexity"] = df["heavy_perplexity"]

    return df

# ── 计算 Spearman ──────────────────────────────────────────────────────────────
def compute_spearman(df: pd.DataFrame, model_name: str) -> dict:
    valid = df.dropna(subset=["average_perplexity", "fitness"])
    if len(valid) < 5:
        return {"spearman": float("nan"), "p_value": float("nan"), "n": len(valid)}
    corr, pval = spearmanr(valid["average_perplexity"], valid["fitness"])
    return {"spearman": corr, "p_value": pval, "n": len(valid)}

# ── 主流程 ────────────────────────────────────────────────────────────────────
def run(model_names: list[str], data_dir: str = "data/binding", max_rows: int = None):
    all_files = []
    for fname in sorted(os.listdir(data_dir)):
        fpath = os.path.join(data_dir, fname)
        if fname.endswith(".csv") or fname.endswith(".csv.zip"):
            all_files.append(fpath)

    print(f"\n找到 {len(all_files)} 个数据集，将使用模型: {model_names}\n")

    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"模型: {model_name}")
        print(f"{'='*60}")

        output_dir = os.path.join("score_batch", model_name)
        os.makedirs(output_dir, exist_ok=True)

        summary_rows = []

        for fpath in all_files:
            fname = os.path.basename(fpath)
            dataset_name = fname.replace(".csv.zip", "").replace(".csv", "")
            print(f"\n[{dataset_name}]")

            df = load_dataset(fpath, max_size=args.max_size)
            if df is None:
                continue

            # 调试模式：进一步截断行数
            if max_rows and len(df) > max_rows:
                print(f"  [截断] 仅取前 {max_rows} 条（原始 {len(df)} 条）")
                df = df.head(max_rows)

            scored_df = score_dataset(df, model_name)
            stats = compute_spearman(scored_df, model_name)

            print(f"  Spearman={stats['spearman']:.4f}  p={stats['p_value']:.4f}  n={stats['n']}")

            # 保存逐序列打分结果
            perseq_path = os.path.join(output_dir, f"{dataset_name}_perseq.csv")
            scored_df.to_csv(perseq_path, index=False)

            summary_rows.append({
                "dataset": dataset_name,
                "n": stats["n"],
                f"spearman": stats["spearman"],
                f"p_value": stats["p_value"],
            })

        # 保存汇总
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(output_dir, "summary.csv")
        summary_df.to_csv(summary_path, index=False)

        valid = summary_df["spearman"].dropna()
        print(f"\n{'─'*40}")
        print(f"[{model_name}] 完成")
        print(f"  有效数据集: {len(valid)} 个")
        print(f"  平均 Spearman: {valid.mean():.4f}")
        print(f"  中位 Spearman: {valid.median():.4f}")
        print(f"  汇总保存至: {summary_path}")


# ── 入口 ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", nargs="+", default=["esm2_650M"],
        help="模型名，可多选。可选: " + ", ".join(MODEL_REGISTRY.keys()) + ", all"
    )
    parser.add_argument(
        "--data_dir", default="data/binding",
        help="数据目录（默认 data/binding）"
    )
    parser.add_argument(
        "--max_rows", type=int, default=None,
        help="调试用：每个数据集最多处理的行数"
    )
    parser.add_argument(
        "--max_size", type=int, default=5000,
        help="自动跳过超过此行数的数据集（默认 5000，大数据集多为预测值）"
    )
    args = parser.parse_args()

    models = list(MODEL_REGISTRY.keys()) if "all" in args.model else args.model
    for m in models:
        if m not in MODEL_REGISTRY:
            print(f"[ERROR] 未知模型: {m}，可选: {list(MODEL_REGISTRY.keys())}")
            sys.exit(1)

    run(models, data_dir=args.data_dir, max_rows=args.max_rows)
