"""
run_v1.py — AffinityMLPSimplified(v1) 入口脚本

这个文件是从 commit 15c8c70 的 FLAb/train.py 逻辑改写来的薄入口。
它只导入当前目录下的 v1 模块，不依赖 AffinityMLP(v2) 或 AffinityTransformer(v3)。
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

FLAB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if FLAB_ROOT not in sys.path:
    sys.path.insert(0, FLAB_ROOT)

from model.config import cfg
from model.data_loader import load_all_datasets
from model.embeddings import embed_all_datasets, load_cached_datasets
from model.losses import LOSS_REGISTRY
from model.trainer import run_all_benchmarks


def set_seed(seed: int = cfg.SEED) -> None:
    """固定随机源，保证实验可复现。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AffinityMLPSimplified v1 per-benchmark MLP baseline"
    )
    parser.add_argument(
        "--mode",
        choices=["embed", "train", "all"],
        default="all",
        help="embed=只提取 embedding；train=只训练；all=全流程",
    )
    parser.add_argument(
        "--data_dir",
        default=cfg.DATA_DIR,
        help=f"FLAb binding 数据目录（默认: {cfg.DATA_DIR}）",
    )
    parser.add_argument(
        "--cache_dir",
        default=cfg.EMBED_CACHE_DIR,
        help=f"ESM2 embedding 缓存目录（默认: {cfg.EMBED_CACHE_DIR}）",
    )
    parser.add_argument(
        "--output_dir",
        default=cfg.OUTPUT_DIR,
        help=f"模型和结果保存目录（默认: {cfg.OUTPUT_DIR}）",
    )
    parser.add_argument(
        "--loss",
        nargs="+",
        default=list(LOSS_REGISTRY.keys()),
        choices=list(LOSS_REGISTRY.keys()),
        help="损失函数，可多选（默认全部）: mse hinge ranknet",
    )
    args = parser.parse_args()

    set_seed(cfg.SEED)
    print(f"[v1] device={cfg.DEVICE}")
    print(f"[v1] mode={args.mode}")

    if args.mode in ("embed", "all"):
        datasets = load_all_datasets(args.data_dir)
        if not datasets:
            raise RuntimeError("没有加载到任何数据集，请检查路径和数据格式")
        embedded_datasets = embed_all_datasets(datasets, args.cache_dir)

    if args.mode in ("train", "all"):
        if args.mode == "train":
            embedded_datasets = load_cached_datasets(args.cache_dir)

        results_df = run_all_benchmarks(
            embedded_datasets,
            args.output_dir,
            loss_names=args.loss,
        )
        print("\n[v1] 前10名 benchmark（按 test Spearman 排序）")
        top = results_df.sort_values("spearman_test", ascending=False).head(10)
        print(top[["name", "n", "spearman_test"]].to_string(index=False))


if __name__ == "__main__":
    main()
