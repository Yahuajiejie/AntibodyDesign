"""
train.py — 主入口脚本

用法（从 FLAb/ 根目录运行）：

  # 步骤1：提取所有序列的 ESM2 embedding 并缓存到磁盘（只需跑一次，耗时）
  python train.py --mode embed

  # 步骤2：从缓存读取 embedding，训练共享 MLP head，按组评估 Spearman
  python train.py --mode train

  # 一键全流程（embed + train）
  python train.py --mode all

  # 后台挂起（云端推荐）
  nohup python train.py --mode all > log_train.txt 2>&1 &
"""

import argparse
import torch
import numpy as np

from affinity_model.config import cfg
from affinity_model.data_loader import load_all_datasets
from affinity_model.embeddings import embed_all_datasets, load_cached_datasets
from affinity_model.losses import LOSS_REGISTRY
from affinity_model.trainer import run_global_training


def set_seed(seed: int = cfg.SEED):
    """固定所有随机源，保证实验可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # 让 cuDNN 使用确定性算法（略微降低速度，但结果完全可复现）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(
        description="FLAb 通用抗体亲和力排序模型训练"
    )
    parser.add_argument(
        "--mode",
        choices=["embed", "train", "all"],
        default="all",
        help=(
            "运行模式：\n"
            "  embed = 仅提取 ESM2 embedding 并缓存\n"
            "  train = 仅训练（需要先跑 embed）\n"
            "  all   = 全流程（默认）"
        ),
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
        default=list(LOSS_REGISTRY.keys()),  # 默认跑全部三种
        choices=list(LOSS_REGISTRY.keys()),
        help="损失函数，可多选（默认全部）: mse hinge ranknet",
    )
    args = parser.parse_args()

    # 固定随机种子，保证结果可复现
    set_seed(cfg.SEED)

    print(f"[设备] {cfg.DEVICE}")
    print(f"[模式] {args.mode}")

    # ── Embed 阶段：提取 ESM2 embedding ─────────────────────────────────────────
    if args.mode in ("embed", "all"):
        # 加载所有合格的 binding 数据集
        datasets = load_all_datasets(args.data_dir)
        if not datasets:
            raise RuntimeError("没有加载到任何数据集，请检查路径和数据格式")

        # 提取 embedding 并缓存到磁盘
        # 这步最耗时（GPU 密集），完成后不需要再跑
        embedded_datasets = embed_all_datasets(datasets, args.cache_dir)

    # ── Train 阶段：从缓存读取 embedding，训练 MLP ───────────────────────────────
    if args.mode in ("train", "all"):
        if args.mode == "train":
            # train-only 模式：直接从磁盘读取之前缓存的 embedded_datasets
            embedded_datasets = load_cached_datasets(args.cache_dir)

        # 每种 loss 各训练一个全局共享模型，不再为每个 benchmark 单独建 head
        results_df = run_global_training(
            embedded_datasets,
            args.output_dir,
            loss_names=args.loss,
        )

        print("\n[全局模型结果]")
        cols = ["loss", "val_mean_spearman", "test_mean_spearman", "test_n_groups"]
        print(results_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
