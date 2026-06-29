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


def refresh_model_input_dim():
    """
    根据当前 cfg.MODEL_FEATURE_MODE 刷新 MLP 输入维度。

    argparse 会在模块 import 之后才覆盖 cfg.MODEL_FEATURE_MODE，所以不能只依赖
    config.py 里类变量初始化时算好的 MODEL_INPUT_DIM。
    """
    if cfg.MODEL_FEATURE_MODE == "chain_concat":
        cfg.MODEL_INPUT_DIM = cfg.ESM_EMBEDDING_DIM * 2
    elif cfg.MODEL_FEATURE_MODE == "scfv_mean":
        cfg.MODEL_INPUT_DIM = cfg.ESM_EMBEDDING_DIM
    else:
        raise ValueError(
            f"未知 model_feature_mode={cfg.MODEL_FEATURE_MODE!r}，"
            "可选 chain_concat / scfv_mean"
        )


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
    parser.add_argument(
        "--model_feature_mode",
        choices=["chain_concat", "scfv_mean"],
        default=cfg.MODEL_FEATURE_MODE,
        help=(
            "MLP 输入特征：chain_concat=heavy/light embedding 朴素拼接；"
            "scfv_mean=沿用 v1 的 heavy+linker+light 单序列 embedding"
        ),
    )
    parser.add_argument(
        "--split_objective",
        choices=["rows_balanced", "groups_then_rows"],
        default=cfg.SPLIT_OBJECTIVE,
        help=(
            "group split 搜索目标：rows_balanced=先让样本行数接近 8:1:1；"
            "groups_then_rows=先让 group 数接近 8:1:1，再看样本行数"
        ),
    )
    parser.add_argument(
        "--checkpoint_metric",
        choices=["val_weighted_spearman", "val_median_spearman"],
        default=cfg.CHECKPOINT_METRIC,
        help="保存最佳模型使用的验证集指标",
    )
    parser.add_argument(
        "--min_label_diff",
        type=float,
        default=cfg.MIN_LABEL_DIFF,
        help="构造 ranking pair 时要求 label_pos - label_neg 大于该阈值",
    )
    args = parser.parse_args()

    cfg.MODEL_FEATURE_MODE = args.model_feature_mode
    cfg.SPLIT_OBJECTIVE = args.split_objective
    cfg.CHECKPOINT_METRIC = args.checkpoint_metric
    cfg.MIN_LABEL_DIFF = args.min_label_diff
    refresh_model_input_dim()

    # 固定随机种子，保证结果可复现
    set_seed(cfg.SEED)

    print(f"[设备] {cfg.DEVICE}")
    print(f"[模式] {args.mode}")
    print(f"[特征] {cfg.MODEL_FEATURE_MODE}, input_dim={cfg.MODEL_INPUT_DIM}")
    print(f"[划分目标] {cfg.SPLIT_OBJECTIVE}")
    print(f"[checkpoint] {cfg.CHECKPOINT_METRIC}")
    print(f"[min_label_diff] {cfg.MIN_LABEL_DIFF}")

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
        cols = [
            "loss",
            "feature_mode",
            "val_weighted_spearman",
            "test_weighted_spearman",
            "test_median_spearman",
            "test_n_groups",
        ]
        print(results_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
