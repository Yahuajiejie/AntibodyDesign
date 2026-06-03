"""
trainer.py — 训练循环与评估

包含：
  - split_dataset()：按 80/10/10 划分 train/val/test
  - evaluate_spearman()：在给定数据集上计算 Spearman 相关系数
  - train_one_benchmark()：对单个 benchmark 完整训练 + 评估流程
  - run_all_benchmarks()：遍历所有 benchmark，汇总结果
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

from .config import cfg
from .dataset import PairwiseRankingDataset, ScoringDataset
from .model import AffinityMLP
from .losses import LOSS_REGISTRY


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    将一个 benchmark 的数据按 80/10/10 划分为 train/val/test。

    策略：
      - 优先使用分层划分（stratify），保证三个集合的 fitness 分布均匀
      - 数据太少时退回到随机划分（分层划分需要足够的类别多样性）

    参数：
      df: 含 embedding 和 fitness 列的 DataFrame

    返回：
      (train_df, val_df, test_df) 三元组
    """
    try:
        # 对 fitness 值分桶（最多 5 桶），用于分层划分
        n_bins = min(5, len(df) // 3)
        bins = pd.qcut(df["fitness"], q=n_bins, labels=False, duplicates="drop")

        # 第一次划分：80% train，20% temp
        train_df, temp_df = train_test_split(
            df, test_size=(1 - cfg.TRAIN_RATIO),
            stratify=bins, random_state=cfg.SEED
        )
        # 第二次划分：temp 对半分为 val 和 test
        temp_bins = pd.qcut(temp_df["fitness"], q=min(3, len(temp_df) // 2),
                            labels=False, duplicates="drop")
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5,
            stratify=temp_bins, random_state=cfg.SEED
        )
    except Exception:
        # 分层划分失败（数据量太少或 fitness 值不够多样）→ 退回随机划分
        train_df, temp_df = train_test_split(
            df, test_size=(1 - cfg.TRAIN_RATIO), random_state=cfg.SEED
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=cfg.SEED
        )

    # 重置 index，避免后续操作因 index 不连续而出错
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def evaluate_spearman(model: AffinityMLP, dataset: ScoringDataset) -> float:
    """
    在给定数据集上计算模型预测分数与真实 fitness 的 Spearman 相关系数。

    步骤：
      1. 对每条序列做 forward pass，得到预测分数
      2. 将预测分数和真实 fitness 一起计算 Spearman

    Spearman 范围 [-1, 1]，越接近 1 表示排序越准确。
    """
    model.eval()  # 关闭 Dropout，进入推理模式

    all_scores  = []  # 模型输出的亲和力分数
    all_fitness = []  # 真实的 fitness 标签

    # 用较大的 batch size 加速推理（不需要保存梯度，显存压力小）
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    with torch.no_grad():
        for emb, fit in loader:
            # forward pass，得到每条序列的分数
            scores = model(emb.to(cfg.DEVICE)).squeeze(-1)  # [batch]
            all_scores.extend(scores.cpu().numpy().tolist())
            all_fitness.extend(fit.numpy().tolist())

    # fitness 值全部相同时 Spearman 无意义（无法排序）
    if len(set(all_fitness)) < 2:
        return float("nan")

    # 计算 Spearman 相关系数（只取 corr，不取 p_value）
    corr, _ = spearmanr(all_scores, all_fitness)
    return float(corr)


def train_one_benchmark(
    name: str,
    df: pd.DataFrame,
    output_dir: str,
    loss_name: str = "ranknet",
) -> dict:
    """
    对单个 benchmark 完整执行训练 + 评估流程，并保存模型。

    Per-benchmark 训练的含义：
      每个 benchmark 代表同一个抗原的不同抗体变体，为其训练一个专属的小模型。
      这样模型可以专门学习该抗原-抗体系统的亲和力规律，而不是跨系统泛化。

    参数：
      name:       数据集名称（用于日志输出和文件命名）
      df:         含 embedding 和 fitness 列的 DataFrame
      output_dir: 模型权重保存路径

    返回：
      包含 spearman_test 等指标的 dict，用于后续汇总
    """
    print(f"\n{'─'*55}")
    print(f"[训练] {name}  ({len(df)} 条序列)  loss={loss_name}")

    # ── 划分数据集 ─────────────────────────────────────────────────────────────
    train_df, val_df, test_df = split_dataset(df)
    print(f"  划分: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # ── 构建 Dataset ───────────────────────────────────────────────────────────
    # 训练集：构造所有亲和力有高低关系的序列对
    train_dataset = PairwiseRankingDataset(train_df)
    # 验证集/测试集：逐条评分，计算 Spearman
    val_dataset   = ScoringDataset(val_df)
    test_dataset  = ScoringDataset(test_df)

    if len(train_dataset) == 0:
        # 训练集所有序列 fitness 相同，无法构造正负对
        print(f"  [SKIP] 无法构造训练对（fitness 值无差异）")
        return {"name": name, "n": len(df), "spearman_test": float("nan")}

    # 构建 DataLoader，每轮打乱顺序
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,      # 每 epoch 打乱，避免模型记住顺序
        drop_last=False,   # 保留最后不完整的 batch
    )

    # ── 初始化模型、优化器、调度器、损失函数 ──────────────────────────────────
    # 每个 benchmark 的模型从零开始训练（不共享参数）
    model     = AffinityMLP().to(cfg.DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,  # L2 正则化
    )
    # 余弦退火：训练后期平滑降低学习率，帮助收敛到更好的极小值
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS
    )
    # 从注册表实例化损失函数（mse / hinge / ranknet）
    loss_cls = LOSS_REGISTRY[loss_name]
    loss_fn  = loss_cls() if loss_name != "hinge" else loss_cls(margin=cfg.MARGIN)

    # ── 训练循环 ───────────────────────────────────────────────────────────────
    best_val_spearman = -float("inf")  # 记录验证集最优 Spearman
    best_state        = None           # 保存对应的模型参数

    for epoch in range(1, cfg.EPOCHS + 1):
        # 切换到训练模式（Dropout 生效）
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for emb_pos, emb_neg, fit_pos, fit_neg in train_loader:
            # 数据移到 GPU
            emb_pos  = emb_pos.to(cfg.DEVICE)
            emb_neg  = emb_neg.to(cfg.DEVICE)
            # fitness 值移到 GPU（MSE 需要；Hinge/RankNet 传入但不使用）
            fit_pos  = fit_pos.unsqueeze(-1).to(cfg.DEVICE)  # [batch, 1]
            fit_neg  = fit_neg.unsqueeze(-1).to(cfg.DEVICE)  # [batch, 1]

            # 清空上一步积累的梯度
            optimizer.zero_grad()

            # 正向传播：分别对正负样本打分
            score_pos = model(emb_pos)  # [batch, 1]
            score_neg = model(emb_neg)  # [batch, 1]

            # 计算损失（三种 loss 的接口统一：score_pos, score_neg, fit_pos, fit_neg）
            loss = loss_fn(score_pos, score_neg, fit_pos, fit_neg)

            # 反向传播：计算各参数的梯度
            loss.backward()

            # 梯度裁剪：防止小数据集上的梯度爆炸
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 更新参数
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        # 更新学习率（每 epoch 调用一次）
        scheduler.step()

        # 每 10 个 epoch 在验证集上评估一次，并保存最优模型
        if epoch % 10 == 0:
            val_spearman = evaluate_spearman(model, val_dataset)
            avg_loss     = total_loss / max(n_batches, 1)
            print(f"  Epoch {epoch:3d}/{cfg.EPOCHS}  "
                  f"loss={avg_loss:.4f}  val_spearman={val_spearman:.4f}")

            # 如果验证集 Spearman 创历史新高，保存当前参数
            if not np.isnan(val_spearman) and val_spearman > best_val_spearman:
                best_val_spearman = val_spearman
                # clone() 深拷贝，避免后续训练覆盖已保存的最优参数
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # ── 在测试集上最终评估 ──────────────────────────────────────────────────────
    if best_state is not None:
        # 恢复验证集表现最好的那个 checkpoint
        model.load_state_dict(best_state)

    test_spearman = evaluate_spearman(model, test_dataset)
    print(f"  ✓ test_spearman = {test_spearman:.4f}")

    # ── 保存模型 ──────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{name}_{loss_name}.pt")
    torch.save({
        "model_state":    best_state or model.state_dict(),
        "val_spearman":   best_val_spearman,
        "test_spearman":  test_spearman,
        # 同时保存超参，方便后续推理时重建模型
        "hyperparams": {
            "input_dim":  cfg.ESM_EMBEDDING_DIM,
            "hidden_dim": cfg.HIDDEN_DIM,
            "dropout":    cfg.DROPOUT,
        },
    }, save_path)

    return {
        "name":           name,
        "loss":           loss_name,
        "n":              len(df),
        "n_train":        len(train_df),
        "n_test":         len(test_df),
        "val_spearman":   best_val_spearman,
        "spearman_test":  test_spearman,
    }


def run_all_benchmarks(
    embedded_datasets: dict,
    output_dir: str,
    loss_names: list[str] = None,
) -> pd.DataFrame:
    """
    遍历所有 benchmark，依次训练并评估，汇总结果为 DataFrame。

    参数：
      embedded_datasets: dict，key = 数据集名，value = 含 embedding 列的 DataFrame
      output_dir:        模型和 summary.csv 的保存目录

    返回：
      results_df: 每行是一个 benchmark 的评估结果
    """
    # 默认跑全部三种损失函数
    if loss_names is None:
        loss_names = list(LOSS_REGISTRY.keys())  # ["mse", "hinge", "ranknet"]

    all_results = []

    # 外层循环：损失函数；内层循环：每个 benchmark
    # 这样同一个 benchmark 的三种 loss 结果都会出现在 summary 里，方便横向对比
    for loss_name in loss_names:
        loss_dir = os.path.join(output_dir, loss_name)  # 每种 loss 单独存一个子目录
        print(f"\n{'═'*55}")
        print(f"  损失函数: {loss_name.upper()}")
        print(f"{'═'*55}")

        for name, df in embedded_datasets.items():
            result = train_one_benchmark(name, df, loss_dir, loss_name=loss_name)
            all_results.append(result)

    results_df = pd.DataFrame(all_results)

    # 保存完整汇总
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "summary_all_losses.csv")
    results_df.to_csv(summary_path, index=False)

    # 按损失函数分组打印对比
    print(f"\n{'═'*55}")
    print("[消融实验汇总] 各损失函数的平均 Spearman（test）")
    print(f"{'─'*35}")
    for loss_name, grp in results_df.groupby("loss"):
        valid = grp["spearman_test"].dropna()
        print(f"  {loss_name:10s}  mean={valid.mean():.4f}  median={valid.median():.4f}  n={len(valid)}")
    print(f"{'─'*35}")
    print(f"  结果已保存: {summary_path}")

    return results_df
