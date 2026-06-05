"""
trainer.py — 通用亲和力排序模型训练与评估

当前版本不再为每个 benchmark 单独训练任务头，而是：

  1. 将所有合格的 Kd 数据集合并成一个训练表；
  2. 按 compatible_group 整组划分 train/val/test；
  3. 使用一个共享 AffinityMLP head 训练；
  4. 评估时仍按 compatible_group 分别计算 Spearman，再汇总。

这样可以避免旧实现中的两个核心问题：
  - 测试集和训练集来自同一 benchmark，分数虚高；
  - 不同 assay/单位/抗原之间被当成同一个绝对回归任务。
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from .config import cfg
from .dataset import PairwiseRankingDataset, PointwiseRegressionDataset, ScoringDataset
from .losses import LOSS_REGISTRY
from .model import AffinityMLP


def flatten_datasets(embedded_datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    将 embed_all_datasets() 返回的 dict 拼成一个总表。

    输入：
      embedded_datasets:
        key = 数据集名，value = 含 embedding/label/compatible_group 的 DataFrame

    输出：
      一个 DataFrame，每行是一条抗体序列。

    这里不会改变标签，也不会合并 compatible_group；只是把训练需要的
    多个 CSV 纵向拼接，方便全局模型一次性训练。
    """
    frames = []
    for name, df in embedded_datasets.items():
        piece = df.copy()
        if "dataset" not in piece.columns:
            piece["dataset"] = name
        if cfg.GROUP_COL not in piece.columns:
            piece[cfg.GROUP_COL] = name
        frames.append(piece)

    if not frames:
        raise ValueError("embedded_datasets 为空，无法训练")

    all_df = pd.concat(frames, ignore_index=True)
    required = ["embedding", cfg.RANK_LABEL_COL, cfg.MSE_LABEL_COL, cfg.GROUP_COL]
    missing = [col for col in required if col not in all_df.columns]
    if missing:
        raise ValueError(f"训练数据缺少必要列: {missing}")

    all_df = all_df.dropna(subset=[cfg.RANK_LABEL_COL, cfg.MSE_LABEL_COL]).reset_index(drop=True)
    print(
        f"\n[训练数据] {len(all_df):,} 条序列，"
        f"{all_df[cfg.GROUP_COL].nunique()} 个 compatible_group"
    )
    return all_df


def split_by_group(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按 compatible_group 整组划分 train/val/test。

    为什么不用随机行划分：
      随机行划分会让同一个 benchmark 的相近突变体同时出现在训练和测试中，
      容易得到不真实的高 Spearman。按组划分更接近“未知数据集/未知抗原”
      的泛化场景。
    """
    group_names = np.array(sorted(df[cfg.GROUP_COL].astype(str).unique()))
    if len(group_names) < 3:
        raise ValueError("compatible_group 少于 3 个，无法按组划分 train/val/test")

    rng = np.random.default_rng(cfg.SEED)
    rng.shuffle(group_names)

    n_groups = len(group_names)
    n_test = max(1, int(round(n_groups * cfg.TEST_RATIO)))
    n_val = max(1, int(round(n_groups * cfg.VAL_RATIO)))
    if n_val + n_test >= n_groups:
        n_val = 1
        n_test = 1

    test_groups = set(group_names[:n_test])
    val_groups = set(group_names[n_test:n_test + n_val])
    train_groups = set(group_names[n_test + n_val:])

    train_df = df[df[cfg.GROUP_COL].astype(str).isin(train_groups)].reset_index(drop=True)
    val_df = df[df[cfg.GROUP_COL].astype(str).isin(val_groups)].reset_index(drop=True)
    test_df = df[df[cfg.GROUP_COL].astype(str).isin(test_groups)].reset_index(drop=True)

    print(
        "[划分] "
        f"train={len(train_df):,} 条/{len(train_groups)} 组，"
        f"val={len(val_df):,} 条/{len(val_groups)} 组，"
        f"test={len(test_df):,} 条/{len(test_groups)} 组"
    )
    return train_df, val_df, test_df


def _predict_scores(model: AffinityMLP, df: pd.DataFrame) -> np.ndarray:
    """
    对 DataFrame 中每条序列推理，返回模型分数数组。

    DataLoader 是 PyTorch 的批处理工具；shuffle=False 保证输出顺序
    与 df 行顺序一致，后续才能把分数安全地写回 DataFrame。
    """
    model.eval()
    dataset = ScoringDataset(df, label_col=cfg.RANK_LABEL_COL)
    loader = DataLoader(dataset, batch_size=cfg.EVAL_BATCH_SIZE, shuffle=False)

    scores = []
    with torch.no_grad():
        for emb, _ in loader:
            pred = model(emb.to(cfg.DEVICE)).squeeze(-1)
            scores.extend(pred.cpu().numpy().tolist())
    return np.asarray(scores, dtype=np.float32)


def evaluate_by_group(
    model: AffinityMLP,
    df: pd.DataFrame,
    split_name: str,
) -> tuple[dict, pd.DataFrame]:
    """
    按 compatible_group 计算 Spearman，并汇总为整体指标。

    返回：
      summary:
        mean_spearman / median_spearman / weighted_mean_spearman / n_groups
      detail_df:
        每个 compatible_group 一行，便于质检哪些数据集表现好/差。
    """
    if len(df) == 0:
        empty = pd.DataFrame()
        return {
            f"{split_name}_mean_spearman": float("nan"),
            f"{split_name}_median_spearman": float("nan"),
            f"{split_name}_weighted_spearman": float("nan"),
            f"{split_name}_n_groups": 0,
        }, empty

    scored = df.copy()
    scored["prediction"] = _predict_scores(model, scored)

    records = []
    for group_name, group_df in scored.groupby(cfg.GROUP_COL):
        n = len(group_df)
        n_unique = group_df[cfg.RANK_LABEL_COL].nunique()
        if n < cfg.MIN_GROUP_SIZE or n_unique < 2:
            continue

        corr, p_value = spearmanr(
            group_df["prediction"].values,
            group_df[cfg.RANK_LABEL_COL].values,
        )
        if np.isnan(corr):
            continue

        records.append({
            "split": split_name,
            "compatible_group": group_name,
            "dataset": group_df["dataset"].iloc[0] if "dataset" in group_df else group_name,
            "n": n,
            "n_unique_label": int(n_unique),
            "spearman": float(corr),
            "p_value": float(p_value),
            "assay_family": group_df["assay_family"].iloc[0] if "assay_family" in group_df else "",
            "assay_units": group_df["assay_units"].iloc[0] if "assay_units" in group_df else "",
        })

    detail_df = pd.DataFrame(records)
    if detail_df.empty:
        summary = {
            f"{split_name}_mean_spearman": float("nan"),
            f"{split_name}_median_spearman": float("nan"),
            f"{split_name}_weighted_spearman": float("nan"),
            f"{split_name}_n_groups": 0,
        }
        return summary, detail_df

    summary = {
        f"{split_name}_mean_spearman": float(detail_df["spearman"].mean()),
        f"{split_name}_median_spearman": float(detail_df["spearman"].median()),
        f"{split_name}_weighted_spearman": float(
            np.average(detail_df["spearman"], weights=detail_df["n"])
        ),
        f"{split_name}_n_groups": int(len(detail_df)),
    }
    return summary, detail_df


def _build_train_loader(train_df: pd.DataFrame, loss_name: str) -> tuple[DataLoader, nn.Module]:
    """
    根据 loss_name 构造训练 Dataset 和损失函数。

    RankNet/Hinge:
      使用 PairwiseRankingDataset，batch 中每条样本是一对抗体。

    MSE:
      使用 PointwiseRegressionDataset，batch 中每条样本是一条抗体，
      target 是 label_z。
    """
    loss_cls = LOSS_REGISTRY[loss_name]

    if loss_name == "mse":
        dataset = PointwiseRegressionDataset(train_df, target_col=cfg.MSE_LABEL_COL)
        loss_fn = loss_cls()
    else:
        dataset = PairwiseRankingDataset(
            train_df,
            label_col=cfg.RANK_LABEL_COL,
            group_col=cfg.GROUP_COL,
            max_pairs_per_group=cfg.MAX_PAIRS_PER_GROUP,
            min_label_diff=cfg.MIN_LABEL_DIFF,
            seed=cfg.SEED,
        )
        loss_fn = loss_cls(margin=cfg.MARGIN) if loss_name == "hinge" else loss_cls()

    if len(dataset) == 0:
        raise ValueError(f"{loss_name} 训练集为空，无法训练")

    loader = DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )
    return loader, loss_fn


def train_global_model(
    embedded_datasets: dict[str, pd.DataFrame],
    output_dir: str,
    loss_name: str = "ranknet",
) -> dict:
    """
    训练一个跨数据集共享参数的通用模型。

    每个 loss_name 会独立训练一个模型，便于做 RankNet/Hinge/MSE 消融。
    """
    print(f"\n{'═' * 70}")
    print(f"[全局训练] loss={loss_name}")
    print(f"{'═' * 70}")

    all_df = flatten_datasets(embedded_datasets)
    train_df, val_df, test_df = split_by_group(all_df)

    train_loader, loss_fn = _build_train_loader(train_df, loss_name)

    model = AffinityMLP().to(cfg.DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)

    best_state = None
    best_val = -float("inf")
    best_epoch = 0

    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()

            if loss_name == "mse":
                emb, target = batch
                emb = emb.to(cfg.DEVICE)
                target = target.unsqueeze(-1).to(cfg.DEVICE)
                score = model(emb)
                loss = loss_fn(score, target)
            else:
                emb_pos, emb_neg, fit_pos, fit_neg = batch
                emb_pos = emb_pos.to(cfg.DEVICE)
                emb_neg = emb_neg.to(cfg.DEVICE)
                fit_pos = fit_pos.unsqueeze(-1).to(cfg.DEVICE)
                fit_neg = fit_neg.unsqueeze(-1).to(cfg.DEVICE)

                score_pos = model(emb_pos)
                score_neg = model(emb_neg)
                loss = loss_fn(score_pos, score_neg, fit_pos, fit_neg)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        scheduler.step()

        if epoch % cfg.EVAL_EVERY == 0 or epoch == cfg.EPOCHS:
            val_summary, _ = evaluate_by_group(model, val_df, "val")
            val_score = val_summary["val_mean_spearman"]
            avg_loss = total_loss / max(n_batches, 1)
            print(
                f"  Epoch {epoch:3d}/{cfg.EPOCHS} "
                f"loss={avg_loss:.4f} val_mean_spearman={val_score:.4f}"
            )

            if not np.isnan(val_score) and val_score > best_val:
                best_val = val_score
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

    if best_state is not None:
        model.load_state_dict(best_state)

    val_summary, val_detail = evaluate_by_group(model, val_df, "val")
    test_summary, test_detail = evaluate_by_group(model, test_df, "test")

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"global_{loss_name}.pt")
    torch.save({
        "model_state": best_state or model.state_dict(),
        "loss": loss_name,
        "best_epoch": best_epoch,
        "best_val_mean_spearman": best_val,
        "hyperparams": {
            "input_dim": cfg.ESM_EMBEDDING_DIM,
            "hidden_dim": cfg.HIDDEN_DIM,
            "dropout": cfg.DROPOUT,
            "label_col": cfg.RANK_LABEL_COL,
            "mse_label_col": cfg.MSE_LABEL_COL,
            "group_col": cfg.GROUP_COL,
        },
        "split_groups": {
            "train": sorted(train_df[cfg.GROUP_COL].astype(str).unique().tolist()),
            "val": sorted(val_df[cfg.GROUP_COL].astype(str).unique().tolist()),
            "test": sorted(test_df[cfg.GROUP_COL].astype(str).unique().tolist()),
        },
    }, model_path)

    val_detail_path = os.path.join(output_dir, f"global_{loss_name}_val_by_group.csv")
    test_detail_path = os.path.join(output_dir, f"global_{loss_name}_test_by_group.csv")
    val_detail.to_csv(val_detail_path, index=False)
    test_detail.to_csv(test_detail_path, index=False)

    result = {
        "loss": loss_name,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "n_train_groups": train_df[cfg.GROUP_COL].nunique(),
        "n_val_groups": val_df[cfg.GROUP_COL].nunique(),
        "n_test_groups": test_df[cfg.GROUP_COL].nunique(),
        "best_epoch": best_epoch,
        **val_summary,
        **test_summary,
        "model_path": model_path,
    }

    print(
        f"[完成] loss={loss_name} "
        f"val_mean={result['val_mean_spearman']:.4f} "
        f"test_mean={result['test_mean_spearman']:.4f} "
        f"model={model_path}"
    )
    return result


def run_global_training(
    embedded_datasets: dict[str, pd.DataFrame],
    output_dir: str,
    loss_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    对多个 loss 依次训练全局模型，并保存汇总表。
    """
    if loss_names is None:
        loss_names = list(LOSS_REGISTRY.keys())

    results = []
    for loss_name in loss_names:
        results.append(train_global_model(embedded_datasets, output_dir, loss_name))

    results_df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "summary_global_losses.csv")
    results_df.to_csv(summary_path, index=False)

    print(f"\n[汇总] 全局模型结果已保存: {summary_path}")
    cols = [
        "loss",
        "val_mean_spearman",
        "test_mean_spearman",
        "test_median_spearman",
        "test_n_groups",
    ]
    print(results_df[cols].to_string(index=False))
    return results_df


# 向后兼容旧入口名；语义已经从 per-benchmark 改为 global training。
run_all_benchmarks = run_global_training
