"""
dataset.py — PyTorch Dataset 类定义

这里包含三种 Dataset：

  1. PairwiseRankingDataset
     用于 RankNet / Ranking Hinge。每个样本是一对可比较抗体：
     label_pos > label_neg，并且二者必须来自同一个 compatible_group。

  2. PointwiseRegressionDataset
     用于 MSE。每个样本是一条序列及其组内标准化标签 label_z。

  3. ScoringDataset
     用于验证/测试。每个样本是一条序列及其排序标签 label。
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import cfg


class PairwiseRankingDataset(Dataset):
    """
    Pairwise 排序训练集。

    关键约束：
      只在同一个 compatible_group 内构造 pair。这样 Kd、IC50、不同抗原、
      不同实验体系之间不会被强行比较。

    为什么存索引而不是直接存 embedding：
      embedding 维度是 1280，如果每个 pair 都复制两份 embedding，会浪费大量内存。
      这里保存全局 embedding 矩阵和 pair 的行索引，__getitem__ 时再取。
    """

    def __init__(
        self,
        df,
        label_col: str = cfg.RANK_LABEL_COL,
        group_col: str = cfg.GROUP_COL,
        max_pairs_per_group: int = cfg.MAX_PAIRS_PER_GROUP,
        min_label_diff: float = cfg.MIN_LABEL_DIFF,
        seed: int = cfg.SEED,
    ):
        """
        参数：
          df:                  含 embedding、label、compatible_group 的 DataFrame
          label_col:           用来判断强弱的标签列，值越大越强
          group_col:           分组信息所在列的名称
          max_pairs_per_group: 每个组最多采样多少个 pair，防止 O(N²) 爆炸
          min_label_diff:      label 差必须大于该阈值才构造 pair
          seed:                随机采样 pair 的种子
        """
        self.embeddings = np.stack(df["embedding"].values).astype(np.float32)
        # stack 堆叠，就是简单的把一堆向量堆成二维数组，而不是把他们弄成栈
        self.labels = df[label_col].values.astype(np.float32)
        self.groups = df[group_col].astype(str).values

        rng = np.random.default_rng(seed)
        # random number generater
        pairs: list[tuple[int, int]] = []
        skipped_groups = 0

        for group_name in sorted(set(self.groups)):
            group_indices = np.where(self.groups == group_name)[0]
            group_labels = self.labels[group_indices]

            if len(group_indices) < 2 or len(np.unique(group_labels)) < 2:
                skipped_groups += 1
                continue

            local_pairs = [
                (int(group_indices[i]), int(group_indices[j]))
                for i in range(len(group_indices))
                for j in range(len(group_indices))
                if group_labels[i] - group_labels[j] > min_label_diff
            ]

            if len(local_pairs) > max_pairs_per_group:
                chosen = rng.choice(
                    len(local_pairs),
                    size=max_pairs_per_group,
                    replace=False,
                )
                local_pairs = [local_pairs[k] for k in chosen]

            pairs.extend(local_pairs)

        self.pairs = pairs
        print(
            f"  [PairwiseDataset] {len(df)} 条序列，{len(set(self.groups))} 个组 "
            f"→ {len(self.pairs):,} 个训练对"
            + (f"，跳过 {skipped_groups} 个不可排序组" if skipped_groups else "")
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pos_idx, neg_idx = self.pairs[idx]
        return (
            torch.tensor(self.embeddings[pos_idx]),
            torch.tensor(self.embeddings[neg_idx]),
            torch.tensor(self.labels[pos_idx]),
            torch.tensor(self.labels[neg_idx]),
        )


class PointwiseRegressionDataset(Dataset):
    """
    MSE 训练集。

    MSE 不直接回归原始 Kd 或 -logKd，而回归组内标准化后的 label_z。
    这让不同实验体系的动态范围不会主导损失。
    """

    def __init__(self, df, target_col: str = cfg.MSE_LABEL_COL):
        self.embeddings = np.stack(df["embedding"].values).astype(np.float32)
        self.targets = df[target_col].values.astype(np.float32)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.embeddings[idx]),
            torch.tensor(self.targets[idx]),
        )


class ScoringDataset(Dataset):
    """
    验证集 / 测试集用 Dataset。

    评估时只需要 embedding 和真实排序标签；compatible_group 由 trainer
    保留在 DataFrame 中，用于按组计算 Spearman。
    """

    def __init__(self, df, label_col: str = cfg.RANK_LABEL_COL):
        self.embeddings = np.stack(df["embedding"].values).astype(np.float32)
        self.labels = df[label_col].values.astype(np.float32)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.embeddings[idx]),
            torch.tensor(self.labels[idx]),
        )
