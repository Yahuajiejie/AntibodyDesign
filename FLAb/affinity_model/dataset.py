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


def _stack_embedding_column(df, column: str) -> np.ndarray:
    """
    将 DataFrame 里存着 np.ndarray 的一列堆成二维矩阵。

    参数：
      df:     含 embedding 列的 DataFrame
      column: embedding 列名，例如 "heavy_embedding"

    返回：
      np.ndarray，形状 [n_rows, embedding_dim]
    """
    if column not in df.columns:
        raise ValueError(f"训练数据缺少 {column!r} 列")
    return np.stack(df[column].values).astype(np.float32)


def build_model_feature_matrix(df) -> np.ndarray:
    """
    根据 cfg.MODEL_FEATURE_MODE 生成 MLP 输入矩阵。

    chain_concat:
      输入需要 v2.1 cache 里的 heavy_embedding 和 light_embedding。
      返回 [heavy_embedding, light_embedding] 的朴素拼接，形状 [N, 2560]。

    scfv_mean:
      沿用 v1 cache 里的 embedding，形状 [N, 1280]。
    """
    if cfg.MODEL_FEATURE_MODE == "chain_concat":
        required = ["heavy_embedding", "light_embedding"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(
                "当前 MODEL_FEATURE_MODE=chain_concat，需要先重新运行 "
                "--mode embed 生成 v2.1 cache；缺少列: "
                f"{missing}"
            )
        heavy = _stack_embedding_column(df, "heavy_embedding")
        light = _stack_embedding_column(df, "light_embedding")
        return np.concatenate([heavy, light], axis=1).astype(np.float32)

    if cfg.MODEL_FEATURE_MODE == "scfv_mean":
        if "embedding" not in df.columns:
            raise ValueError(
                "当前 MODEL_FEATURE_MODE=scfv_mean，但缓存缺少 v1 的 "
                "'embedding' 列；请用 --model_feature_mode scfv_mean 重新运行 --mode embed"
            )
        return _stack_embedding_column(df, "embedding")

    raise ValueError(
        f"未知 MODEL_FEATURE_MODE={cfg.MODEL_FEATURE_MODE!r}，"
        "可选 chain_concat / scfv_mean"
    )


class PairwiseRankingDataset(Dataset):
    """
    Pairwise 排序训练集。

    关键约束：
      只在同一个 compatible_group 内构造 pair。这样 Kd、IC50、不同抗原、
      不同实验体系之间不会被强行比较。

    为什么存索引而不是直接存特征：
      模型输入维度较高，如果每个 pair 都复制两份特征，会浪费大量内存。
      这里保存全局特征矩阵和 pair 的行索引，__getitem__ 时再取。
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
          df:                  含模型特征列、label、compatible_group 的 DataFrame
          label_col:           用来判断强弱的标签列，值越大越强
          group_col:           分组信息所在列的名称
          max_pairs_per_group: 每个组最多采样多少个 pair，防止 O(N²) 爆炸
          min_label_diff:      label 差必须大于该阈值才构造 pair
          seed:                随机采样 pair 的种子
        """
        self.features = build_model_feature_matrix(df)
        self.feature_dim = int(self.features.shape[1])
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
            f"→ {len(self.pairs):,} 个训练对，feature_dim={self.feature_dim}"
            + (f"，跳过 {skipped_groups} 个不可排序组" if skipped_groups else "")
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pos_idx, neg_idx = self.pairs[idx]
        return (
            torch.tensor(self.features[pos_idx]),
            torch.tensor(self.features[neg_idx]),
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
        self.features = build_model_feature_matrix(df)
        self.feature_dim = int(self.features.shape[1])
        self.targets = df[target_col].values.astype(np.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx]),
            torch.tensor(self.targets[idx]),
        )


class ScoringDataset(Dataset):
    """
    验证集 / 测试集用 Dataset。

    评估时只需要 embedding 和真实排序标签；compatible_group 由 trainer
    保留在 DataFrame 中，用于按组计算 Spearman。
    """

    def __init__(self, df, label_col: str = cfg.RANK_LABEL_COL):
        self.features = build_model_feature_matrix(df)
        self.feature_dim = int(self.features.shape[1])
        self.labels = df[label_col].values.astype(np.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx]),
            torch.tensor(self.labels[idx]),
        )
