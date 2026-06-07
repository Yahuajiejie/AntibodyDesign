"""
dataset.py — PyTorch Dataset 类定义

包含两种 Dataset：
  1. PairwiseRankingDataset：训练用，每个样本是一对序列（高亲和力 vs 低亲和力）
  2. ScoringDataset：验证/测试用，每个样本是单条序列及其 fitness 标签
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class PairwiseRankingDataset(Dataset):
    """
    训练集用的 Dataset。

    核心思路：
      亲和力预测本质上是一个排序问题——我们不需要预测绝对 Kd 值，
      只需要让模型判断"哪个抗体比哪个好"。

      因此构造所有满足 fitness_A > fitness_B 的有序对 (A, B)，
      训练目标是让 score_A > score_B + margin。

    数据量分析：
      N 条序列最多产生 O(N²) 个对，对小数据集（<5000 条）完全可接受。
      实际对数 = 满足 fitness_i > fitness_j 的 (i,j) 数量。
    """

    def __init__(self, df):
        """
        参数：
          df: 含 'embedding'（numpy array）和 'fitness'（float）列的 DataFrame
        """
        # 将所有 embedding 堆叠成矩阵，形状 [N, 1280]
        embeddings = np.stack(df["embedding"].values).astype(np.float32)
        fitness    = df["fitness"].values.astype(np.float32)

        N = len(df)
        # 存储 (emb_正样本, emb_负样本, fitness_正, fitness_负)
        # Hinge/RankNet 只用前两个，MSE 需要全部四个
        self.pairs = []

        # 枚举所有有序对 (i, j)，i 的亲和力高于 j
        for i in range(N):
            for j in range(N):
                if i != j and fitness[i] > fitness[j]:
                    self.pairs.append((
                        embeddings[i],        # 正样本 embedding
                        embeddings[j],        # 负样本 embedding
                        np.float32(fitness[i]),  # 正样本真实 fitness（MSE 需要）
                        np.float32(fitness[j]),  # 负样本真实 fitness（MSE 需要）
                    ))

        print(f"  [PairwiseDataset] {N} 条序列 → {len(self.pairs)} 个训练对")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # 返回 (正样本 emb, 负样本 emb, 正样本 fitness, 负样本 fitness)
        emb_pos, emb_neg, fit_pos, fit_neg = self.pairs[idx]
        return (
            torch.tensor(emb_pos),
            torch.tensor(emb_neg),
            torch.tensor(fit_pos),
            torch.tensor(fit_neg),
        )


class ScoringDataset(Dataset):
    """
    验证集 / 测试集用的 Dataset。

    每个样本是单条序列的 embedding + fitness 标签，
    用于评估模型预测分数与真实亲和力的 Spearman 相关性。
    """

    def __init__(self, df):
        """
        参数：
          df: 含 'embedding' 和 'fitness' 列的 DataFrame
        """
        # 将所有 embedding 堆叠，fitness 转为 float32
        self.embeddings = np.stack(df["embedding"].values).astype(np.float32)
        self.fitness    = df["fitness"].values.astype(np.float32)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        # 返回 (embedding tensor, fitness tensor)
        return (
            torch.tensor(self.embeddings[idx]),
            torch.tensor(self.fitness[idx]),
        )
