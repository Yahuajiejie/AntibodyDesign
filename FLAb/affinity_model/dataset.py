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

    def __init__(self, df, max_pairs: int = 10000):
        """
        参数：
          df:        含 'embedding' 和 'fitness' 列的 DataFrame
          max_pairs: 最多使用的训练对数量。
                     N 条序列理论上有 N*(N-1)/2 个对，
                     对大数据集（N>200）会爆炸到百万级，
                     随机采样 max_pairs 个对，保持训练效率。
        """
        embeddings = np.stack(df["embedding"].values).astype(np.float32)
        fitness    = df["fitness"].values.astype(np.float32)
        N = len(df)

        # 先把所有合法的 (i, j) 索引对找出来（只存索引，不存 embedding，省内存）
        # fitness[i] > fitness[j] 的有序对才是正负样本对
        valid_pairs = [
            (i, j)
            for i in range(N)
            for j in range(N)
            if i != j and fitness[i] > fitness[j]
        ]

        # 如果对数超过 max_pairs，随机采样；否则全部使用
        if len(valid_pairs) > max_pairs:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(valid_pairs), size=max_pairs, replace=False)
            valid_pairs = [valid_pairs[k] for k in indices]

        # 根据采样后的索引构造实际数据
        self.pairs = [
            (
                embeddings[i],
                embeddings[j],
                np.float32(fitness[i]),
                np.float32(fitness[j]),
            )
            for i, j in valid_pairs
        ]

        print(f"  [PairwiseDataset] {N} 条序列 → {len(self.pairs)} 个训练对"
              + (f"（从 {N*(N-1)//2:,} 个中随机采样）" if N*(N-1)//2 > max_pairs else ""))

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
