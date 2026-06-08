"""
losses.py — 三种亲和力排序损失函数

本模块实现三种损失函数用于消融实验（ablation study），对比它们的效果：

  1. MSELoss        — 均方误差，假设误差服从正态分布（高斯噪声）
  2. PairwiseHinge  — 成对铰链损失，基于 SVM 间隔理论（无分布假设）
  3. RankNetLoss    — Bradley-Terry 模型的最大似然，假设误差服从 Gumbel 分布

三种损失在优化目标上的根本区别：
  MSE:       直接最小化预测值与组内标准化标签的差距（回归视角）
  Hinge:     要求正样本分数比负样本高出一个固定 margin（几何视角）
  RankNet:   最大化"观测到正确排序"的对数似然（概率视角）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import cfg


# ── 1. MSE Loss（均方误差）──────────────────────────────────────────────────────

class MSELoss(nn.Module):
    """
    均方误差损失，回归组内标准化后的 label_z。

    统计假设：
      预测误差 ε = score - fitness 服从正态分布 N(0, σ²)。
      最小化 MSE 等价于在高斯误差假设下做最大似然估计。

    输入模式：
      与 Pairwise 损失不同，MSE 不构造序列对，而是直接接收：
        score:  [batch, 1]
        target: [batch, 1]

    注意：
      target 必须是 data_loader.py 生成的 label_z，而不是原始 Kd、
      -logKd 或 IC50。这样 MSE baseline 才不会把不同实验体系的量纲混用。
    """

    def __init__(self):
        super().__init__()
        # PyTorch 内置的 MSE，reduction='mean' 对 batch 取平均
        self.mse = nn.MSELoss(reduction="mean")

    def forward(
        self,
        score: torch.Tensor,   # [batch, 1]：模型预测分数
        target: torch.Tensor,  # [batch, 1]：组内标准化标签 label_z
    ) -> torch.Tensor:
        """调用 PyTorch 内置 nn.MSELoss，对 batch 取均值。"""
        return self.mse(score, target)


# ── 2. Pairwise Hinge Loss（成对铰链损失）──────────────────────────────────────

class PairwiseHingeLoss(nn.Module):
    """
    成对铰链排序损失（Pairwise Hinge Ranking Loss）。

    来源：SVM 分类框架的排序扩展（Ranking SVM）。

    几何假设：
      对于每对 (A, B)（fitness_A > fitness_B），
      我们希望 score_A - score_B ≥ margin，
      即正样本的分数至少比负样本高出 margin。

    损失公式：
      L = max(0, margin - (score_A - score_B))
      差距足够时 loss=0（无惩罚），差距不够时线性惩罚。

    margin 的作用：
      - margin=0：只要 A > B 就不产生损失，容易得到平凡的常数解
      - margin>0：强制要求一定的分离度，促使模型学到更有区分度的特征
    """

    def __init__(self, margin: float = cfg.MARGIN):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        score_pos: torch.Tensor,    # [batch, 1]：高亲和力样本的预测分数
        score_neg: torch.Tensor,    # [batch, 1]：低亲和力样本的预测分数
        fitness_pos: torch.Tensor = None,  # 占位参数，与接口保持统一（未使用）
        fitness_neg: torch.Tensor = None,  # 占位参数，与接口保持统一（未使用）
    ) -> torch.Tensor:
        # 计算预测分数差（希望 score_pos - score_neg > margin）
        diff = score_pos - score_neg  # [batch, 1]

        # Hinge：当差距小于 margin 时产生线性惩罚，差距足够时 loss=0
        loss = torch.clamp(self.margin - diff, min=0.0)
        return loss.mean()


# ── 3. RankNet Loss（Bradley-Terry 模型）───────────────────────────────────────

class RankNetLoss(nn.Module):
    """
    RankNet 损失，基于 Bradley-Terry 配对比较模型。

    概率假设（Bradley-Terry 模型）：
      对于任意两个抗体 A 和 B，A 比 B 亲和力更高的概率服从 logistic 函数：
        P(A ≻ B) = σ(score_A - score_B) = 1 / (1 + exp(-(score_A - score_B)))

      这等价于假设：每个抗体的真实 fitness 加上独立同分布的 Gumbel 噪声，
      然后观察谁的带噪 fitness 更高。

    损失公式（最大化正确排序的对数似然）：
      L = -log P(A ≻ B)
        = -log σ(score_A - score_B)
        = log(1 + exp(-(score_A - score_B)))
        = softplus(score_B - score_A)   ← 数值稳定的等价写法

    为什么对抗体任务合理：
      实验测量的 Kd 值本身有测量噪声，
      "A 比 B 结合更强"这一相对判断比精确的 Kd 比值更可靠。
      Gumbel 噪声正好描述了这种测量不确定性下的排序观测。
    """

    def forward(
        self,
        score_pos: torch.Tensor,    # [batch, 1]：高亲和力样本的预测分数
        score_neg: torch.Tensor,    # [batch, 1]：低亲和力样本的预测分数
        fitness_pos: torch.Tensor = None,  # 占位参数，与接口保持统一（未使用）
        fitness_neg: torch.Tensor = None,  # 占位参数，与接口保持统一（未使用）
    ) -> torch.Tensor:
        # softplus(x) = log(1 + exp(x))，数值稳定版本的 -log(sigmoid(-x))
        # 这里 x = score_neg - score_pos（正样本的分数应该更高，所以这个值越小越好）
        loss = F.softplus(score_neg - score_pos)
        return loss.mean()


# ── 损失函数注册表 ──────────────────────────────────────────────────────────────
# 将名称映射到类，方便 train.py 通过字符串选择损失函数

LOSS_REGISTRY: dict[str, type] = {
    "mse":     MSELoss,
    "hinge":   PairwiseHingeLoss,
    "ranknet": RankNetLoss,
}
