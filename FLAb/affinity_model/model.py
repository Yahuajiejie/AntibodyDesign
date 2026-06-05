"""
model.py — 通用 MLP 亲和力预测头

接在冻结的 ESM2 embedding 后面，输出一个标量分数。
分数越高，模型认为该抗体的结合亲和力越强。

重要变化：
  当前模型是跨数据集共享参数的通用 head，不再为每个 benchmark
  单独创建任务头。不同实验体系的不可比性由 data_loader/dataset
  在标签标准化和组内 pair 构造阶段处理。
"""

import torch
import torch.nn as nn

from .config import cfg


class AffinityMLP(nn.Module):
    """
    两层共享 MLP 预测头。

    架构：
      输入 [1280] → Linear → GELU → Dropout → Linear → 输出 [1]

    输入：ESM2 mean pooling embedding，维度 = cfg.ESM_EMBEDDING_DIM
    输出：单个标量分数，值越大代表预测亲和力越强
    """

    def __init__(
        self,
        input_dim:  int   = cfg.ESM_EMBEDDING_DIM,
        hidden_dim: int   = cfg.HIDDEN_DIM,
        dropout:    float = cfg.DROPOUT,
    ):
        super().__init__()

        self.net = nn.Sequential(
            # 第一层：高维 embedding → 低维特征
            nn.Linear(input_dim, hidden_dim),

            # GELU 激活：比 ReLU 更平滑，在 transformer 相关任务上表现更好
            # ESM2 内部也使用 GELU，保持一致
            nn.GELU(),

            # Dropout：训练时随机丢弃 dropout 比例的神经元，防止过拟合
            # 评估时自动关闭（model.eval() 后生效）
            nn.Dropout(dropout),

            # 第二层：输出单个亲和力分数
            nn.Linear(hidden_dim, 1),
        )

        # Xavier 初始化：防止深层网络训练初期的梯度消失/爆炸
        self._init_weights()

    def _init_weights(self):
        """对所有 Linear 层做 Xavier Uniform 初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)  # bias 初始化为零

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数：
          x: [batch_size, input_dim] 的 embedding 矩阵

        返回：
          [batch_size, 1] 的亲和力预测分数
        """
        return self.net(x)
