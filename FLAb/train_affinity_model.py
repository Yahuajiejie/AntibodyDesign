"""
FLAb 抗体亲和力预测模型 —— 训练脚本

整体思路：
  1. 从 FLAb binding 数据集加载抗体序列和亲和力标签
  2. 用冻结的 ESM2-650M 提取每条序列的 embedding（mean pooling）
  3. 在 embedding 上训练一个轻量 MLP 回归头，使用 pairwise ranking loss
  4. 在每个 benchmark 的测试集上计算 Spearman 相关系数

为什么用 pairwise ranking loss 而不是 MSE：
  - 比赛评分是 Spearman（排序相关），不是绝对值预测精度
  - Pairwise loss 直接优化"哪个抗体排名更高"，和评分目标对齐
  - MSE 优化绝对值，和 Spearman 之间有 gap

用法（从 FLAb/ 根目录运行）：
  # 先提取并缓存所有 embedding（耗时，只需跑一次）
  python train_affinity_model.py --mode embed

  # 训练 + 评估（per-benchmark 模式，每个数据集独立训练）
  python train_affinity_model.py --mode train

  # 一键跑完（embed + train）
  python train_affinity_model.py --mode all
"""

import os
import sys
import argparse
import zipfile
import pickle
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from transformers import EsmTokenizer, EsmForMaskedLM, EsmModel


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 超参数配置
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    """
    所有超参数集中在这里，方便调整。
    修改参数只需改这里，不需要翻遍整个代码。
    """

    # ── ESM2 backbone ──────────────────────────────────────────────────────────
    ESM_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"  # 使用 650M 版本
    ESM_EMBEDDING_DIM = 1280                          # 650M 对应的 embedding 维度
    LINKER = "GGGGSGGGGSGGGGS"                        # 连接重链和轻链的 linker 序列（常用 GS linker）
    MAX_SEQ_LEN = 512                                 # token 最大长度，超出截断（ESM2 上限 1022，留 buffer）

    # ── MLP head ──────────────────────────────────────────────────────────────
    HIDDEN_DIM = 256      # 隐藏层维度
    DROPOUT = 0.2         # Dropout 比例，防止小数据集过拟合

    # ── 训练 ───────────────────────────────────────────────────────────────────
    EPOCHS = 100          # 训练轮数（小数据集，100轮足够）
    LR = 1e-4             # 学习率
    BATCH_SIZE = 32       # 每批次样本对数量（pairwise 时每对算一个样本）
    MARGIN = 0.1          # Pairwise hinge loss 的 margin（差距低于此值才产生损失）
    WEIGHT_DECAY = 1e-4   # L2 正则化

    # ── 数据集过滤 ──────────────────────────────────────────────────────────────
    MAX_DATASET_SIZE = 5000   # 超过这个规模的数据集跳过（大多是预测值，不是实验 Kd）
    MIN_DATASET_SIZE = 10     # 少于这个数量不够做 train/val/test 划分

    # ── 路径 ───────────────────────────────────────────────────────────────────
    DATA_DIR = "data/binding"                  # FLAb binding 数据目录
    EMBED_CACHE_DIR = "cache/embeddings"       # embedding 缓存目录（避免重复计算）
    OUTPUT_DIR = "results/affinity_model"      # 模型和结果的输出目录

    # ── 训练集划分比例 ──────────────────────────────────────────────────────────
    TRAIN_RATIO = 0.8   # 80% 训练
    VAL_RATIO   = 0.1   # 10% 验证
    TEST_RATIO  = 0.1   # 10% 测试（用于最终 Spearman 评估）

    # ── 随机种子（保证结果可复现）──────────────────────────────────────────────
    SEED = 42


cfg = Config()

# 设置全局随机种子，保证每次跑结果一致
torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)

# 自动检测 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[设备] 使用: {device}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 数据加载
# ═══════════════════════════════════════════════════════════════════════════════

# 不同数据集使用不同的列名，这里统一映射成 heavy/light/fitness
# key = 原始列名，value = 标准列名
COLUMN_ALIASES = {
    "Ab_heavy_chain_seq": "heavy",    # AbRank 数据集
    "Ab_light_chain_seq": "light",    # AbRank 数据集
    "VHH_sequence":       "heavy",    # COGNANO 纳米抗体数据集
}


def load_one_dataset(filepath: str) -> pd.DataFrame | None:
    """
    从单个 CSV 或 CSV.ZIP 文件加载一个 benchmark 数据集。

    返回标准化后的 DataFrame（列：heavy, light[可选], fitness, sequence）
    如果数据不合格，返回 None。

    标准化步骤：
      1. 解压/读取文件
      2. 统一列名（处理不同数据集的命名差异）
      3. 检查必要列是否存在
      4. 删除缺失值行
      5. 检查数据集规模
      6. 拼接 heavy+linker+light 作为模型输入序列
    """

    # ── 步骤1：读取文件 ────────────────────────────────────────────────────────
    try:
        if filepath.endswith(".csv.zip"):
            # zip 文件，找到里面的 csv 并解压读取
            with zipfile.ZipFile(filepath) as z:
                csv_names = [n for n in z.namelist() if n.endswith(".csv")]
                if not csv_names:
                    return None
                with z.open(csv_names[0]) as f:
                    # low_memory=False 避免混合类型警告
                    df = pd.read_csv(f, low_memory=False)
        else:
            df = pd.read_csv(filepath, low_memory=False)
    except Exception as e:
        print(f"  [ERROR] 读取失败: {e}")
        return None

    # ── 步骤2：统一列名 ────────────────────────────────────────────────────────
    # 把各种别名映射成标准的 heavy/light/fitness
    df = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})

    # ── 步骤3：检查必要列 ──────────────────────────────────────────────────────
    if "heavy" not in df.columns:
        print(f"  [SKIP] 找不到重链序列列")
        return None
    if "fitness" not in df.columns:
        print(f"  [SKIP] 找不到 fitness 列")
        return None

    # ── 步骤4：删除缺失值 ──────────────────────────────────────────────────────
    # 至少要 heavy 和 fitness 存在；如果有 light 列，light 也不能是空
    required = ["heavy", "fitness"] + (["light"] if "light" in df.columns else [])
    df = df.dropna(subset=required).reset_index(drop=True)

    # ── 步骤5：检查数据集规模 ──────────────────────────────────────────────────
    if len(df) > cfg.MAX_DATASET_SIZE:
        print(f"  [SKIP] 数据集过大（{len(df):,} 条），跳过（大多为预测值）")
        return None
    if len(df) < cfg.MIN_DATASET_SIZE:
        print(f"  [SKIP] 数据量不足（{len(df)} 条）")
        return None

    # ── 步骤6：拼接输入序列 ────────────────────────────────────────────────────
    # ESM2 是单序列模型，将 heavy+linker+light 拼成一条序列输入
    # 只有重链（纳米抗体）的情况下，直接用重链序列
    if "light" in df.columns:
        # 双链抗体（Fv）：heavy + GS linker + light
        df["sequence"] = df["heavy"] + cfg.LINKER + df["light"]
    else:
        # 纳米抗体（VHH）：只有重链
        df["sequence"] = df["heavy"]

    # 截断过长序列（避免 OOM）
    df["sequence"] = df["sequence"].str[:cfg.MAX_SEQ_LEN * 3]  # 氨基酸长度约为 token 数的 1倍

    # 把 fitness 转成 float（防止有字符串型数字）
    df["fitness"] = pd.to_numeric(df["fitness"], errors="coerce")
    df = df.dropna(subset=["fitness"]).reset_index(drop=True)

    return df


def load_all_datasets(data_dir: str) -> dict[str, pd.DataFrame]:
    """
    加载 data_dir 下所有合法的 binding 数据集。

    返回：dict，key 是数据集名（文件名去掉扩展名），value 是标准化后的 DataFrame
    """
    datasets = {}

    # 遍历目录下所有文件
    all_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith(".csv") or f.endswith(".csv.zip")
    ])

    print(f"\n[数据] 扫描 {len(all_files)} 个文件...")

    for fname in all_files:
        # 从文件名提取数据集名（去掉 .csv 或 .csv.zip 后缀）
        dataset_name = fname.replace(".csv.zip", "").replace(".csv", "")
        fpath = os.path.join(data_dir, fname)

        print(f"\n  [{dataset_name}]")
        df = load_one_dataset(fpath)

        if df is not None:
            datasets[dataset_name] = df
            print(f"  → 加载成功：{len(df)} 条，有{'双链' if 'light' in df.columns else '单链'}序列")

    print(f"\n[数据] 成功加载 {len(datasets)} 个数据集")
    return datasets


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ESM2 Embedding 提取与缓存
# ═══════════════════════════════════════════════════════════════════════════════

# 全局变量：只加载一次模型，避免重复占用显存
_esm_tokenizer = None
_esm_model = None

def get_esm_model():
    """
    懒加载 ESM2 模型。第一次调用时加载，之后直接返回缓存的实例。
    使用 EsmModel（不是 EsmForMaskedLM），因为我们只需要 embedding，不需要 LM head。
    """
    global _esm_tokenizer, _esm_model
    if _esm_model is None:
        print(f"\n[模型] 加载 ESM2: {cfg.ESM_MODEL_NAME} ...")
        _esm_tokenizer = EsmTokenizer.from_pretrained(cfg.ESM_MODEL_NAME)
        # EsmModel 输出 hidden states，EsmForMaskedLM 输出 logits
        # 我们需要 hidden states 做 embedding，所以用 EsmModel
        _esm_model = EsmModel.from_pretrained(cfg.ESM_MODEL_NAME).to(device)
        _esm_model.eval()  # 冻结 BN/Dropout 层，进入推理模式
        print(f"[模型] ESM2 加载完毕，参数量: {sum(p.numel() for p in _esm_model.parameters()):,}")
    return _esm_tokenizer, _esm_model


def seq_to_cache_key(seq: str) -> str:
    """
    将序列转为哈希值，用作缓存文件名。
    直接用序列当文件名太长，用 MD5 哈希缩短。
    """
    return hashlib.md5(seq.encode()).hexdigest()


def embed_sequence(seq: str) -> np.ndarray:
    """
    用 ESM2 将单条氨基酸序列转为固定长度的 embedding 向量。

    实现步骤：
      1. tokenize 序列（氨基酸 → token id）
      2. ESM2 forward pass，得到每个位置的 hidden state（形状: [1, L, 1280]）
      3. 对所有位置做 mean pooling，得到一个 1280 维向量（忽略 [CLS] 和 [EOS]）
    """
    tokenizer, model = get_esm_model()

    # tokenize：将氨基酸序列转成模型能读的 token id
    # truncation=True 防止超长序列报错
    inputs = tokenizer(
        seq,
        return_tensors="pt",
        truncation=True,
        max_length=cfg.MAX_SEQ_LEN,
        padding=False  # 单条序列不需要 padding
    )

    # 把 token id 移到 GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        # forward pass，获取所有层的 hidden states
        outputs = model(**inputs)
        # last_hidden_state shape: [1, seq_len, embedding_dim]
        # 第0维是 batch size（=1），第1维是序列长度，第2维是 embedding 维度
        hidden = outputs.last_hidden_state  # [1, L, 1280]

    # ── Mean Pooling ──────────────────────────────────────────────────────────
    # ESM2 输出包含 [CLS] token（第0位）和 [EOS] token（最后一位）
    # 这两个是特殊标记，不代表真实氨基酸，去掉它们只保留中间部分
    seq_hidden = hidden[0, 1:-1, :]  # [L-2, 1280]，去掉首尾特殊 token

    # 对所有氨基酸位置取平均，得到一个代表整条序列的向量
    embedding = seq_hidden.mean(dim=0)  # [1280]

    # 转成 numpy array，方便后续存储和处理
    return embedding.cpu().numpy()


def get_or_compute_embedding(seq: str, cache_dir: str) -> np.ndarray:
    """
    获取序列的 embedding。优先从缓存读取，缓存不存在时计算并保存。

    缓存策略：
      - 用序列的 MD5 哈希作为文件名（避免文件名过长）
      - 保存为 .npy 文件（numpy 二进制格式，读写快）
    """
    # 计算缓存文件路径
    key = seq_to_cache_key(seq)
    cache_path = os.path.join(cache_dir, f"{key}.npy")

    if os.path.exists(cache_path):
        # 缓存命中：直接读取，不需要跑模型
        return np.load(cache_path)
    else:
        # 缓存未命中：调用 ESM2 计算
        emb = embed_sequence(seq)
        # 保存到缓存，供下次使用
        np.save(cache_path, emb)
        return emb


def embed_all_datasets(datasets: dict, cache_dir: str) -> dict:
    """
    对所有数据集中的所有序列提取 embedding 并缓存到磁盘。

    这一步是最耗时的（GPU 密集），但只需要跑一次。
    之后的训练直接从缓存读取 embedding，速度很快。

    返回：dict，key 是数据集名，value 是带有 "embedding" 列的 DataFrame
    """
    os.makedirs(cache_dir, exist_ok=True)

    # 收集所有唯一序列（跨数据集去重，相同序列只计算一次）
    all_sequences = set()
    for df in datasets.values():
        all_sequences.update(df["sequence"].tolist())

    print(f"\n[Embedding] 共 {len(all_sequences)} 条唯一序列，开始提取...")

    # 逐条计算或从缓存读取 embedding
    seq_to_emb = {}
    for i, seq in enumerate(all_sequences):
        if (i + 1) % 100 == 0:
            # 每处理100条打印一次进度
            print(f"  进度: {i+1}/{len(all_sequences)}")
        seq_to_emb[seq] = get_or_compute_embedding(seq, cache_dir)

    print(f"[Embedding] 完成，embedding 维度: {list(seq_to_emb.values())[0].shape}")

    # 把 embedding 填回每个数据集的 DataFrame
    embedded_datasets = {}
    for name, df in datasets.items():
        df = df.copy()
        # 将每条序列的 embedding 存为 numpy 数组（list of arrays）
        df["embedding"] = df["sequence"].map(seq_to_emb)
        embedded_datasets[name] = df

    return embedded_datasets


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PyTorch Dataset 和 DataLoader
# ═══════════════════════════════════════════════════════════════════════════════

class PairwiseRankingDataset(Dataset):
    """
    Pairwise Ranking Dataset：每个样本是一对序列（A, B），其中 A 的亲和力高于 B。

    训练目标：让模型给 A 打出比 B 更高的分数。

    为什么用 pairwise 而不是直接回归：
      - 直接回归 fitness 值会受不同数据集量纲差异的影响（有的是 Kd nM，有的是 -log(Kd)）
      - Pairwise 只需要知道"谁比谁好"，不需要绝对值可比较
      - 直接和 Spearman 评分对齐
    """

    def __init__(self, df: pd.DataFrame):
        """
        从 DataFrame 中构造所有合法的 (A, B) 对，其中 fitness_A > fitness_B。

        参数：
          df: 包含 embedding 和 fitness 列的 DataFrame
        """
        # 提取所有 embedding 和 fitness 值
        embeddings = np.stack(df["embedding"].values)  # [N, 1280]
        fitness    = df["fitness"].values               # [N]

        self.pairs = []  # 存储所有 (emb_A, emb_B) 对

        N = len(df)
        # 枚举所有有序对 (i, j)，其中 fitness[i] > fitness[j]
        # 这是 O(N²) 的，对小数据集（<5000）可以接受
        for i in range(N):
            for j in range(N):
                if i != j and fitness[i] > fitness[j]:
                    # i 比 j 亲和力更高，构成一个训练对
                    self.pairs.append((
                        embeddings[i].astype(np.float32),   # 更好的序列的 embedding
                        embeddings[j].astype(np.float32),   # 更差的序列的 embedding
                    ))

        print(f"  [Dataset] 构造了 {len(self.pairs)} 个训练对（来自 {N} 条序列）")

    def __len__(self):
        # 返回总样本数（总对数）
        return len(self.pairs)

    def __getitem__(self, idx):
        # 返回第 idx 个样本：(正样本 embedding, 负样本 embedding)
        emb_pos, emb_neg = self.pairs[idx]
        return torch.tensor(emb_pos), torch.tensor(emb_neg)


class ScoringDataset(Dataset):
    """
    用于推理（打分）的 Dataset：每个样本是单条序列的 embedding 和 fitness 标签。
    验证集和测试集使用这个 Dataset。
    """

    def __init__(self, df: pd.DataFrame):
        self.embeddings = np.stack(df["embedding"].values).astype(np.float32)
        self.fitness    = df["fitness"].values.astype(np.float32)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        # 返回 (embedding, fitness)
        return torch.tensor(self.embeddings[idx]), torch.tensor(self.fitness[idx])


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MLP 模型定义
# ═══════════════════════════════════════════════════════════════════════════════

class AffinityMLP(nn.Module):
    """
    轻量 MLP 回归头，接在 ESM2 embedding 后面，输出一个亲和力分数。

    架构：
      1280 → Linear → GELU → Dropout → 256 → Linear → 1

    为什么用 GELU 而不是 ReLU：
      - GELU 在 transformer 相关任务上普遍比 ReLU 效果好
      - ESM2 内部也使用 GELU

    为什么这么浅：
      - 数据集通常只有几十到几百条，深层网络容易过拟合
      - ESM2 已经提取了丰富的特征，MLP 只是做映射
    """

    def __init__(self, input_dim: int = cfg.ESM_EMBEDDING_DIM,
                 hidden_dim: int = cfg.HIDDEN_DIM,
                 dropout: float = cfg.DROPOUT):
        super().__init__()

        self.net = nn.Sequential(
            # 第一层：把高维 embedding 压缩到 hidden_dim
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),                          # 非线性激活
            nn.Dropout(dropout),                # 随机丢弃神经元，防止过拟合

            # 第二层：输出单个标量分数
            nn.Linear(hidden_dim, 1),
        )

        # 初始化权重：使用 Xavier 初始化，防止梯度消失/爆炸
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform 初始化线性层权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)  # bias 初始化为0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数：
          x: [batch_size, 1280] 的 embedding 矩阵

        返回：
          [batch_size, 1] 的亲和力分数（值越大代表亲和力越高）
        """
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. 损失函数
# ═══════════════════════════════════════════════════════════════════════════════

class PairwiseHingeLoss(nn.Module):
    """
    Pairwise Hinge Ranking Loss（成对铰链排序损失）。

    核心思想：
      对于每一对 (正样本 A, 负样本 B)（fitness_A > fitness_B），
      我们希望 score_A - score_B > margin（即 A 的得分比 B 高出至少 margin）。

      如果差距不够：loss = max(0, margin - (score_A - score_B))
      如果差距足够：loss = 0

    margin 的作用：
      - margin=0 时，只要 A > B 就不产生损失（容易学到平凡解）
      - margin>0 时，要求 A 比 B 高出一定程度，鼓励更有区分度的排序
    """

    def __init__(self, margin: float = cfg.MARGIN):
        super().__init__()
        self.margin = margin

    def forward(self, score_pos: torch.Tensor, score_neg: torch.Tensor) -> torch.Tensor:
        """
        参数：
          score_pos: [batch_size, 1] 正样本（高亲和力）的预测分数
          score_neg: [batch_size, 1] 负样本（低亲和力）的预测分数

        返回：
          标量 loss
        """
        # 计算分数差：希望 score_pos - score_neg > margin
        diff = score_pos - score_neg  # [batch_size, 1]，正数越大越好

        # Hinge loss：当差距小于 margin 时产生损失
        loss = torch.clamp(self.margin - diff, min=0.0)  # max(0, margin - diff)

        # 对 batch 内所有样本对取平均
        return loss.mean()


# ═══════════════════════════════════════════════════════════════════════════════
# 7. 评估函数
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_spearman(model: AffinityMLP, dataset: ScoringDataset) -> float:
    """
    在给定数据集上计算 Spearman 相关系数。

    步骤：
      1. 对每条序列用模型预测分数
      2. 把预测分数和真实 fitness 排序后计算相关性

    Spearman 范围 [-1, 1]，越接近 1 越好。
    """
    model.eval()  # 切换到评估模式（关闭 Dropout）

    all_scores   = []  # 模型预测的分数
    all_fitness  = []  # 真实亲和力标签

    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    with torch.no_grad():
        for emb, fit in loader:
            # 前向传播得到分数
            scores = model(emb.to(device)).squeeze(-1)  # [batch]
            all_scores.extend(scores.cpu().numpy().tolist())
            all_fitness.extend(fit.numpy().tolist())

    # 计算 Spearman 相关系数
    if len(set(all_fitness)) < 2:
        # 所有 fitness 值相同时无法计算相关性
        return float("nan")

    corr, pval = spearmanr(all_scores, all_fitness)
    return float(corr)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. 单个 Benchmark 的训练流程
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_benchmark(
    name: str,
    df: pd.DataFrame,
    output_dir: str
) -> dict:
    """
    对单个 benchmark（一个数据集文件）训练一个独立的 MLP head，并评估。

    策略：
      - 每个 benchmark 代表同一个抗原的不同抗体变体
      - 为每个 benchmark 训练一个专属的小模型（per-benchmark fine-tuning）
      - 这样模型可以专门学习该抗原系统的亲和力规律
      - 最终在 test set 上报告 Spearman

    参数：
      name:       数据集名称（用于日志和保存文件）
      df:         包含 embedding 和 fitness 的 DataFrame
      output_dir: 模型和结果的保存路径

    返回：
      包含 Spearman 等指标的 dict
    """
    print(f"\n{'─'*50}")
    print(f"[训练] {name}（{len(df)} 条序列）")

    # ── 划分 train/val/test ──────────────────────────────────────────────────
    # 第一次划分：80% train，20% temp
    # 使用分层划分（stratify）让三个集合的 fitness 分布均匀
    try:
        # 对 fitness 分桶，用于 stratified split
        # 如果 fitness 值太少，分桶会失败，退回到随机划分
        fitness_bins = pd.qcut(df["fitness"], q=min(5, len(df)//3), labels=False, duplicates="drop")
        train_df, temp_df = train_test_split(
            df, test_size=(1 - cfg.TRAIN_RATIO),
            stratify=fitness_bins, random_state=cfg.SEED
        )
        temp_bins = pd.qcut(temp_df["fitness"], q=min(5, len(temp_df)//2), labels=False, duplicates="drop")
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5,
            stratify=temp_bins, random_state=cfg.SEED
        )
    except Exception:
        # 分层划分失败（数据太少或 fitness 值不够多样），用随机划分
        train_df, temp_df = train_test_split(
            df, test_size=(1 - cfg.TRAIN_RATIO), random_state=cfg.SEED
        )
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=cfg.SEED)

    print(f"  划分: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # ── 构建 Dataset ─────────────────────────────────────────────────────────
    # 训练集用 pairwise dataset（构造成对样本，优化排序）
    train_dataset = PairwiseRankingDataset(train_df)
    # 验证集和测试集用普通 scoring dataset（逐条打分，计算 Spearman）
    val_dataset   = ScoringDataset(val_df)
    test_dataset  = ScoringDataset(test_df)

    if len(train_dataset) == 0:
        # 如果所有序列亲和力相同，无法构造正负对
        print(f"  [SKIP] 无法构造训练对（所有亲和力值相同）")
        return {"name": name, "spearman_test": float("nan"), "n": len(df)}

    # 构建 DataLoader，打乱训练顺序
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,      # 每个 epoch 打乱顺序，防止模型记住顺序
        drop_last=False    # 保留最后一个不完整的 batch
    )

    # ── 初始化模型、优化器、损失函数 ─────────────────────────────────────────
    # 每个 benchmark 用一个全新的模型（不共享参数）
    model = AffinityMLP().to(device)

    # AdamW：Adam + weight decay，比普通 Adam 更防过拟合
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY
    )

    # 余弦退火学习率：训练后期降低学习率，收敛更稳定
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS
    )

    loss_fn = PairwiseHingeLoss(margin=cfg.MARGIN)

    # ── 训练循环 ─────────────────────────────────────────────────────────────
    best_val_spearman = -float("inf")  # 记录最优验证集 Spearman
    best_model_state  = None           # 保存最优模型参数

    for epoch in range(cfg.EPOCHS):
        # 切换到训练模式（开启 Dropout）
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for emb_pos, emb_neg in train_loader:
            # 把数据移到 GPU
            emb_pos = emb_pos.to(device)
            emb_neg = emb_neg.to(device)

            # 清空上一步的梯度（PyTorch 默认累积梯度）
            optimizer.zero_grad()

            # 正向传播：分别计算正负样本的分数
            score_pos = model(emb_pos)  # [batch, 1]
            score_neg = model(emb_neg)  # [batch, 1]

            # 计算 pairwise ranking loss
            loss = loss_fn(score_pos, score_neg)

            # 反向传播：计算梯度
            loss.backward()

            # 梯度裁剪：防止梯度爆炸（对小数据集的 MLP 尤其有用）
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 更新参数
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        # 更新学习率
        scheduler.step()

        # 每10个 epoch 在验证集上评估一次
        if (epoch + 1) % 10 == 0:
            val_spearman = evaluate_spearman(model, val_dataset)
            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"  Epoch {epoch+1:3d}/{cfg.EPOCHS}  loss={avg_loss:.4f}  val_spearman={val_spearman:.4f}")

            # 保存验证集最优模型（early stopping 的简单版本）
            if not np.isnan(val_spearman) and val_spearman > best_val_spearman:
                best_val_spearman = val_spearman
                # 深拷贝模型参数（不是引用）
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

    # ── 加载最优模型，在测试集上评估 ─────────────────────────────────────────
    if best_model_state is not None:
        # 恢复验证集上表现最好的模型参数
        model.load_state_dict(best_model_state)
    test_spearman = evaluate_spearman(model, test_dataset)
    print(f"  [最终] test_spearman = {test_spearman:.4f}")

    # ── 保存模型 ──────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{name}.pt")
    torch.save({
        "model_state": best_model_state or model.state_dict(),
        "config": {
            "input_dim":  cfg.ESM_EMBEDDING_DIM,
            "hidden_dim": cfg.HIDDEN_DIM,
            "dropout":    cfg.DROPOUT,
        },
        "val_spearman":  best_val_spearman,
        "test_spearman": test_spearman,
    }, model_path)

    return {
        "name":            name,
        "n":               len(df),
        "n_train":         len(train_df),
        "n_test":          len(test_df),
        "val_spearman":    best_val_spearman,
        "spearman_test":   test_spearman,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 9. 主函数
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="FLAb 亲和力预测模型：embedding 提取 + MLP 训练"
    )
    parser.add_argument(
        "--mode", choices=["embed", "train", "all"], default="all",
        help="运行模式：embed=只提取 embedding，train=只训练，all=全流程"
    )
    parser.add_argument(
        "--data_dir", default=cfg.DATA_DIR,
        help=f"数据目录（默认 {cfg.DATA_DIR}）"
    )
    parser.add_argument(
        "--cache_dir", default=cfg.EMBED_CACHE_DIR,
        help=f"embedding 缓存目录（默认 {cfg.EMBED_CACHE_DIR}）"
    )
    parser.add_argument(
        "--output_dir", default=cfg.OUTPUT_DIR,
        help=f"结果输出目录（默认 {cfg.OUTPUT_DIR}）"
    )
    args = parser.parse_args()

    # ── 步骤1：加载数据集 ──────────────────────────────────────────────────────
    datasets = load_all_datasets(args.data_dir)
    if not datasets:
        print("[ERROR] 没有加载到任何数据集，请检查路径和数据格式")
        sys.exit(1)

    # ── 步骤2：提取 embedding（如果需要）────────────────────────────────────────
    if args.mode in ("embed", "all"):
        embedded_datasets = embed_all_datasets(datasets, args.cache_dir)
        # 把结果存到磁盘，供后续 train 步骤使用
        cache_pkl = os.path.join(args.cache_dir, "embedded_datasets.pkl")
        with open(cache_pkl, "wb") as f:
            pickle.dump(embedded_datasets, f)
        print(f"\n[Embedding] 已缓存到 {cache_pkl}")
    else:
        # train-only 模式：直接从磁盘读取之前缓存的 embedding
        cache_pkl = os.path.join(args.cache_dir, "embedded_datasets.pkl")
        if not os.path.exists(cache_pkl):
            print(f"[ERROR] 找不到 embedding 缓存 {cache_pkl}，请先运行 --mode embed")
            sys.exit(1)
        with open(cache_pkl, "rb") as f:
            embedded_datasets = pickle.load(f)
        print(f"[Embedding] 从缓存加载了 {len(embedded_datasets)} 个数据集")

    # ── 步骤3：训练（如果需要）────────────────────────────────────────────────
    if args.mode in ("train", "all"):
        all_results = []
        for name, df in embedded_datasets.items():
            result = train_one_benchmark(name, df, args.output_dir)
            all_results.append(result)

        # 汇总所有 benchmark 的 Spearman
        results_df = pd.DataFrame(all_results)
        summary_path = os.path.join(args.output_dir, "summary.csv")
        os.makedirs(args.output_dir, exist_ok=True)
        results_df.to_csv(summary_path, index=False)

        # 打印总结
        valid_spearman = results_df["spearman_test"].dropna()
        print(f"\n{'═'*50}")
        print(f"[总结] 在 {len(valid_spearman)} 个 benchmark 上完成训练")
        print(f"  平均 Spearman（test）: {valid_spearman.mean():.4f}")
        print(f"  中位 Spearman（test）: {valid_spearman.median():.4f}")
        print(f"  最高 Spearman（test）: {valid_spearman.max():.4f}")
        print(f"  结果保存至: {summary_path}")


if __name__ == "__main__":
    main()
