"""
config.py — 全局超参数配置

所有可调参数集中在这里。修改实验设置只需改这一个文件。
"""

import torch


class Config:

    # ── ESM2 backbone ──────────────────────────────────────────────────────────
    # 使用 650M 版本，在性能和显存之间取较好平衡
    ESM_MODEL_NAME   = "facebook/esm2_t33_650M_UR50D"
    ESM_EMBEDDING_DIM = 1280          # 650M 对应的 hidden size
    LINKER           = "GGGGSGGGGSGGGGS"  # 拼接重链和轻链的 GS linker，模拟 scFv 结构
    MAX_SEQ_LEN      = 512            # tokenizer 最大长度，超出截断（ESM2 上限 1022）

    # ── MLP head ──────────────────────────────────────────────────────────────
    HIDDEN_DIM = 256    # 隐藏层维度（数据量小，不需要太宽）
    DROPOUT    = 0.2    # Dropout 比例，防止小数据集过拟合

    # ── 训练 ───────────────────────────────────────────────────────────────────
    EPOCHS       = 100    # 训练轮数
    LR           = 1e-4   # 学习率
    BATCH_SIZE   = 32     # 每批次样本对数
    MARGIN       = 0.1    # Pairwise hinge loss 的 margin
    WEIGHT_DECAY = 1e-4   # L2 正则化系数
    MAX_PAIRS    = 10000  # 每个 benchmark 最多使用的训练对数量
                          # 防止大数据集 O(N²) 爆炸（3000条序列→900万对→改为1万对）

    # ── 数据集过滤 ──────────────────────────────────────────────────────────────
    MAX_DATASET_SIZE = 5000  # 超过此行数的数据集跳过（多为预测值，非实验 Kd）
    MIN_DATASET_SIZE = 30    # 少于此行数无法有效划分 train/val/test
                              # 30条 → train≈24, val≈3, test≈3，最低保证 val/test 各有3条

    # ── 划分比例 ───────────────────────────────────────────────────────────────
    TRAIN_RATIO = 0.8   # 80% 训练集
    VAL_RATIO   = 0.1   # 10% 验证集
    TEST_RATIO  = 0.1   # 10% 测试集（最终 Spearman 评估）

    # ── 路径 ───────────────────────────────────────────────────────────────────
    DATA_DIR       = "data/binding"           # FLAb binding 数据目录
    EMBED_CACHE_DIR = "cache/embeddings"      # embedding 缓存目录
    OUTPUT_DIR     = "results/affinity_model" # 模型权重和结果输出目录

    # ── 随机种子 ───────────────────────────────────────────────────────────────
    SEED = 42

    # ── 设备（自动检测 GPU）────────────────────────────────────────────────────
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 全局单例，其他模块直接 from config import cfg 使用
cfg = Config()
