"""
config.py — 全局超参数配置

所有可调参数集中在这里。修改实验设置只需改这一个文件。
当前版本的核心约束是：训练一个跨数据集共享参数的通用亲和力排序模型，
不再为每个 benchmark 单独训练任务头。
"""

try:
    import torch
except ModuleNotFoundError:
    # 允许在没有 PyTorch 的轻量环境中运行 data_loader 做数据质检。
    # 真正训练/embedding 时仍然必须安装 torch；model/trainer 会直接依赖它。
    torch = None


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
    EPOCHS          = 100    # 训练轮数
    LR              = 1e-4   # 学习率
    BATCH_SIZE      = 64     # pairwise loss 下表示样本对数；MSE 下表示单条序列数
    EVAL_BATCH_SIZE = 512    # 验证/测试推理 batch size
    EVAL_EVERY      = 5      # 每隔多少个 epoch 在验证集上评估一次
    MARGIN          = 0.1    # Pairwise hinge loss 的 margin
    WEIGHT_DECAY    = 1e-4   # L2 正则化系数

    # Pairwise 数据量是 O(N²)，必须按“可比较组”限流。
    # 这里的可比较组默认是单个数据集：同一抗原、同一测量方法、同一标签方向。
    MAX_PAIRS_PER_GROUP = 10000
    MIN_LABEL_DIFF      = 0.0    # 构造 pair 时要求 label_pos - label_neg > 该阈值

    # ── 数据集过滤 ──────────────────────────────────────────────────────────────
    MAX_DATASET_SIZE = 50000  # 超过此行数的数据集跳过，避免 embedding 阶段失控
    MIN_GROUP_SIZE   = 5      # 少于 5 条的数据集不用于训练/评估，Spearman 统计意义太弱

    # 亲和力主模型默认只使用真实 Kd 类数据。
    # predicted Kd、bind/no bind、IC50/EC50/ADCC 等会在 data_loader 中跳过。
    ALLOWED_ASSAY_FAMILIES = {"kd"}

    # ── 划分比例 ───────────────────────────────────────────────────────────────
    TRAIN_RATIO = 0.8   # 80% 数据集组用于训练
    VAL_RATIO   = 0.1   # 10% 数据集组用于验证
    TEST_RATIO  = 0.1   # 10% 数据集组用于测试

    # Group split 表示按 compatible_group 整组划分，避免同一 benchmark
    # 的标签同时出现在训练集和测试集里。
    GROUP_COL        = "compatible_group"
    RANK_LABEL_COL   = "label"      # 排序用标签：方向已统一，越大亲和力越强
    MSE_LABEL_COL    = "label_z"    # MSE 用组内 z-score，避免不同量纲支配训练
    SPLIT_STRATEGY   = "group"

    # ── 路径 ───────────────────────────────────────────────────────────────────
    DATA_DIR        = "data/binding"          # FLAb binding 数据目录
    METADATA_PATH   = "data/flab_metadata.csv"
    EMBED_CACHE_DIR = "cache/embeddings"      # embedding 缓存目录
    OUTPUT_DIR     = "results/affinity_model" # 模型权重和结果输出目录

    # ── 随机种子 ───────────────────────────────────────────────────────────────
    SEED = 42

    # ── 设备（自动检测 GPU）────────────────────────────────────────────────────
    DEVICE = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch is not None else "cpu"
    )


# 全局单例，其他模块直接 from config import cfg 使用
cfg = Config()
