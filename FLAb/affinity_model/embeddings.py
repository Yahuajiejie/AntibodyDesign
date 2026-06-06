"""
embeddings.py — ESM2 序列 Embedding 提取与磁盘缓存

为什么要缓存：
  - ESM2-650M 提取一条序列的 embedding 需要数秒（GPU）
  - 训练时会多次访问同一条序列（不同 epoch）
  - 缓存到磁盘后，后续直接读取，不再重复跑 GPU
  - embedding 计算和模型训练完全解耦，互不影响
"""

import os
import hashlib
import pickle
import numpy as np
import torch
from transformers import EsmTokenizer, EsmModel

from .config import cfg


# ── 全局模型缓存（模块级变量，只加载一次）─────────────────────────────────────────
# 避免在每次 embed_sequence() 调用时重复从磁盘加载模型
_tokenizer: EsmTokenizer | None = None
_model: EsmModel | None = None


def get_esm_model() -> tuple[EsmTokenizer, EsmModel]:
    """
    懒加载 ESM2 模型（第一次调用时加载，之后返回缓存实例）。

    使用 EsmModel 而非 EsmForMaskedLM：
      - EsmForMaskedLM 有一个 LM 输出头，用于预测被 mask 的 token
      - EsmModel 只输出 hidden states，更轻量，适合提取特征
    """
    global _tokenizer, _model

    if _model is None:
        print(f"\n[ESM2] 加载模型: {cfg.ESM_MODEL_NAME} ...")
        _tokenizer = EsmTokenizer.from_pretrained(cfg.ESM_MODEL_NAME)
        _model = EsmModel.from_pretrained(cfg.ESM_MODEL_NAME).to(cfg.DEVICE)
        _model.eval()  # 关闭 Dropout 和 BatchNorm 的训练模式
        n_params = sum(p.numel() for p in _model.parameters())
        print(f"[ESM2] 加载完毕，参数量: {n_params:,}，设备: {cfg.DEVICE}")

    return _tokenizer, _model


def _seq_hash(seq: str) -> str:
    """
    将氨基酸序列转为 MD5 哈希字符串，用作缓存文件名。
    直接用序列做文件名太长且可能含非法字符，哈希后固定为 32 位十六进制串。
    """
    return hashlib.md5(seq.encode()).hexdigest()


def embed_sequence(seq: str) -> np.ndarray:
    """
    用 ESM2 将单条氨基酸序列转为固定长度的 1280 维 embedding 向量。

    实现：
      1. Tokenize：氨基酸字符 → token id（ESM2 专属词表）
      2. Forward pass：ESM2 输出每个位置的 hidden state，形状 [1, L, 1280]
      3. Mean pooling：对所有真实氨基酸位置取平均，得到 [1280] 向量
         （忽略 [CLS] 和 [EOS] 这两个特殊 token）

    为什么用 mean pooling 而不是 [CLS] token：
      - [CLS] 的语义在 ESM2 中并不如 BERT 那样被明确训练为全局表示
      - Mean pooling 保留了所有位置的信息，实验中表现更稳定
    """
    tokenizer, model = get_esm_model()

    # Tokenize：将氨基酸序列编码成模型可读的 tensor
    # truncation=True：超过 MAX_SEQ_LEN 时自动截断，防止 OOM
    inputs = tokenizer(
        seq,
        return_tensors="pt",
        truncation=True,
        max_length=cfg.MAX_SEQ_LEN,
        padding=False,
    )
    # 将 input tensor 移到 GPU（如果可用）
    inputs = {k: v.to(cfg.DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        # 前向传播，获取所有层的输出
        outputs = model(**inputs)
        # last_hidden_state: [1, seq_len, 1280]
        # dim 0 = batch size (=1)，dim 1 = 序列长度，dim 2 = embedding 维度
        hidden = outputs.last_hidden_state

    # Mean pooling：去掉首尾的 [CLS] 和 [EOS] token，对中间的氨基酸位置取平均
    # hidden[0]     → [seq_len, 1280]，去掉 batch 维
    # [1:-1, :]     → [seq_len-2, 1280]，去掉首（[CLS]）和尾（[EOS]）
    # .mean(dim=0)  → [1280]，对位置维度取平均
    embedding = hidden[0, 1:-1, :].mean(dim=0)

    # 转为 numpy array，方便存储和后续处理
    return embedding.cpu().float().numpy()


def get_or_compute_embedding(seq: str, cache_dir: str) -> np.ndarray:
    """
    从缓存读取 embedding；如果缓存不存在，则计算后写入缓存。

    缓存格式：每条序列保存为独立的 .npy 文件（numpy 二进制，读写极快）
    """
    # 构造缓存文件路径
    cache_path = os.path.join(cache_dir, f"{_seq_hash(seq)}.npy")

    if os.path.exists(cache_path):
        # 缓存命中：直接读取，完全不需要 GPU
        return np.load(cache_path)

    # 缓存未命中：调用 ESM2 计算
    emb = embed_sequence(seq)
    # 保存到磁盘，供后续使用
    np.save(cache_path, emb)
    return emb


def _normalize_sequence_value(value) -> str:
    """
    把 DataFrame 里的序列值整理成稳定字符串。

    DataFrame 单元格可能是 NaN、空字符串，或者带有空格/换行的字符串。
    这里统一转成大写、去掉空白字符，避免同一条序列因为格式不同而重复缓存。
    """
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return ""
    return "".join(str(value).split()).upper()


def _is_valid_sequence_value(seq: str) -> bool:
    """
    判断整理后的序列是否真的存在。

    这里不严格检查每个氨基酸字符是否合法，因为 FLAb 中可能包含 X 等未知
    残基；只过滤明显的缺失值。
    """
    return bool(seq) and seq.lower() not in {"nan", "none", "null", "na"}


def _collect_unique_sequences(datasets: dict, columns: list[str]) -> set[str]:
    """
    从多个 DataFrame 的指定列中收集唯一序列。

    参数：
      datasets: load_all_datasets() 返回的 dict
      columns:  需要扫描的序列列名，例如 ["heavy", "light"]

    返回：
      set[str]，所有有效的唯一序列。
    """
    all_seqs: set[str] = set()
    for df in datasets.values():
        for col in columns:
            if col not in df.columns:
                continue
            for value in df[col].tolist():
                seq = _normalize_sequence_value(value)
                if _is_valid_sequence_value(seq):
                    all_seqs.add(seq)
    return all_seqs


def embed_all_datasets(datasets: dict, cache_dir: str) -> dict:
    """
    对所有数据集中的每条序列提取 embedding，写入磁盘缓存。

    优化：
      - 跨数据集去重：相同序列只计算一次（某些序列在多个 benchmark 中出现）
      - v2.1 默认分别提取 heavy/light embedding，新增
        "heavy_embedding"、"light_embedding"、"has_light" 列
      - scfv_mean 消融模式沿用 v1，新增 "embedding" 列

    这是训练前最耗时的步骤（GPU 密集），但只需要执行一次。
    """
    os.makedirs(cache_dir, exist_ok=True)

    if cfg.MODEL_FEATURE_MODE == "chain_concat":
        sequence_columns = ["heavy", "light"]
    elif cfg.MODEL_FEATURE_MODE == "scfv_mean":
        sequence_columns = ["sequence"]
    else:
        raise ValueError(
            "未知 MODEL_FEATURE_MODE="
            f"{cfg.MODEL_FEATURE_MODE!r}，可选 chain_concat / scfv_mean"
        )

    # 收集所有唯一序列，跨数据集去重。
    all_seqs = _collect_unique_sequences(datasets, sequence_columns)
    if not all_seqs:
        raise ValueError(
            f"没有找到可 embedding 的序列列：{sequence_columns}，请检查数据加载结果"
        )

    print(f"\n[Embedding] {len(all_seqs)} 条唯一序列，开始提取...")
    print(f"[Embedding] feature_mode={cfg.MODEL_FEATURE_MODE}")

    # 逐条提取（或从缓存读取）
    seq_to_emb: dict[str, np.ndarray] = {}
    for i, seq in enumerate(all_seqs):
        seq_to_emb[seq] = get_or_compute_embedding(seq, cache_dir)
        # 每 100 条打印一次进度
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(all_seqs)}")

    sample_emb = next(iter(seq_to_emb.values()))
    print(f"[Embedding] 完成，维度: {sample_emb.shape}")

    # 把 embedding 填回各数据集的 DataFrame
    embedded = {}
    zero_embedding = np.zeros(cfg.ESM_EMBEDDING_DIM, dtype=np.float32)
    for name, df in datasets.items():
        df = df.copy()

        if cfg.MODEL_FEATURE_MODE == "chain_concat":
            if "heavy" not in df.columns:
                raise ValueError(f"{name} 缺少 heavy 列，无法生成 chain_concat 特征")

            heavy_seqs = df["heavy"].map(_normalize_sequence_value)
            missing_heavy = [seq for seq in heavy_seqs if not _is_valid_sequence_value(seq)]
            if missing_heavy:
                raise ValueError(f"{name} 存在空 heavy 序列，无法生成 embedding")

            df["heavy_embedding"] = heavy_seqs.map(seq_to_emb)

            if "light" in df.columns:
                light_seqs = df["light"].map(_normalize_sequence_value)
                has_light = light_seqs.map(_is_valid_sequence_value)
                df["has_light"] = has_light.astype(bool)
                df["light_embedding"] = [
                    seq_to_emb[seq] if has else zero_embedding
                    for seq, has in zip(light_seqs, has_light)
                ]
            else:
                df["has_light"] = False
                df["light_embedding"] = [zero_embedding] * len(df)
        else:
            if "sequence" not in df.columns:
                raise ValueError(f"{name} 缺少 sequence 列，无法生成 scfv_mean 特征")
            seqs = df["sequence"].map(_normalize_sequence_value)
            df["embedding"] = seqs.map(seq_to_emb)

        embedded[name] = df

    # 将整个 embedded_datasets 额外序列化到磁盘，方便 train 阶段直接加载
    pkl_path = os.path.join(cache_dir, "embedded_datasets.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(embedded, f)
    print(f"[Embedding] 已序列化到 {pkl_path}")

    return embedded


def load_cached_datasets(cache_dir: str) -> dict:
    """
    从磁盘加载已缓存的 embedded_datasets（embed 阶段生成的 .pkl 文件）。
    train-only 模式使用此函数，无需重新计算 embedding。
    """
    pkl_path = os.path.join(cache_dir, "embedded_datasets.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(
            f"找不到 embedding 缓存 {pkl_path}，请先运行 --mode embed"
        )
    with open(pkl_path, "rb") as f:
        datasets = pickle.load(f)
    print(f"[Embedding] 从缓存加载 {len(datasets)} 个数据集")
    return datasets
