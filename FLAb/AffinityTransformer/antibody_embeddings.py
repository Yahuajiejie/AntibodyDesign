"""
antibody_embeddings.py — v3 抗体 encoder 与 embedding cache

v3 不绑定 ESM2。抗体侧 encoder 可选择：
  - ESM2：heavy/light 分别编码后拼接；
  - IgBert：抗体预训练 BERT，通常用空格分隔氨基酸；
  - IgT5：paired antibody encoder，可把 heavy/light 作为一条 paired 输入。
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .antigen_schema import clean_text, normalize_antigen_sequence
from .config import cfg, get_encoder_spec
from .sequence_encoders import (
    HuggingFaceSequenceEncoder,
    get_or_compute_sequence_embeddings,
    load_cached_sequence_embedding,
    paired_sequence_hash,
    save_sequence_embedding,
)


def _valid_sequence(value) -> bool:
    """判断 DataFrame 单元格是否是有效序列。"""
    return bool(normalize_antigen_sequence(value))


def _zero(dim: int) -> np.ndarray:
    """返回零向量，用于缺失 light 链显式占位。"""
    return np.zeros(int(dim), dtype=np.float32)


def embed_antibody_dataframe(
    df: pd.DataFrame,
    cache_dir: str = cfg.ANTIBODY_CACHE_DIR,
    encoder_alias: str = cfg.ANTIBODY_ENCODER,
    layout: str = cfg.ANTIBODY_ENCODER_LAYOUT,
    batch_size: int = 16,
) -> pd.DataFrame:
    """
    为抗体 DataFrame 添加 v3 antibody embedding 列。

    参数：
      df:            需要含 heavy，light 可选
      cache_dir:     embedding cache 目录
      encoder_alias: ESM2/IgBert/IgT5 alias 或 Hugging Face model name
      layout:        separate_chains / paired_chains
      batch_size:    encoder 推理 batch size

    返回：
      DataFrame copy。不会原地修改输入。

    输出列：
      separate_chains:
        heavy_embedding, light_embedding, has_light, antibody_encoder

      paired_chains:
        antibody_embedding, has_light, antibody_encoder
    """
    if "heavy" not in df.columns:
        raise ValueError("v3 antibody embedding 需要 heavy 列")

    out = df.copy()
    spec = get_encoder_spec(encoder_alias)
    encoder = HuggingFaceSequenceEncoder(spec)
    os.makedirs(cache_dir, exist_ok=True)

    heavy = out["heavy"].map(normalize_antigen_sequence)
    if not heavy.map(bool).all():
        raise ValueError("存在空 heavy 序列，无法生成 antibody embedding")

    if "light" in out.columns:
        light = out["light"].map(normalize_antigen_sequence)
        has_light = light.map(bool)
    else:
        light = pd.Series([""] * len(out), index=out.index)
        has_light = pd.Series([False] * len(out), index=out.index)

    out["has_light"] = has_light.astype(bool)
    out["antibody_encoder"] = spec.alias
    out["antibody_encoder_model"] = spec.model_name

    if layout == "separate_chains":
        heavy_map = get_or_compute_sequence_embeddings(
            heavy.tolist(),
            encoder=encoder,
            cache_dir=cache_dir,
            batch_size=batch_size,
        )
        light_map = get_or_compute_sequence_embeddings(
            [seq for seq in light.tolist() if seq],
            encoder=encoder,
            cache_dir=cache_dir,
            batch_size=batch_size,
        )

        encoder.load()
        assert encoder.hidden_size is not None
        zero = _zero(encoder.hidden_size)
        out["heavy_embedding"] = heavy.map(heavy_map)
        out["light_embedding"] = [
            light_map[seq] if present else zero
            for seq, present in zip(light.tolist(), has_light.tolist())
        ]
        return out

    if layout == "paired_chains":
        texts_to_compute: list[tuple[str, str, str]] = []
        embeddings: list[np.ndarray | None] = []
        for h_seq, l_seq in zip(heavy.tolist(), light.tolist()):
            key = paired_sequence_hash(h_seq, l_seq, spec.model_name)
            cached = load_cached_sequence_embedding(cache_dir, key)
            embeddings.append(cached)
            if cached is None:
                texts_to_compute.append((key, h_seq, l_seq))

        if texts_to_compute:
            computed = encoder.encode_paired_antibodies(
                [item[1] for item in texts_to_compute],
                [item[2] for item in texts_to_compute],
                batch_size=batch_size,
            )
            computed_by_key = {}
            for (key, h_seq, l_seq), emb in zip(texts_to_compute, computed):
                save_sequence_embedding(
                    emb,
                    cache_dir=cache_dir,
                    cache_key=key,
                    encoder=encoder,
                    source=f"heavy={h_seq};light={l_seq}",
                )
                computed_by_key[key] = emb
            embeddings = [
                emb if emb is not None
                else computed_by_key[
                    paired_sequence_hash(h_seq, l_seq, spec.model_name)
                ]
                for emb, h_seq, l_seq in zip(embeddings, heavy.tolist(), light.tolist())
            ]

        out["antibody_embedding"] = embeddings
        return out

    raise ValueError("layout 必须是 separate_chains 或 paired_chains")


def build_antibody_feature_matrix(
    df: pd.DataFrame,
    layout: str = cfg.ANTIBODY_ENCODER_LAYOUT,
) -> np.ndarray:
    """
    从 v3 embedding DataFrame 构建抗体侧矩阵。

    separate_chains 返回 [heavy, light]；
    paired_chains 返回 [antibody_embedding]。
    """
    if layout == "separate_chains":
        for col in ["heavy_embedding", "light_embedding"]:
            if col not in df.columns:
                raise ValueError(f"缺少 {col}，请先运行 embed_antibody_dataframe")
        heavy = np.stack(df["heavy_embedding"].values).astype(np.float32)
        light = np.stack(df["light_embedding"].values).astype(np.float32)
        return np.concatenate([heavy, light], axis=1).astype(np.float32)

    if layout == "paired_chains":
        if "antibody_embedding" not in df.columns:
            raise ValueError("缺少 antibody_embedding，请先运行 paired_chains embedding")
        return np.stack(df["antibody_embedding"].values).astype(np.float32)

    raise ValueError("layout 必须是 separate_chains 或 paired_chains")

