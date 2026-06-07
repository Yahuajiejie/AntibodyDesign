"""
msa_embeddings.py — v3 ESM-MSA-1b antigen embedding

同源搜索和 MSA 构建是离线预处理；本模块只读取已有 A3M/FASTA-like MSA，
调用 ESM-MSA-1b 生成 query antigen embedding，并写入 cache。
"""

from __future__ import annotations

import os

import numpy as np

from .antigen_embeddings import (
    MSA_ESM1B_DIM,
    AntigenEmbeddingManifest,
    has_cached_embedding,
    read_embedding_manifest,
    save_embedding_with_manifest,
)
from .antigen_schema import clean_text, normalize_antigen_sequence
from .config import cfg
from .msa_builder import read_a3m, sample_msa_depth, strip_a3m_insertions


_msa_model = None
_msa_alphabet = None


def get_msa_model(device: str | None = None):
    """
    懒加载 ESM-MSA-1b。

    依赖：
      pip/conda 环境中需要安装 fair-esm 的 `esm` 包。
    """
    global _msa_model, _msa_alphabet
    if _msa_model is not None:
        return _msa_model, _msa_alphabet, device

    import torch
    import esm

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _msa_model, _msa_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    _msa_model = _msa_model.to(device)
    _msa_model.eval()
    return _msa_model, _msa_alphabet, device


def _pool_query_embedding(representations, tokens, alphabet) -> np.ndarray:
    """
    对 MSA 中第一条 query 序列做 mean pooling。

    tokens 形状通常是 [batch=1, msa_depth, seq_len_with_special]；
    representations 形状通常是 [1, msa_depth, seq_len, 768]。
    """
    import torch

    query_tokens = tokens[0, 0]
    query_hidden = representations[0, 0]

    invalid_ids = {
        getattr(alphabet, "padding_idx", None),
        getattr(alphabet, "cls_idx", None),
        getattr(alphabet, "eos_idx", None),
        getattr(alphabet, "gap_idx", None),
    }
    invalid_ids = {idx for idx in invalid_ids if idx is not None}
    valid_mask = torch.ones_like(query_tokens, dtype=torch.bool)
    for token_id in invalid_ids:
        valid_mask &= query_tokens != token_id

    valid_hidden = query_hidden[valid_mask]
    if valid_hidden.numel() == 0:
        raise ValueError("MSA query 没有可 pooling 的有效 token")
    return valid_hidden.mean(dim=0).detach().cpu().float().numpy()


def embed_msa_file(
    msa_path: str,
    antigen_id: str,
    cache_root: str = cfg.ANTIGEN_CACHE_DIR,
    force: bool = False,
    max_depth: int = 128,
    max_length: int = 1024,
    device: str | None = None,
) -> AntigenEmbeddingManifest:
    """
    从 A3M/FASTA-like MSA 生成 antigen_msa_embedding。

    参数：
      msa_path:   MSA 文件路径，第一条必须是 query antigen
      antigen_id: antigen_registry 中的 ID
      cache_root: cache 根目录
      force:      是否强制重算
      max_depth:  最多使用多少条 MSA 序列
      max_length: query 超过该长度时报错，避免显存失控
      device:     cuda/cpu

    返回：
      AntigenEmbeddingManifest。
    """
    antigen_id = clean_text(antigen_id)
    if not antigen_id:
        raise ValueError("缺少 antigen_id")
    if has_cached_embedding(cache_root, antigen_id, "msa_esm1b") and not force:
        return AntigenEmbeddingManifest(
            **read_embedding_manifest(cache_root, antigen_id, "msa_esm1b")
        )

    records = read_a3m(msa_path)
    if not records:
        raise ValueError(f"MSA 文件为空: {msa_path}")
    records = sample_msa_depth(records, max_depth=max_depth, keep_first=True)
    query_sequence = strip_a3m_insertions(records[0].sequence)
    if not query_sequence:
        raise ValueError("MSA 第一条 query 序列为空")
    if len(query_sequence) > max_length:
        raise ValueError(
            f"query antigen 长度 {len(query_sequence)} > max_length={max_length}"
        )

    model, alphabet, device = get_msa_model(device=device)
    batch_converter = alphabet.get_batch_converter()
    msa_data = [(
        antigen_id,
        [(record.header, record.sequence) for record in records],
    )]
    _, _, tokens = batch_converter(msa_data)
    tokens = tokens.to(device)

    import torch

    with torch.no_grad():
        output = model(tokens, repr_layers=[12], return_contacts=False)
    embedding = _pool_query_embedding(output["representations"][12], tokens, alphabet)
    if int(embedding.shape[-1]) != MSA_ESM1B_DIM:
        raise ValueError(
            f"ESM-MSA embedding dim={embedding.shape[-1]}，期望 {MSA_ESM1B_DIM}"
        )

    return save_embedding_with_manifest(
        embedding=embedding,
        cache_root=cache_root,
        antigen_id=antigen_id,
        embedding_type="msa_esm1b",
        model_name=cfg.ANTIGEN_MSA_MODEL_NAME,
        source_sequence=normalize_antigen_sequence(query_sequence),
        notes=f"msa_path={msa_path};depth={len(records)}",
    )
