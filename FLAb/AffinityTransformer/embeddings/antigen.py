"""
embeddings/antigen.py — 抗原 embedding 缓存

这个模块负责把 antigen_registry 中的抗原上下文转成可训练特征。
当前实现重点是单序列蛋白抗原 ESM2 embedding 的缓存协议；MSA、配体和糖
分支先提供明确接口与 manifest 规范，避免后续临时散落脚本。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import os
from typing import Any

import numpy as np

from ..antigen_schema import (
    EMBEDDING_TYPES,
    PROTEIN_LIKE_TYPES,
    clean_text,
    normalize_antigen_sequence,
    sequence_hash,
)
from ..config import cfg, get_encoder_spec
from ..encoders.sequence import HuggingFaceSequenceEncoder


SINGLE_ESM2_DIM = 1280
MSA_ESM1B_DIM = 768
DEFAULT_LIGAND_DIM = 2048
DEFAULT_GLYCAN_DIM = 256


@dataclass
class AntigenEmbeddingManifest:
    """
    每个抗原 embedding cache 对应的 manifest。

    参数：
      antigen_id:            registry 中的 antigen_id
      embedding_type:        single_esm2 / msa_esm1b / ligand / glycan
      model_name:            产生该 embedding 的模型或方法
      embedding_dim:         向量维度
      source_sequence_hash:  输入序列 hash；非序列 embedding 可为空
      cache_path:            .npy 路径
      created_at:            ISO 时间戳
      notes:                 质检备注
    """

    antigen_id: str
    embedding_type: str
    model_name: str
    embedding_dim: int
    source_sequence_hash: str
    cache_path: str
    created_at: str
    notes: str = ""


def embedding_cache_paths(
    cache_root: str,
    antigen_id: str,
    embedding_type: str,
) -> tuple[str, str]:
    """
    返回 embedding 和 manifest 路径。

    缓存布局：
      cache_root/{embedding_type}/{antigen_id}.npy
      cache_root/{embedding_type}/{antigen_id}.manifest.json
    """
    if embedding_type not in EMBEDDING_TYPES:
        raise ValueError(f"未知 embedding_type={embedding_type!r}")
    base_dir = os.path.join(cache_root, embedding_type)
    embedding_path = os.path.join(base_dir, f"{antigen_id}.npy")
    manifest_path = os.path.join(base_dir, f"{antigen_id}.manifest.json")
    return embedding_path, manifest_path


def _record_get(record: dict[str, Any] | Any, key: str, default: Any = "") -> Any:
    """兼容 dict、pandas.Series 和 dataclass-like 对象的取值。"""
    if isinstance(record, dict):
        return record.get(key, default)
    if hasattr(record, "get"):
        return record.get(key, default)
    return getattr(record, key, default)


def save_embedding_with_manifest(
    embedding: np.ndarray,
    cache_root: str,
    antigen_id: str,
    embedding_type: str,
    model_name: str,
    source_sequence: str = "",
    notes: str = "",
) -> AntigenEmbeddingManifest:
    """
    保存 embedding，并写出 JSON manifest。

    返回：
      AntigenEmbeddingManifest。
    """
    embedding = np.asarray(embedding, dtype=np.float32)
    embedding_path, manifest_path = embedding_cache_paths(
        cache_root,
        antigen_id,
        embedding_type,
    )
    os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
    np.save(embedding_path, embedding)

    manifest = AntigenEmbeddingManifest(
        antigen_id=antigen_id,
        embedding_type=embedding_type,
        model_name=model_name,
        embedding_dim=int(embedding.shape[-1]),
        source_sequence_hash=sequence_hash(source_sequence) if source_sequence else "",
        cache_path=embedding_path,
        created_at=datetime.now(timezone.utc).isoformat(),
        notes=notes,
    )
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(asdict(manifest), handle, indent=2, ensure_ascii=False)
    return manifest


def read_embedding_manifest(
    cache_root: str,
    antigen_id: str,
    embedding_type: str,
) -> dict[str, Any]:
    """
    读取 manifest JSON。

    如果 manifest 不存在，会抛出 FileNotFoundError。
    """
    _, manifest_path = embedding_cache_paths(cache_root, antigen_id, embedding_type)
    with open(manifest_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_antigen_embedding_cache(
    cache_root: str,
    antigen_id: str,
    embedding_type: str,
    expected_dim: int | None = None,
) -> np.ndarray:
    """
    从 cache 读取抗原 embedding。

    参数：
      cache_root:    cache 根目录
      antigen_id:    registry antigen_id
      embedding_type: single_esm2 / msa_esm1b / ligand / glycan
      expected_dim:  可选，检查维度

    返回：
      np.ndarray，dtype=float32。
    """
    embedding_path, _ = embedding_cache_paths(cache_root, antigen_id, embedding_type)
    embedding = np.load(embedding_path).astype(np.float32)
    if expected_dim is not None and int(embedding.shape[-1]) != expected_dim:
        raise ValueError(
            f"{embedding_path} 维度为 {embedding.shape[-1]}，"
            f"期望 {expected_dim}"
        )
    return embedding


def has_cached_embedding(
    cache_root: str,
    antigen_id: str,
    embedding_type: str,
) -> bool:
    """检查某个 antigen embedding 是否已经缓存。"""
    embedding_path, manifest_path = embedding_cache_paths(
        cache_root,
        antigen_id,
        embedding_type,
    )
    return os.path.exists(embedding_path) and os.path.exists(manifest_path)


def embed_antigen_single(
    record: dict[str, Any] | Any,
    cache_root: str,
    encoder_alias: str = cfg.ANTIGEN_SINGLE_ENCODER,
    force: bool = False,
) -> AntigenEmbeddingManifest:
    """
    用 ESM2 计算单序列蛋白抗原 embedding。

    参数：
      record:        antigen_registry 的一行，需包含 antigen_id/antigen_type/sequence
      cache_root:    cache 根目录
      encoder_alias: ESM2 alias 或 Hugging Face model name；默认 esm2_650m
      force:         True 时即使 cache 存在也重新计算

    返回：
      AntigenEmbeddingManifest。

    实现：
      使用 v3 sequence_encoders.HuggingFaceSequenceEncoder。默认 ESM2-650M；
      如果后续要换 antigen single encoder，只需改 encoder_alias。
    """
    antigen_id = clean_text(_record_get(record, "antigen_id"))
    antigen_type = clean_text(_record_get(record, "antigen_type")).lower()
    sequence = normalize_antigen_sequence(_record_get(record, "antigen_sequence"))
    spec = get_encoder_spec(encoder_alias)
    model_name = spec.model_name

    if not antigen_id:
        raise ValueError("record 缺少 antigen_id")
    if antigen_type not in PROTEIN_LIKE_TYPES:
        raise ValueError(
            f"{antigen_id} antigen_type={antigen_type!r} 不是蛋白/肽类，"
            "不能使用 ESM2 single embedding"
        )
    if not sequence:
        raise ValueError(f"{antigen_id} 缺少 antigen_sequence")
    if has_cached_embedding(cache_root, antigen_id, "single_esm2") and not force:
        return AntigenEmbeddingManifest(
            **read_embedding_manifest(cache_root, antigen_id, "single_esm2")
        )

    encoder = HuggingFaceSequenceEncoder(spec)
    embedding = encoder.encode_sequences([sequence])[0]
    return save_embedding_with_manifest(
        embedding=embedding,
        cache_root=cache_root,
        antigen_id=antigen_id,
        embedding_type="single_esm2",
        model_name=model_name,
        source_sequence=sequence,
        notes=f"single-sequence antigen embedding; encoder_alias={spec.alias}",
    )


def embed_antigen_msa(
    record: dict[str, Any] | Any,
    cache_root: str,
    force: bool = False,
) -> AntigenEmbeddingManifest:
    """
    用 ESM-MSA-1b 计算 MSA-aware antigen embedding。

    record 需要提供：
      antigen_id
      msa_cache_path
    """
    antigen_id = clean_text(_record_get(record, "antigen_id"))
    if has_cached_embedding(cache_root, antigen_id, "msa_esm1b") and not force:
        return AntigenEmbeddingManifest(
            **read_embedding_manifest(cache_root, antigen_id, "msa_esm1b")
        )
    msa_path = clean_text(_record_get(record, "msa_cache_path"))
    if not msa_path:
        raise ValueError(f"{antigen_id} 缺少 msa_cache_path")

    from .msa import embed_msa_file

    return embed_msa_file(
        msa_path=msa_path,
        antigen_id=antigen_id,
        cache_root=cache_root,
        force=force,
    )


def embed_ligand(
    record: dict[str, Any] | Any,
    cache_root: str,
    force: bool = False,
) -> AntigenEmbeddingManifest:
    """
    小分子 embedding 接口占位。

    后续可接 RDKit Morgan fingerprint 或 ChemBERTa。当前不伪造 ligand 向量。
    """
    antigen_id = clean_text(_record_get(record, "antigen_id"))
    if has_cached_embedding(cache_root, antigen_id, "ligand") and not force:
        return AntigenEmbeddingManifest(
            **read_embedding_manifest(cache_root, antigen_id, "ligand")
        )
    raise NotImplementedError("ligand embedding 尚未接入 RDKit/ChemBERTa")


def zero_embedding(dim: int) -> np.ndarray:
    """
    返回显式 missing 用零向量。

    这个函数只用于 allow_missing=True 的消融实验；主实验不应默认吞掉缺失。
    """
    return np.zeros(int(dim), dtype=np.float32)
