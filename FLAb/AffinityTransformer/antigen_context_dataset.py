"""
antigen_context_dataset.py — v3 抗体-抗原上下文特征矩阵

该模块不定义 PyTorch Dataset，也不接入 v2.1 trainer。它只负责把已有抗体
embedding 与 antigen_registry/cache 中的抗原上下文拼接成 numpy feature matrix。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .antigen_embeddings import (
    MSA_ESM1B_DIM,
    SINGLE_ESM2_DIM,
    has_cached_embedding,
    load_antigen_embedding_cache,
    zero_embedding,
)
from .antigen_schema import ANTIGEN_TYPES, clean_text, coerce_bool
from .config import cfg


ANTIGEN_TYPE_ORDER = [
    "protein",
    "glycoprotein",
    "peptide",
    "small_molecule",
    "carbohydrate",
    "unknown",
]

FLAG_COLUMNS = [
    "has_antigen_sequence",
    "has_official_antigen_sequence",
    "has_single_embedding",
    "has_msa_embedding",
    "uses_single_plus_msa_policy",
    "uses_msa_only_policy",
    "is_protein",
    "is_glycoprotein",
    "is_peptide",
    "is_small_molecule",
    "is_carbohydrate",
    "sequence_confidence_high",
    "sequence_confidence_medium",
    "sequence_confidence_low",
]

OFFICIAL_SEQUENCE_SOURCES = {
    "csv",
    "tasks",
    "andd",
    "proteinbase",
    "sabdab",
    "pdb",
}


def _stack_embedding_column(df: pd.DataFrame, column: str) -> np.ndarray:
    """
    将 DataFrame 中存储 np.ndarray 的列堆叠成矩阵。

    这里复制 v2.1 dataset.py 的基础逻辑，但不导入旧 Dataset 模块，
    避免新 v3 工具在轻量环境中强依赖 torch。
    """
    if column not in df.columns:
        raise ValueError(f"缺少 embedding 列: {column}")
    return np.stack(df[column].values).astype(np.float32)


def build_antibody_feature_matrix(
    df: pd.DataFrame,
    feature_mode: str = "chain_concat",
) -> np.ndarray:
    """
    构建抗体侧特征矩阵。

    参数：
      df:           已经包含 antibody embedding 的 DataFrame
      feature_mode: chain_concat / separate_chains / paired_chains / scfv_mean

    返回：
      np.ndarray。
    """
    if feature_mode in {"chain_concat", "separate_chains"}:
        heavy = _stack_embedding_column(df, "heavy_embedding")
        light = _stack_embedding_column(df, "light_embedding")
        return np.concatenate([heavy, light], axis=1).astype(np.float32)
    if feature_mode == "paired_chains":
        return _stack_embedding_column(df, "antibody_embedding")
    if feature_mode == "scfv_mean":
        return _stack_embedding_column(df, "embedding")
    raise ValueError(f"未知 feature_mode={feature_mode!r}")


def antigen_type_one_hot(antigen_type: str) -> np.ndarray:
    """
    将 antigen_type 转为 one-hot。

    unknown 或非法类型都会落到 unknown 位置。
    """
    antigen_type = clean_text(antigen_type).lower()
    if antigen_type not in ANTIGEN_TYPES:
        antigen_type = "unknown"
    vector = np.zeros(len(ANTIGEN_TYPE_ORDER), dtype=np.float32)
    vector[ANTIGEN_TYPE_ORDER.index(antigen_type)] = 1.0
    return vector


def sequence_source_prefix(sequence_source: str) -> str:
    """
    取 sequence_source 的前缀。

    例：
      csv:Ag_Seq -> csv
      tasks      -> tasks
    """
    return clean_text(sequence_source).lower().split(":", 1)[0]


def has_official_antigen_sequence(registry_row: pd.Series | dict[str, Any]) -> bool:
    """
    判断 registry 行是否有“官方提供”的抗原序列。

    这里的官方包括比赛/FLAb/ANDD/proteinbase/SAbDab/PDB 等直接给出的序列。
    UniProt/manual 补序列不算官方原始提供，后续可作为单独消融。
    """
    get = registry_row.get
    if not coerce_bool(get("has_antigen_sequence", False)):
        return False
    source = sequence_source_prefix(get("sequence_source", ""))
    confidence = clean_text(get("sequence_confidence", "")).lower()
    return source in OFFICIAL_SEQUENCE_SOURCES and confidence in {"high", "medium"}


def build_antigen_flags(
    registry_row: pd.Series | dict[str, Any],
    has_single_embedding_value: bool,
    has_msa_embedding_value: bool,
    official_sequence_value: bool,
    msa_only_policy_value: bool,
) -> np.ndarray:
    """
    构建抗原上下文 flag 向量。

    flags 的作用是告诉模型哪些上下文是真的，哪些是缺失补零。
    """
    get = registry_row.get
    confidence = clean_text(get("sequence_confidence", "")).lower()
    values = {
        "has_antigen_sequence": coerce_bool(get("has_antigen_sequence", False)),
        "has_official_antigen_sequence": bool(official_sequence_value),
        "has_single_embedding": bool(has_single_embedding_value),
        "has_msa_embedding": bool(has_msa_embedding_value),
        "uses_single_plus_msa_policy": bool(official_sequence_value)
        and bool(has_single_embedding_value)
        and bool(has_msa_embedding_value),
        "uses_msa_only_policy": bool(msa_only_policy_value),
        "is_protein": coerce_bool(get("is_protein", False)),
        "is_glycoprotein": coerce_bool(get("is_glycoprotein", False)),
        "is_peptide": coerce_bool(get("is_peptide", False)),
        "is_small_molecule": coerce_bool(get("is_small_molecule", False)),
        "is_carbohydrate": coerce_bool(get("is_carbohydrate", False)),
        "sequence_confidence_high": confidence == "high",
        "sequence_confidence_medium": confidence == "medium",
        "sequence_confidence_low": confidence == "low",
    }
    return np.array([float(values[col]) for col in FLAG_COLUMNS], dtype=np.float32)


def _registry_by_group(registry: pd.DataFrame) -> dict[str, pd.Series]:
    """把 registry 转为 compatible_group -> row 映射，并检查重复。"""
    if "compatible_group" not in registry.columns:
        raise ValueError("registry 缺少 compatible_group 列")
    duplicated = registry["compatible_group"][
        registry["compatible_group"].duplicated(keep=False)
    ]
    if len(duplicated) > 0:
        groups = sorted(set(duplicated.astype(str)))
        raise ValueError(f"registry 中 compatible_group 重复: {groups[:10]}")
    return {
        str(row["compatible_group"]): row
        for _, row in registry.iterrows()
    }


def _load_or_zero(
    cache_root: str,
    antigen_id: str,
    embedding_type: str,
    expected_dim: int,
    allow_missing: bool,
) -> tuple[np.ndarray, bool]:
    """
    读取 embedding；缺失时根据 allow_missing 决定报错或返回零向量。

    返回：
      (embedding, exists)。
    """
    if has_cached_embedding(cache_root, antigen_id, embedding_type):
        return (
            load_antigen_embedding_cache(
                cache_root,
                antigen_id,
                embedding_type,
                expected_dim=expected_dim,
            ),
            True,
        )
    if allow_missing:
        return zero_embedding(expected_dim), False
    raise FileNotFoundError(
        f"缺少 {embedding_type} cache: antigen_id={antigen_id}, cache_root={cache_root}"
    )


def build_antigen_context_matrix(
    df: pd.DataFrame,
    registry: pd.DataFrame,
    cache_root: str,
    group_col: str = "compatible_group",
    use_single: bool = True,
    use_msa: bool = True,
    include_type_flags: bool = True,
    allow_missing: bool = False,
    context_policy: str = cfg.ANTIGEN_CONTEXT_POLICY,
) -> np.ndarray:
    """
    为 df 中每一行构建 antigen context 矩阵。

    参数：
      df:                 带 compatible_group 的样本表
      registry:           antigen_registry DataFrame
      cache_root:          antigen embedding cache 根目录
      group_col:           df 中的 group 列
      use_single:          是否保留 single antigen slot [1280]
      use_msa:             是否拼接 msa_esm1b embedding [768]
      include_type_flags:  是否拼接 antigen_type one-hot 和 flags
      allow_missing:       缺失 cache 时是否用零向量，并通过 flag 标记
      context_policy:      official_sequence_then_msa / manual

    返回：
      np.ndarray，形状 [len(df), antigen_context_dim]。

    official_sequence_then_msa 策略：
      - 官方提供抗原序列：读取 single_esm2，再读取 msa_esm1b；
      - 官方没有抗原序列：single slot 显式补零，只读取 msa_esm1b；
      - 默认 allow_missing=False，因此缺 MSA 会直接报错。
    """
    if group_col not in df.columns:
        raise ValueError(f"df 缺少 {group_col!r} 列")

    registry_map = _registry_by_group(registry)
    rows: list[np.ndarray] = []

    for group in df[group_col].astype(str).tolist():
        if group not in registry_map:
            raise KeyError(f"registry 中找不到 compatible_group={group!r}")
        record = registry_map[group]
        antigen_id = clean_text(record.get("antigen_id", ""))
        if not antigen_id:
            raise ValueError(f"{group} registry 行缺少 antigen_id")

        parts: list[np.ndarray] = []
        has_single = False
        has_msa = False
        official_sequence = has_official_antigen_sequence(record)
        msa_only_policy = False

        if use_single:
            if context_policy == "official_sequence_then_msa":
                if official_sequence:
                    single, has_single = _load_or_zero(
                        cache_root,
                        antigen_id,
                        "single_esm2",
                        SINGLE_ESM2_DIM,
                        allow_missing,
                    )
                else:
                    # 固定维度占位，不代表存在真实 single embedding。
                    single = zero_embedding(SINGLE_ESM2_DIM)
                    has_single = False
                    msa_only_policy = True
            elif context_policy == "manual":
                single, has_single = _load_or_zero(
                    cache_root,
                    antigen_id,
                    "single_esm2",
                    SINGLE_ESM2_DIM,
                    allow_missing,
                )
            else:
                raise ValueError(f"未知 context_policy={context_policy!r}")
            parts.append(single)

        if use_msa:
            msa, has_msa = _load_or_zero(
                cache_root,
                antigen_id,
                "msa_esm1b",
                MSA_ESM1B_DIM,
                allow_missing,
            )
            parts.append(msa)

        if include_type_flags:
            parts.append(antigen_type_one_hot(record.get("antigen_type", "unknown")))
            parts.append(build_antigen_flags(
                record,
                has_single_embedding_value=has_single,
                has_msa_embedding_value=has_msa,
                official_sequence_value=official_sequence,
                msa_only_policy_value=msa_only_policy,
            ))

        if not parts:
            raise ValueError("至少需要一种 antigen context 特征")
        rows.append(np.concatenate(parts).astype(np.float32))

    return np.stack(rows).astype(np.float32)


def build_antigen_context_feature_matrix(
    df: pd.DataFrame,
    registry: pd.DataFrame,
    cache_root: str,
    antibody_feature_mode: str = cfg.ANTIBODY_ENCODER_LAYOUT,
    group_col: str = "compatible_group",
    use_single: bool = True,
    use_msa: bool = True,
    include_type_flags: bool = True,
    allow_missing: bool = False,
    context_policy: str = cfg.ANTIGEN_CONTEXT_POLICY,
) -> np.ndarray:
    """
    拼接完整 v3 输入特征。

    输出：
      [antibody_features, antigen_context_features]。
    """
    antibody = build_antibody_feature_matrix(df, feature_mode=antibody_feature_mode)
    antigen = build_antigen_context_matrix(
        df=df,
        registry=registry,
        cache_root=cache_root,
        group_col=group_col,
        use_single=use_single,
        use_msa=use_msa,
        include_type_flags=include_type_flags,
        allow_missing=allow_missing,
        context_policy=context_policy,
    )
    return np.concatenate([antibody, antigen], axis=1).astype(np.float32)


def antigen_context_dim(
    use_single: bool = True,
    use_msa: bool = True,
    include_type_flags: bool = True,
) -> int:
    """返回当前 antigen context 配置对应的维度。"""
    dim = 0
    if use_single:
        dim += SINGLE_ESM2_DIM
    if use_msa:
        dim += MSA_ESM1B_DIM
    if include_type_flags:
        dim += len(ANTIGEN_TYPE_ORDER) + len(FLAG_COLUMNS)
    return dim
