"""
pipeline.py — AffinityTransformer(v3) 预处理调度层

这个模块把 v3 的几个独立组件串起来，但不接入 AffinityMLP(v1/v2)：

  antigen_registry
    -> antigen single/MSA embedding cache
    -> antibody embedding cache
    -> v3 feature matrix

核心规则来自当前 v3 架构：
  1. 官方提供抗原序列：先算 antigen single embedding，再与 MSA embedding concat；
  2. 官方没有抗原序列：不伪造 single embedding，直接依赖 MSA embedding；
  3. 抗体 encoder 可选择 ESM/IgBert/IgT5/custom HF；
  4. 所有输出都写 cache，训练阶段只读 cache。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .antibody_embeddings import embed_antibody_dataframe
from .antigen_context_dataset import (
    build_antigen_context_feature_matrix,
    has_official_antigen_sequence,
)
from .antigen_embeddings import (
    AntigenEmbeddingManifest,
    embed_antigen_msa,
    embed_antigen_single,
    has_cached_embedding,
)
from .antigen_schema import clean_text
from .config import cfg


@dataclass
class AntigenEmbeddingPlanItem:
    """
    单个抗原的 embedding 计划。

    参数：
      antigen_id:               registry antigen_id
      compatible_group:         对应 group
      needs_single:             是否需要 single antigen embedding
      needs_msa:                是否需要 MSA embedding
      has_official_sequence:    官方是否提供了抗原序列
      reason:                   计划原因，便于质检
    """

    antigen_id: str
    compatible_group: str
    needs_single: bool
    needs_msa: bool
    has_official_sequence: bool
    reason: str


def plan_antigen_embeddings(
    registry: pd.DataFrame,
) -> list[AntigenEmbeddingPlanItem]:
    """
    根据 registry 生成 v3 抗原 embedding 计划。

    规则：
      - 官方有抗原序列：single + MSA；
      - 官方无抗原序列：MSA only。
    """
    plans: list[AntigenEmbeddingPlanItem] = []
    for _, row in registry.iterrows():
        antigen_id = clean_text(row.get("antigen_id", ""))
        group = clean_text(row.get("compatible_group", ""))
        if not antigen_id:
            raise ValueError(f"{group} 缺少 antigen_id")

        official_sequence = has_official_antigen_sequence(row)
        if official_sequence:
            plans.append(AntigenEmbeddingPlanItem(
                antigen_id=antigen_id,
                compatible_group=group,
                needs_single=True,
                needs_msa=True,
                has_official_sequence=True,
                reason="official antigen sequence available: use single + MSA",
            ))
        else:
            plans.append(AntigenEmbeddingPlanItem(
                antigen_id=antigen_id,
                compatible_group=group,
                needs_single=False,
                needs_msa=True,
                has_official_sequence=False,
                reason="no official antigen sequence: use MSA only",
            ))
    return plans


def embed_antigens_from_registry(
    registry: pd.DataFrame,
    cache_root: str = cfg.ANTIGEN_CACHE_DIR,
    force: bool = False,
    strict: bool = True,
) -> list[AntigenEmbeddingManifest]:
    """
    按 v3 规则为 registry 中的抗原生成 embedding cache。

    参数：
      registry:   antigen_registry DataFrame
      cache_root: antigen embedding cache 根目录
      force:      是否强制重算已有 cache
      strict:     True 时缺少必需 MSA/sequence 会报错；False 时跳过并记录

    返回：
      list[AntigenEmbeddingManifest]。
    """
    manifests: list[AntigenEmbeddingManifest] = []
    plan_by_group = {
        item.compatible_group: item
        for item in plan_antigen_embeddings(registry)
    }

    for _, row in registry.iterrows():
        group = clean_text(row.get("compatible_group", ""))
        plan = plan_by_group[group]

        try:
            if plan.needs_single:
                manifests.append(embed_antigen_single(
                    row,
                    cache_root=cache_root,
                    encoder_alias=cfg.ANTIGEN_SINGLE_ENCODER,
                    force=force,
                ))

            if plan.needs_msa:
                manifests.append(embed_antigen_msa(
                    row,
                    cache_root=cache_root,
                    force=force,
                ))
        except Exception:
            if strict:
                raise
    return manifests


def validate_required_antigen_cache(
    registry: pd.DataFrame,
    cache_root: str = cfg.ANTIGEN_CACHE_DIR,
) -> pd.DataFrame:
    """
    检查 v3 训练所需的 antigen cache 是否存在。

    返回：
      DataFrame，列为 compatible_group / antigen_id / missing_embedding / reason。
    """
    rows: list[dict[str, str]] = []
    plan_by_group = {
        item.compatible_group: item
        for item in plan_antigen_embeddings(registry)
    }

    for _, row in registry.iterrows():
        group = clean_text(row.get("compatible_group", ""))
        antigen_id = clean_text(row.get("antigen_id", ""))
        plan = plan_by_group[group]

        if plan.needs_single and not has_cached_embedding(
            cache_root,
            antigen_id,
            "single_esm2",
        ):
            rows.append({
                "compatible_group": group,
                "antigen_id": antigen_id,
                "missing_embedding": "single_esm2",
                "reason": plan.reason,
            })

        if plan.needs_msa and not has_cached_embedding(
            cache_root,
            antigen_id,
            "msa_esm1b",
        ):
            rows.append({
                "compatible_group": group,
                "antigen_id": antigen_id,
                "missing_embedding": "msa_esm1b",
                "reason": plan.reason,
            })

    return pd.DataFrame(rows, columns=[
        "compatible_group",
        "antigen_id",
        "missing_embedding",
        "reason",
    ])


def prepare_v3_dataframe(
    df: pd.DataFrame,
    antibody_cache_dir: str = cfg.ANTIBODY_CACHE_DIR,
    antibody_encoder: str = cfg.ANTIBODY_ENCODER,
    antibody_layout: str = cfg.ANTIBODY_ENCODER_LAYOUT,
    batch_size: int = 16,
) -> pd.DataFrame:
    """
    为一个样本表生成 antibody embedding 列。

    这个函数不处理 antigen cache，因为 antigen 是按 registry/group 级别缓存。
    """
    return embed_antibody_dataframe(
        df=df,
        cache_dir=antibody_cache_dir,
        encoder_alias=antibody_encoder,
        layout=antibody_layout,
        batch_size=batch_size,
    )


def build_v3_feature_matrix(
    df: pd.DataFrame,
    registry: pd.DataFrame,
    antigen_cache_dir: str = cfg.ANTIGEN_CACHE_DIR,
    antibody_layout: str = cfg.ANTIBODY_ENCODER_LAYOUT,
    allow_missing_antigen_context: bool = cfg.ALLOW_MISSING_ANTIGEN_CONTEXT,
) -> np.ndarray:
    """
    构建完整 v3 feature matrix。

    输出顺序：
      antibody, antigen_single_slot, antigen_msa, antigen_type_one_hot + flags

    如果官方没有抗原序列，antigen_single_slot 是显式零占位，并由 flags 标记。
    """
    return build_antigen_context_feature_matrix(
        df=df,
        registry=registry,
        cache_root=antigen_cache_dir,
        antibody_feature_mode=antibody_layout,
        use_single=True,
        use_msa=True,
        include_type_flags=cfg.INCLUDE_ANTIGEN_TYPE_FLAGS,
        allow_missing=allow_missing_antigen_context,
        context_policy=cfg.ANTIGEN_CONTEXT_POLICY,
    )

