"""
registry/workflow.py — antigen_registry 构建工作流

这个模块把 FLAb 训练 CSV 和外部任务资料串起来，输出 registry stage 的核心数据资产：
`antigen_registry.csv`。它只做可复现的数据整理，不计算 embedding，不训练模型。
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import pandas as pd

from .core import (
    build_antigen_registry,
    validate_antigen_registry,
    write_antigen_registry,
)
from ..antigen_schema import REGISTRY_COLUMNS, clean_text
from .sources import (
    build_external_antigen_registry,
    load_binding_tables,
    load_flab_metadata,
)


@dataclass
class RegistryBuildResult:
    """
    registry 构建结果。

    参数：
      registry:         合并后的 antigen_registry。
      issues:           validate_antigen_registry 生成的质检表。
      auxiliary_tables: 辅助表，例如 task_controls、proteinbase_targets。
    """

    registry: pd.DataFrame
    issues: pd.DataFrame
    auxiliary_tables: dict[str, pd.DataFrame]


def _deduplicate_compatible_groups(registry: pd.DataFrame) -> pd.DataFrame:
    """
    去除重复 compatible_group。

    参数：
      registry: REGISTRY_COLUMNS 形状的 DataFrame。

    返回：
      DataFrame。

    实现：
      保留第一次出现的 group，并在 notes 中记录被跳过的重复数。registry stage 的
      registry 必须满足“一组一抗原上下文”，否则后续 feature matrix 会歧义。
    """
    if registry.empty:
        return registry

    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    duplicate_counts: dict[str, int] = {}
    for row in registry.to_dict("records"):
        group = clean_text(row.get("compatible_group", ""))
        if group in seen:
            duplicate_counts[group] = duplicate_counts.get(group, 0) + 1
            continue
        seen.add(group)
        rows.append(row)

    out = pd.DataFrame(rows, columns=REGISTRY_COLUMNS)
    for group, count in duplicate_counts.items():
        mask = out["compatible_group"].astype(str) == group
        old_notes = out.loc[mask, "notes"].astype(str)
        out.loc[mask, "notes"] = old_notes + f"; skipped_duplicate_group_rows={count}"
    return out


def build_registry(
    datasets: dict[str, pd.DataFrame] | None = None,
    metadata: dict[str, dict] | None = None,
    group_col: str = "compatible_group",
    tasks_md: str | None = None,
    proteinbase_csv: str | None = None,
    andd_xlsx: str | None = None,
    sabdab_summary_tsv: str | None = None,
    include_external: bool = True,
) -> RegistryBuildResult:
    """
    构建 antigen_registry。

    参数：
      datasets:            FLAb binding DataFrame dict，可为空。
      metadata:            FLAb metadata dict，可为空。
      group_col:           compatible group 列名。
      tasks_md:            TASKS.md 路径。
      proteinbase_csv:     proteinbase CSV 路径。
      andd_xlsx:           ANDD.xlsx 路径。
      sabdab_summary_tsv:  SAbDab summary tsv 路径。
      include_external:    是否合并 TASKS/proteinbase/ANDD/SAbDab 来源。

    返回：
      RegistryBuildResult。

    改进思路：
      旧 v3 只从 FLAb 行内字段/metadata 建 registry。registry stage 把赛事官方文档
      和额外数据源纳入同一个 registry 构建流程，但仍保留 source/confidence
      字段，避免把人工补充和原始标签混在一起。
    """
    parts: list[pd.DataFrame] = []
    auxiliary: dict[str, pd.DataFrame] = {}

    if datasets:
        parts.append(build_antigen_registry(
            datasets=datasets,
            metadata=metadata,
            group_col=group_col,
        ))

    if include_external:
        external, external_auxiliary = build_external_antigen_registry(
            tasks_md=tasks_md,
            proteinbase_csv=proteinbase_csv,
            andd_xlsx=andd_xlsx,
            sabdab_summary_tsv=sabdab_summary_tsv,
        )
        if not external.empty:
            parts.append(external)
        auxiliary.update(external_auxiliary)

    if parts:
        registry = pd.concat(parts, ignore_index=True)
    else:
        registry = pd.DataFrame(columns=REGISTRY_COLUMNS)
    registry = _deduplicate_compatible_groups(registry[REGISTRY_COLUMNS])
    issues = validate_antigen_registry(registry, strict=False)
    return RegistryBuildResult(
        registry=registry,
        issues=issues,
        auxiliary_tables=auxiliary,
    )


def build_registry_from_paths(
    binding_dir: str | None = None,
    metadata_csv: str | None = None,
    binding_pattern: str = "*.csv*",
    max_binding_files: int | None = None,
    tasks_md: str | None = None,
    proteinbase_csv: str | None = None,
    andd_xlsx: str | None = None,
    sabdab_summary_tsv: str | None = None,
    include_external: bool = True,
) -> RegistryBuildResult:
    """
    从磁盘路径直接构建 antigen_registry。

    参数：
      binding_dir:         FLAb/data/binding 目录；为空则跳过 FLAb CSV。
      metadata_csv:        FLAb/data/flab_metadata.csv；为空则不使用 metadata。
      binding_pattern:     binding_dir 下的 glob pattern。
      max_binding_files:   可选，只读前 N 个 binding 文件，便于 debug。
      tasks_md/proteinbase_csv/andd_xlsx/sabdab_summary_tsv:
                           外部来源路径。
      include_external:    是否合并外部来源。

    返回：
      RegistryBuildResult。

    实现：
      该函数是 CLI 的主要入口。它会先读取 FLAb binding CSV，再读取 metadata，
      最后调用 build_registry 合并外部来源。
    """
    datasets = None
    metadata = None
    if binding_dir:
        datasets = load_binding_tables(
            binding_dir=binding_dir,
            pattern=binding_pattern,
            max_files=max_binding_files,
        )
    if metadata_csv:
        metadata = load_flab_metadata(metadata_csv)

    return build_registry(
        datasets=datasets,
        metadata=metadata,
        tasks_md=tasks_md,
        proteinbase_csv=proteinbase_csv,
        andd_xlsx=andd_xlsx,
        sabdab_summary_tsv=sabdab_summary_tsv,
        include_external=include_external,
    )


def write_registry_result(
    result: RegistryBuildResult,
    registry_path: str,
    issues_path: str | None = None,
    auxiliary_dir: str | None = None,
) -> dict[str, str]:
    """
    写出 registry 构建结果。

    参数：
      result:        build_registry 返回值。
      registry_path: antigen_registry.csv 输出路径。
      issues_path:   质检报告输出路径；为空则不写。
      auxiliary_dir: 辅助表输出目录；为空则不写。

    返回：
      dict，记录实际写出的文件路径。
    """
    written: dict[str, str] = {}
    write_antigen_registry(result.registry, registry_path)
    written["registry"] = registry_path

    if issues_path:
        os.makedirs(os.path.dirname(os.path.abspath(issues_path)), exist_ok=True)
        result.issues.to_csv(issues_path, index=False)
        written["issues"] = issues_path

    if auxiliary_dir:
        os.makedirs(auxiliary_dir, exist_ok=True)
        for name, table in result.auxiliary_tables.items():
            path = os.path.join(auxiliary_dir, f"{name}.csv")
            table.to_csv(path, index=False)
            written[name] = path

    return written
