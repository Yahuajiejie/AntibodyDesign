"""
antigen_registry.py — v3 抗原登记表构建与质检

v3 的第一步不是直接训练模型，而是建立 antigen_registry。这个表负责把
compatible_group 映射到一个明确的抗原上下文：名字、类型、序列来源、
置信度、MSA cache 等。

本模块只新增独立 API，不会修改 v2.1 的数据加载或训练逻辑。
"""

from __future__ import annotations

from collections import Counter
import os
from typing import Any

import pandas as pd

from .antigen_schema import (
    ANTIGEN_TYPES,
    BOOL_COLUMNS,
    REGISTRY_COLUMNS,
    SEQUENCE_CONFIDENCES,
    SEQUENCE_SOURCES,
    AntigenRecord,
    clean_text,
    coerce_bool,
    flags_from_antigen_type,
    infer_antigen_type,
    normalize_antigen_sequence,
    ordered_registry_dict,
    stable_antigen_id,
)


ANTIGEN_NAME_COLUMNS = [
    "antigen_name",
    "Antigen_name",
    "Antigen_Name",
    "antigen",
    "Antigen",
    "Ag_Name",
    "Ag_name",
    "ag_name",
    "Ag_label",
    "target",
    "Target",
    "target_name",
    "Target_name",
    "Target_Name",
    "Target_Name(s)",
    "TargetName",
    "protein_name",
    "Protein_name",
]

ANTIGEN_SEQUENCE_COLUMNS = [
    "antigen_sequence",
    "Antigen_sequence",
    "Antigen_Sequence",
    "antigen_seq",
    "Antigen_seq",
    "Ag_Seq",
    "Ag_seq",
    "ag_seq",
    "Ag_sequence",
    "target_sequence",
    "Target_sequence",
    "Target_Sequence",
    "target_seq",
    "Target_seq",
    "Target_Seq",
    "Ag_Sequence",
]

METADATA_ANTIGEN_KEYS = [
    "antigen_name",
    "antigen",
    "target",
    "target_name",
    "Target",
    "publication_title",
    "key_words",
]


def _mode_text(series: pd.Series) -> tuple[str, int]:
    """
    从一列候选文本中取最常见的非空值。

    返回：
      (value, n_unique)。n_unique 用于 notes 记录“该组内是否出现多个抗原名”。
    """
    values = [clean_text(value) for value in series.tolist()]
    values = [value for value in values if value]
    if not values:
        return "", 0
    counts = Counter(values)
    return counts.most_common(1)[0][0], len(counts)


def _mode_sequence(series: pd.Series) -> tuple[str, int]:
    """从一列候选序列中取最常见的非空标准化序列。"""
    values = [normalize_antigen_sequence(value) for value in series.tolist()]
    values = [value for value in values if value]
    if not values:
        return "", 0
    counts = Counter(values)
    return counts.most_common(1)[0][0], len(counts)


def _first_available_column(columns: list[str], candidates: list[str]) -> str | None:
    """返回 candidates 中第一个存在于 columns 的列名。"""
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def _metadata_for_group(
    group_df: pd.DataFrame,
    dataset_name: str,
    metadata: dict[str, dict] | None,
) -> dict[str, Any]:
    """
    找到当前 group 对应的 metadata row。

    data_loader.load_metadata() 通常按 filename 建索引；因此优先使用
    group_df["source_file"]，再退回 dataset_name。
    """
    if not metadata:
        return {}
    source_file = ""
    if "source_file" in group_df.columns and len(group_df) > 0:
        source_file = clean_text(group_df["source_file"].iloc[0])
    return metadata.get(source_file) or metadata.get(dataset_name) or {}


def _metadata_antigen_name(row: dict[str, Any]) -> str:
    """
    从 metadata 中提取抗原名字候选。

    metadata 经常没有专门的 antigen 字段，所以这里只做保守提取：
    找到明确存在的 key，返回非空文本；不会从长标题里硬猜序列。
    """
    for key in METADATA_ANTIGEN_KEYS:
        value = clean_text(row.get(key, ""))
        if value:
            return value
    return ""


def _build_record_for_group(
    group_name: str,
    group_df: pd.DataFrame,
    dataset_name: str,
    metadata_row: dict[str, Any] | None,
) -> AntigenRecord:
    """
    根据一个 compatible_group 生成 AntigenRecord。

    输入：
      group_name:    compatible_group 名称
      group_df:      该组对应的行
      dataset_name:  外层 datasets dict 的 key
      metadata_row:  FLAb metadata 中对应的行，可为空

    返回：
      AntigenRecord。缺失抗原序列时不会伪造 embedding 信息。
    """
    notes: list[str] = []

    name_col = _first_available_column(list(group_df.columns), ANTIGEN_NAME_COLUMNS)
    if name_col is not None:
        antigen_name, n_names = _mode_text(group_df[name_col])
        name_source = f"csv:{name_col}"
        if n_names > 1:
            notes.append(f"group has {n_names} unique antigen names in {name_col}")
    else:
        antigen_name = _metadata_antigen_name(metadata_row or {})
        name_source = "metadata" if antigen_name else "missing"

    seq_col = _first_available_column(list(group_df.columns), ANTIGEN_SEQUENCE_COLUMNS)
    if seq_col is not None:
        antigen_sequence, n_sequences = _mode_sequence(group_df[seq_col])
        sequence_source = f"csv:{seq_col}" if antigen_sequence else "missing"
        sequence_confidence = "high" if antigen_sequence else "none"
        if n_sequences > 1:
            notes.append(f"group has {n_sequences} unique antigen sequences in {seq_col}")
    else:
        antigen_sequence = ""
        sequence_source = "missing"
        sequence_confidence = "none"

    antigen_type = infer_antigen_type(
        antigen_name=antigen_name,
        antigen_sequence=antigen_sequence,
    )
    flags = flags_from_antigen_type(antigen_type, antigen_sequence)
    source_file = ""
    if "source_file" in group_df.columns and len(group_df) > 0:
        source_file = clean_text(group_df["source_file"].iloc[0])

    antigen_id = stable_antigen_id(
        compatible_group=group_name,
        antigen_name=antigen_name,
        antigen_sequence=antigen_sequence,
    )

    return AntigenRecord(
        antigen_id=antigen_id,
        compatible_group=group_name,
        dataset=dataset_name,
        source_file=source_file,
        antigen_name=antigen_name,
        antigen_type=antigen_type,
        antigen_sequence=antigen_sequence,
        sequence_source=sequence_source,
        sequence_accession="",
        sequence_confidence=sequence_confidence,
        ligand_smiles="",
        glycan_info="",
        msa_source="",
        msa_cache_path="",
        notes="; ".join(notes + ([f"name_source={name_source}"] if name_source else [])),
        **flags,
    )


def build_antigen_registry(
    datasets: dict[str, pd.DataFrame],
    metadata: dict[str, dict] | None = None,
    group_col: str = "compatible_group",
) -> pd.DataFrame:
    """
    从已加载的数据集构建 antigen_registry 初稿。

    参数：
      datasets: data_loader.load_all_datasets() 返回的 dict，value 是 DataFrame
      metadata: 可选，data_loader.load_metadata() 返回的 dict
      group_col: 分组列名，默认 compatible_group

    返回：
      pandas.DataFrame，列顺序符合 REGISTRY_COLUMNS。

    注意：
      这一步只做“能从表里直接读到的信息”。没有序列的抗原不会自动联网补齐。
    """
    records: list[dict[str, Any]] = []

    for dataset_name, df in datasets.items():
        if df.empty:
            continue

        if group_col in df.columns:
            grouped = df.groupby(group_col, sort=True)
        else:
            df = df.copy()
            df[group_col] = dataset_name
            grouped = df.groupby(group_col, sort=True)

        for group_name, group_df in grouped:
            metadata_row = _metadata_for_group(group_df, dataset_name, metadata)
            record = _build_record_for_group(
                group_name=str(group_name),
                group_df=group_df,
                dataset_name=dataset_name,
                metadata_row=metadata_row,
            )
            records.append(record.to_dict())

    registry = pd.DataFrame(records, columns=REGISTRY_COLUMNS)
    return registry


def load_antigen_registry(path: str) -> pd.DataFrame:
    """
    读取 antigen_registry.csv，并补齐缺失列。

    参数：
      path: registry CSV 路径

    返回：
      pandas.DataFrame。布尔列会转成 bool，序列列会标准化。
    """
    registry = pd.read_csv(path, low_memory=False)
    for col in REGISTRY_COLUMNS:
        if col not in registry.columns:
            registry[col] = False if col in BOOL_COLUMNS else ""
    registry = registry[REGISTRY_COLUMNS].copy()
    for col in BOOL_COLUMNS:
        registry[col] = registry[col].map(coerce_bool)
    registry["antigen_sequence"] = registry["antigen_sequence"].map(
        normalize_antigen_sequence
    )
    return registry


def write_antigen_registry(registry: pd.DataFrame, path: str) -> None:
    """
    写出 antigen_registry.csv。

    参数：
      registry: 待写出的 registry DataFrame
      path:     输出路径

    实现：
      - 自动补齐列；
      - 按 REGISTRY_COLUMNS 排序；
      - 创建父目录；
      - 不自动覆盖质量问题，质量问题请先调用 validate_antigen_registry。
    """
    rows = [ordered_registry_dict(row) for row in registry.to_dict("records")]
    out = pd.DataFrame(rows, columns=REGISTRY_COLUMNS)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    out.to_csv(path, index=False)


def validate_antigen_registry(
    registry: pd.DataFrame,
    strict: bool = False,
) -> pd.DataFrame:
    """
    质检 antigen_registry。

    参数：
      registry: antigen registry DataFrame
      strict:   True 时，存在 error 级问题会抛出 ValueError

    返回：
      pandas.DataFrame，列为 severity / compatible_group / field / message。
    """
    issues: list[dict[str, str]] = []

    def add(severity: str, group: str, field: str, message: str) -> None:
        issues.append({
            "severity": severity,
            "compatible_group": clean_text(group),
            "field": field,
            "message": message,
        })

    missing_cols = [col for col in REGISTRY_COLUMNS if col not in registry.columns]
    for col in missing_cols:
        add("error", "", col, "missing required registry column")

    if missing_cols:
        report = pd.DataFrame(issues)
        if strict:
            raise ValueError(report.to_string(index=False))
        return report

    duplicated_groups = registry["compatible_group"][
        registry["compatible_group"].duplicated(keep=False)
    ]
    for group in sorted(set(duplicated_groups.astype(str))):
        add("error", group, "compatible_group", "compatible_group appears multiple times")

    duplicated_ids = registry["antigen_id"][
        registry["antigen_id"].duplicated(keep=False)
    ]
    for antigen_id in sorted(set(duplicated_ids.astype(str))):
        add("error", "", "antigen_id", f"duplicated antigen_id={antigen_id}")

    for _, row in registry.iterrows():
        group = clean_text(row.get("compatible_group", ""))
        antigen_type = clean_text(row.get("antigen_type", "")).lower()
        sequence = normalize_antigen_sequence(row.get("antigen_sequence", ""))
        confidence = clean_text(row.get("sequence_confidence", "")).lower()
        source = clean_text(row.get("sequence_source", "")).lower()
        has_sequence_flag = coerce_bool(row.get("has_antigen_sequence", False))

        if not group:
            add("error", group, "compatible_group", "empty compatible_group")
        if not clean_text(row.get("antigen_id", "")):
            add("error", group, "antigen_id", "empty antigen_id")
        if antigen_type not in ANTIGEN_TYPES:
            add("error", group, "antigen_type", f"unknown antigen_type={antigen_type!r}")
        if confidence not in SEQUENCE_CONFIDENCES:
            add(
                "warning",
                group,
                "sequence_confidence",
                f"unexpected sequence_confidence={confidence!r}",
            )
        if source and source.split(":", 1)[0] not in SEQUENCE_SOURCES:
            add("warning", group, "sequence_source", f"unexpected source={source!r}")
        if bool(sequence) != has_sequence_flag:
            add(
                "error",
                group,
                "has_antigen_sequence",
                "flag does not match antigen_sequence emptiness",
            )
        if antigen_type in {"small_molecule", "carbohydrate"} and sequence:
            add(
                "error",
                group,
                "antigen_sequence",
                "non-protein antigen should not store protein sequence",
            )
        if antigen_type in {"protein", "glycoprotein", "peptide"} and not sequence:
            add(
                "warning",
                group,
                "antigen_sequence",
                "protein-like antigen has no sequence; embedding cannot be computed",
            )
        if source == "":
            add("error", group, "sequence_source", "sequence_source should not be empty")

    report = pd.DataFrame(issues, columns=[
        "severity",
        "compatible_group",
        "field",
        "message",
    ])
    if strict and not report.empty and (report["severity"] == "error").any():
        raise ValueError(report.to_string(index=False))
    return report


def merge_registry_updates(
    base_registry: pd.DataFrame,
    updates: pd.DataFrame,
    key: str = "compatible_group",
    prefer_updates: bool = True,
) -> pd.DataFrame:
    """
    将人工补充的 registry 更新合并回基础 registry。

    参数：
      base_registry:   自动构建的 registry
      updates:         人工补充表，至少包含 key 列
      key:             默认按 compatible_group 对齐
      prefer_updates:  True 时 updates 中非空字段覆盖 base

    返回：
      pandas.DataFrame，列顺序符合 REGISTRY_COLUMNS。
    """
    base = load_like_registry(base_registry).set_index(key, drop=False)
    patch = load_like_registry(updates).set_index(key, drop=False)

    for group, row in patch.iterrows():
        if group not in base.index:
            base.loc[group] = row
            continue
        if not prefer_updates:
            continue
        for col in REGISTRY_COLUMNS:
            value = row.get(col, "")
            if col in BOOL_COLUMNS:
                base.at[group, col] = coerce_bool(value)
            elif clean_text(value):
                base.at[group, col] = value

    merged = base.reset_index(drop=True)
    return merged[REGISTRY_COLUMNS]


def load_like_registry(data: pd.DataFrame | list[dict[str, Any]]) -> pd.DataFrame:
    """
    把 DataFrame 或 list[dict] 整理成 registry 形状。

    这个函数用于单元测试和人工 patch 表，不读写磁盘。
    """
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    else:
        frame = pd.DataFrame(data)

    for col in REGISTRY_COLUMNS:
        if col not in frame.columns:
            frame[col] = False if col in BOOL_COLUMNS else ""
    for col in BOOL_COLUMNS:
        frame[col] = frame[col].map(coerce_bool)
    frame["antigen_sequence"] = frame["antigen_sequence"].map(
        normalize_antigen_sequence
    )
    return frame[REGISTRY_COLUMNS]

