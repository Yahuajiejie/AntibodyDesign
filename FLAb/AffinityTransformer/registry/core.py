"""
registry/core.py - 抗原登记表构建与质检。

这个文件只做一件事：给每个 compatible_group 建一张“抗原身份证”。

为什么要有这张表：
  - 训练数据里每个 compatible_group 是一组可以互相比较亲和力的样本；
  - v3 想加入抗原 embedding，所以必须知道这一组样本对应哪个抗原；
  - 原始 CSV 的列名不统一，有的叫 antigen，有的叫 target，有的只在 metadata
    里写了名字，所以这里集中做一次整理。

antigen_registry 的每一行表示：
  一个 compatible_group -> 一个抗原名字/类型/序列来源/置信度/MSA cache 信息。
"""

from __future__ import annotations

from collections import Counter
import os
from typing import Any

import pandas as pd

from ..antigen_schema import (
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
    clean_registry_enum,
    stable_antigen_id,
)


# 原始 CSV 里“抗原名字”可能使用的列名。
#
# 举例：
#   有的表叫 antigen_name，有的叫 target，有的叫 Protein_name。
#   这些列名表达的意思类似：这一组抗体要结合的对象是谁。
#
# 代码不会要求所有列都存在，而是从上到下找“第一个存在的列”。
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

# 原始 CSV 里“抗原氨基酸序列”可能使用的列名。
#
# 注意：
#   这里找的是序列，不是抗原名字。
#   如果只有名字，比如 "Nipah G protein"，还不能直接拿去做 ESM embedding；
#   只有拿到类似 "MST..." 这样的氨基酸序列，才算有直接可用的蛋白序列。
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

# 如果 CSV 本身没有抗原名字，就去 metadata 里尝试找这些字段。
#
# metadata 的信息通常比 CSV 间接：它可能写了 target，也可能只在论文标题或关键词
# 里出现抗原名字。因此 metadata 只能作为“名字候选来源”，不能当成高置信序列来源。
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
    从一列文本里取“出现次数最多的非空值”。

    参数：
      series:
        pandas 的一列数据。通常是某个 compatible_group 内的 antigen/target 列。

    返回：
      (value, n_unique)
        value:
          出现次数最多的文本。
        n_unique:
          非空文本一共有多少种。

    为什么取众数：
      同一个 compatible_group 理论上应该对应同一个抗原。
      但原始表可能每行都重复写了一遍抗原名字，也可能少数行有脏值。
      取最常见值比随便取第一行稳一点；n_unique 用来记录是否存在不一致。
    """
    values = [clean_text(value) for value in series.tolist()]
    values = [value for value in values if value]
    if not values:
        return "", 0
    counts = Counter(values)
    return counts.most_common(1)[0][0], len(counts)


def _mode_sequence(series: pd.Series) -> tuple[str, int]:
    """
    从一列候选序列中取“出现次数最多的非空标准化序列”。

    参数：
      series:
        pandas 的一列数据。通常是某个 compatible_group 内的 antigen_sequence 列。

    返回：
      (sequence, n_unique)
        sequence:
          标准化后的氨基酸序列。标准化会去掉空格、统一大小写等。
        n_unique:
          这一组里非空序列一共有多少种。

    为什么要标准化：
      "ACD EF" 和 "acdef" 本质上是同一条序列；先标准化再计数，可以减少
      格式差异造成的假冲突。
    """
    values = [normalize_antigen_sequence(value) for value in series.tolist()]
    values = [value for value in values if value]
    if not values:
        return "", 0
    counts = Counter(values)
    return counts.most_common(1)[0][0], len(counts)


def _first_available_column(columns: list[str], candidates: list[str]) -> str | None:
    """
    在一堆候选列名里，找当前表真正拥有的第一个列名。

    参数：
      columns:
        当前 CSV/DataFrame 的实际列名列表。
      candidates:
        我们预先整理好的候选列名列表，例如 ANTIGEN_NAME_COLUMNS。

    返回：
      找到则返回列名字符串；找不到则返回 None。

    实现思路：
      先把实际列名转成 set，查找更快；然后按 candidates 的顺序依次检查。
      candidates 的顺序代表优先级，所以越明确的列名应该放越前面。
    """
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
    找到当前 compatible_group 对应的 metadata 行。

    参数：
      group_df:
        一个 compatible_group 对应的所有样本行。
      dataset_name:
        外层 datasets 字典的 key，通常来自文件名或数据集名。
      metadata:
        由 data_loader.load_metadata() 读出的 metadata 字典；可以为空。

    返回：
      一个 dict。如果没找到对应 metadata，就返回空 dict。

    实现思路：
      1. 如果 group_df 里有 source_file，优先用 source_file 去 metadata 里查；
      2. 如果查不到，再用 dataset_name 去查；
      3. 两个都查不到，就返回空 dict。

    为什么这样查：
      metadata 常常是按原始文件名建索引的，而 dataset_name 有时是整理后的名字。
      source_file 更接近原始文件名，所以优先级更高。
    """
    if not metadata:
        return {}
    source_file = ""
    if "source_file" in group_df.columns and len(group_df) > 0:
        source_file = clean_text(group_df["source_file"].iloc[0])
    return metadata.get(source_file) or metadata.get(dataset_name) or {}


def _metadata_antigen_name(row: dict[str, Any]) -> str:
    """
    从 metadata 里提取一个抗原名字候选。

    参数：
      row:
        某个数据集对应的一行 metadata，已经被整理成 dict。

    返回：
      一个抗原名字字符串；如果找不到就返回空字符串。

    实现思路：
      按 METADATA_ANTIGEN_KEYS 的顺序找字段，找到第一个非空文本就返回。

    重要限制：
      这里最多只能找“名字”，不会从论文标题里硬猜序列。
      比如 publication_title 里出现 "SARS-CoV-2 spike"，
      这只能说明抗原可能叫 spike，不能说明我们已经有 spike 的氨基酸序列。
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

    总体流程：
      1. 先从 CSV 列里找抗原名字；
      2. CSV 没有名字时，再从 metadata 里找名字；
      3. 再从 CSV 列里找抗原序列；
      4. 根据名字和序列推断抗原类型；
      5. 生成稳定 antigen_id；
      6. 把所有信息打包成 AntigenRecord。

    为什么不自动联网补序列：
      同一个抗原名字可能对应不同物种、不同片段、不同突变体。
      自动补错序列会比没有序列更危险，所以这里只登记“已有证据”。
    """
    notes: list[str] = []

    # name_col 是当前表里最像“抗原名字”的列。
    # 如果找到了，就在这个 compatible_group 内取最常见的名字。
    name_col = _first_available_column(list(group_df.columns), ANTIGEN_NAME_COLUMNS)
    if name_col is not None:
        antigen_name, n_names = _mode_text(group_df[name_col])
        name_source = f"csv:{name_col}"
        if n_names > 1:
            notes.append(f"group has {n_names} unique antigen names in {name_col}")
    else:
        # CSV 没有抗原名字时，metadata 只作为备选。
        # name_source 会写进 notes，方便之后人工质检这个名字从哪里来。
        antigen_name = _metadata_antigen_name(metadata_row or {})
        name_source = "metadata" if antigen_name else "missing"

    # seq_col 是当前表里最像“抗原氨基酸序列”的列。
    # 只有拿到真实序列时，sequence_confidence 才会设为 high。
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

    # 这里其实有问题，得确认仅遍历第一个真实存在的列

    # antigen_type 会影响后续怎么做 embedding：
    #   protein/glycoprotein/peptide 才适合用蛋白语言模型；
    #   small_molecule/carbohydrate 需要另一路特征，不能硬塞成蛋白序列。
    antigen_type = infer_antigen_type(
        antigen_name=antigen_name,
        antigen_sequence=antigen_sequence,
    )
    # flags 是若干布尔列，例如 has_antigen_sequence、is_protein_like。
    # 这些列让后续 pipeline 不用每次重新判断字符串。
    flags = flags_from_antigen_type(antigen_type, antigen_sequence)

    source_file = ""
    if "source_file" in group_df.columns and len(group_df) > 0:
        source_file = clean_text(group_df["source_file"].iloc[0])

    # antigen_id 是稳定 ID：同样的 group/name/sequence 会得到同样的 ID。
    # 这样后续缓存 embedding 或 MSA 文件时，不会因为 DataFrame 顺序变化而换名字。
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

    实现思路：
      datasets 里每个 DataFrame 可能包含多个 compatible_group。
      这里会逐个 group 拆开，每个 group 生成一行 AntigenRecord。
      如果某个 DataFrame 没有 compatible_group 列，就把整个数据集当成一个 group。
    """
    records: list[dict[str, Any]] = []

    for dataset_name, df in datasets.items():
        if df.empty:
            continue

        # 正常情况下，我们已经在数据处理阶段建立了 compatible_group。
        # 如果没有这列，就退回到“一个文件 = 一个 group”，避免函数直接崩掉。
        if group_col in df.columns:
            grouped = df.groupby(group_col, sort=True)
        else:
            df = df.copy()
            df[group_col] = dataset_name
            grouped = df.groupby(group_col, sort=True)

        for group_name, group_df in grouped:
            # metadata 是按数据集/源文件查的，不是按每一行查的。
            # 所以先为当前 group 找到对应 metadata，再交给 _build_record_for_group。
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

    为什么要补齐列：
      人工编辑 registry 时可能漏列；补齐后，后续代码可以稳定依赖
      REGISTRY_COLUMNS，不需要到处写“这一列是否存在”的判断。
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

    检查重点：
      1. 必需列是否完整；
      2. compatible_group 和 antigen_id 是否重复；
      3. 抗原类型、序列来源、置信度是否是允许值；
      4. has_antigen_sequence 这类布尔标记是否和真实序列一致；
      5. 非蛋白抗原是否错误地塞了蛋白序列。
    """
    issues: list[dict[str, str]] = []

    # 小工具：把一个质检问题追加到 issues。
    # severity 用 error/warning 区分“必须修”和“建议人工看一眼”。
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

    # 如果列都缺了，后面很多逐行检查会连字段都取不到。
    # 所以这里提前返回，避免制造一堆无意义报错。
    if missing_cols:
        report = pd.DataFrame(issues)
        if strict:
            raise ValueError(report.to_string(index=False))
        return report

    # 一个 compatible_group 应该只对应一行 registry。
    # 如果重复，后续按 group merge 特征时会出现一对多，结果不可控。
    duplicated_groups = registry["compatible_group"][
        registry["compatible_group"].duplicated(keep=False)
    ]
    for group in sorted(set(duplicated_groups.astype(str))):
        add("error", group, "compatible_group", "compatible_group appears multiple times")

    # antigen_id 用于缓存和追踪抗原上下文，也应该唯一。
    duplicated_ids = registry["antigen_id"][
        registry["antigen_id"].duplicated(keep=False)
    ]
    for antigen_id in sorted(set(duplicated_ids.astype(str))):
        add("error", "", "antigen_id", f"duplicated antigen_id={antigen_id}")

    for _, row in registry.iterrows():
        # 逐行清洗成统一格式，避免 " Protein "、NaN、大小写差异影响判断。
        group = clean_text(row.get("compatible_group", ""))
        antigen_type = clean_registry_enum(row.get("antigen_type", ""), "unknown")
        sequence = normalize_antigen_sequence(row.get("antigen_sequence", ""))
        confidence = clean_registry_enum(row.get("sequence_confidence", ""), "none")
        source = clean_registry_enum(row.get("sequence_source", ""), "missing")
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

    使用场景：
      自动构建出来的 registry 可能只有抗原名字，没有序列。
      你人工补了一张 updates 表后，可以用这个函数把人工补充列合并回来。

    合并规则：
      - key 默认是 compatible_group；
      - updates 里出现了 base 没有的新 group，就新增这一行；
      - prefer_updates=True 时，updates 里的非空字段覆盖 base；
      - 布尔列会强制转成 bool，避免 "TRUE"/"1"/"yes" 混在一起。
    """
    base = load_like_registry(base_registry).set_index(key, drop=False)
    patch = load_like_registry(updates).set_index(key, drop=False)

    for group, row in patch.iterrows():
        # 人工表里新增的 group，直接加入 registry。
        if group not in base.index:
            base.loc[group] = row
            continue
        if not prefer_updates:
            continue
        # 对已有 group，只有 updates 里“确实填了内容”的字段才覆盖 base。
        # 这样人工补表只需要填想改的列，不需要把整行全部复制一遍。
        for col in REGISTRY_COLUMNS:
            value = row.get(col, "")
            if col in BOOL_COLUMNS:
                base.at[group, col] = coerce_bool(value)
            elif clean_text(value):
                base.at[group, col] = value

    merged = base.reset_index(drop=True)
    # Ensure we always index with a list to avoid returning a Series when
    # REGISTRY_COLUMNS is a single string (which would make the return type
    # a Series instead of DataFrame).
    cols = REGISTRY_COLUMNS if isinstance(REGISTRY_COLUMNS, (list, tuple)) else [REGISTRY_COLUMNS]
    return merged[list(cols)]


def load_like_registry(data: pd.DataFrame | list[dict[str, Any]]) -> pd.DataFrame:
    """
    把 DataFrame 或 list[dict] 整理成 registry 形状。

    参数：
      data:
        已经在内存里的 DataFrame，或者 list[dict]。

    返回：
      pandas.DataFrame，列顺序符合 REGISTRY_COLUMNS。

    这个函数用于单元测试和人工 patch 表，不读写磁盘。
    它和 load_antigen_registry 的区别是：load_antigen_registry 从 CSV 文件读；
    load_like_registry 处理已经加载到内存的数据。
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
