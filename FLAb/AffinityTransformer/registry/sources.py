"""
registry/sources.py — 外部抗原信息解析器

registry stage 的目标不是直接训练 attention 模型，而是先把“抗原是谁、有没有序列、
信息来自哪里”整理成可质检的数据表。本模块只解析输入文件，不生成 embedding，
也不 import v1/v2 代码。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import glob
import io
import json
import os
import re
import zipfile
from typing import Any

import pandas as pd

from ..antigen_schema import (
    REGISTRY_COLUMNS,
    clean_text,
    flags_from_antigen_type,
    infer_antigen_type,
    normalize_antigen_sequence,
    ordered_registry_dict,
    stable_antigen_id,
)


PROTEIN_ALPHABET = set("ACDEFGHIKLMNPQRSTVWYBXZUOJ")
CSV_ENCODING_CANDIDATES = ("utf-8", "utf-8-sig", "gb18030", "latin1")
INVISIBLE_PREFIX_CHARS = "\ufeff\u200b\u200c\u200d"


@dataclass(frozen=True)
class MarkdownFastaRecord:
    """
    从 markdown 文档中抽取到的一条 FASTA-like 记录。

    参数：
      header:     markdown 中 `>` 后面的标题，已去掉粗体符号
      sequence:   只保留英文字母后的标准化蛋白序列
      line_start: header 在原文中的 1-based 行号
    """

    header: str
    sequence: str
    line_start: int


@dataclass(frozen=True)
class TaskControlAntibody:
    """
    TASKS.md 中给出的对照抗体序列。

    参数：
      control_name:   m102.4 / n425 / HENV-26 等对照名
      antibody_format: Fv / VHH / unknown
      chain_role:     heavy / light / vhh / unknown
      sequence:       可变区序列
      sequence_source: 例如 tasks:m102.4
      target_antigen: 对应任务靶标名
      notes:          来源、专利/PDB 等说明
    """

    control_name: str
    antibody_format: str
    chain_role: str
    sequence: str
    sequence_source: str
    target_antigen: str
    notes: str = ""


def slugify(value: Any) -> str:
    """
    将抗原名或 target slug 变成稳定的短文本。

    参数：
      value: 任意文本。

    返回：
      小写、用下划线连接的 slug。空值返回 unknown。
    """
    text = clean_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "unknown"


def _letters_only(value: str) -> str:
    """只保留英文字符并转大写，用于从 markdown 行中提取序列。"""
    return re.sub(r"[^A-Za-z]", "", value).upper()


def _looks_like_protein_sequence_line(value: str) -> bool:
    """
    判断 markdown 中的一行是否像蛋白序列。

    实现：
      - 只看英文字母；
      - 至少 10 个字母；
      - 90% 以上字符落在蛋白序列常见字母表中。
    """
    letters = _letters_only(value)
    if len(letters) < 10:
        return False
    valid = sum(1 for char in letters if char in PROTEIN_ALPHABET)
    return valid / len(letters) >= 0.90


def _clean_markdown_fasta_header(value: str) -> str:
    """清理 markdown FASTA header 中的 `>`、粗体符号和多余空白。"""
    text = clean_text(_remove_invisible_chars(value))
    text = text.lstrip("\\").lstrip(">＞").replace("**", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" :")


def _remove_invisible_chars(value: str) -> str:
    """删除 markdown 复制文本中常见的零宽字符。"""
    return value.translate({ord(char): None for char in INVISIBLE_PREFIX_CHARS})


def _is_markdown_fasta_header(value: str) -> bool:
    """判断一行是否是普通或转义 markdown FASTA header。"""
    stripped = _remove_invisible_chars(value).strip()
    return (
        stripped.startswith(">")
        or stripped.startswith("＞")
        or stripped.startswith("\\>")
        or stripped.startswith("\\＞")
    )


def extract_markdown_fasta_records(path: str) -> list[MarkdownFastaRecord]:
    """
    从 markdown 文档中抽取 FASTA-like 记录。

    参数：
      path: TASKS.md 或其它 markdown 文档路径。

    返回：
      list[MarkdownFastaRecord]。

    实现：
      扫描以 `>` 开头的 header，并收集后续看起来像蛋白序列的行。中文说明、
      空行、普通段落会被跳过，因此适合解析赛事文档这类非标准 FASTA。
    """
    records: list[MarkdownFastaRecord] = []
    current_header = ""
    current_start = 0
    sequence_parts: list[str] = []

    def flush() -> None:
        nonlocal current_header, current_start, sequence_parts
        if current_header and sequence_parts:
            records.append(MarkdownFastaRecord(
                header=current_header,
                sequence=normalize_antigen_sequence("".join(sequence_parts)),
                line_start=current_start,
            ))
        current_header = ""
        current_start = 0
        sequence_parts = []

    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if _is_markdown_fasta_header(stripped):
                flush()
                current_header = _clean_markdown_fasta_header(stripped)
                current_start = line_no
                sequence_parts = []
                continue
            if current_header:
                if _looks_like_protein_sequence_line(stripped):
                    sequence_parts.append(_letters_only(stripped))
                elif sequence_parts:
                    flush()

    flush()
    return records


def _task_header_kind(header: str) -> str:
    """把 TASKS.md FASTA header 归类为 antigen 或 control。"""
    text = header.lower()
    if "40980-v08h" in text:
        return "antigen"
    if any(name in text for name in ["m102.4", "n425", "henv-26"]):
        return "control"
    return "unknown"


def _control_from_task_record(record: MarkdownFastaRecord) -> TaskControlAntibody:
    """
    将 TASKS.md 中的一条对照抗体 FASTA 记录转成结构化对象。

    改进思路：
      早期文档只说明 TASKS.md “可能有用”。registry stage 明确把对照抗体拆成独立表，
      供候选序列相似性过滤和 sanity check 使用，不混入亲和力训练标签。
    """
    header_lower = record.header.lower()
    if "m102.4" in header_lower:
        name = "m102.4"
    elif "n425" in header_lower:
        name = "n425"
    elif "henv-26" in header_lower:
        name = "HENV-26"
    else:
        name = "unknown_control"

    if "vhh" in header_lower:
        antibody_format = "VHH"
        chain_role = "vhh"
    else:
        antibody_format = "Fv" if "fv" in header_lower else "unknown"
        if " h chain" in f" {header_lower} " or "heavy" in header_lower:
            chain_role = "heavy"
        elif " l chain" in f" {header_lower} " or "light" in header_lower:
            chain_role = "light"
        else:
            chain_role = "unknown"

    return TaskControlAntibody(
        control_name=name,
        antibody_format=antibody_format,
        chain_role=chain_role,
        sequence=record.sequence,
        sequence_source=f"tasks:{slugify(name)}",
        target_antigen="Nipah virus G protein",
        notes=f"header={record.header}; line_start={record.line_start}",
    )


def parse_tasks_markdown(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    解析 TASKS.md 中的官方靶标和对照抗体。

    参数：
      path: references/task_docs/TASKS.md。

    返回：
      (task_antigen_registry, task_controls)：
        task_antigen_registry: REGISTRY_COLUMNS 形状，通常包含 Nipah G protein；
        task_controls:         对照抗体表，不进入训练标签。

    改进思路：
      registry stage 把 TASKS.md 从“人工阅读材料”升级为可复现输入源。官方给出的
      40980-V08H 序列被标记为 sequence_confidence=high，后续可直接用于
      antigen single embedding 和 MSA query。
    """
    fasta_records = extract_markdown_fasta_records(path)
    antigen_rows: list[dict[str, Any]] = []
    controls: list[TaskControlAntibody] = []

    for record in fasta_records:
        kind = _task_header_kind(record.header)
        if kind == "antigen":
            compatible_group = "tasks:nipah_g_40980_v08h"
            antigen_name = "Nipah virus G protein"
            antigen_type = "glycoprotein"
            antigen_id = stable_antigen_id(
                compatible_group=compatible_group,
                antigen_name=antigen_name,
                antigen_sequence=record.sequence,
                sequence_accession="40980-V08H",
            )
            flags = flags_from_antigen_type(antigen_type, record.sequence)
            antigen_rows.append(ordered_registry_dict({
                "antigen_id": antigen_id,
                "compatible_group": compatible_group,
                "dataset": "TASKS.md",
                "source_file": os.path.basename(path),
                "antigen_name": antigen_name,
                "antigen_type": antigen_type,
                "antigen_sequence": record.sequence,
                "sequence_source": "tasks:40980-V08H",
                "sequence_accession": "40980-V08H",
                "sequence_confidence": "high",
                "ligand_smiles": "",
                "glycan_info": "Nipah G is a viral glycoprotein; glycan sites not enumerated in TASKS.md",
                "msa_source": "",
                "msa_cache_path": "",
                "notes": (
                    "official wet-lab example antigen; sequence includes C-terminal His tag; "
                    f"header={record.header}; line_start={record.line_start}"
                ),
                **flags,
            }))
        elif kind == "control":
            controls.append(_control_from_task_record(record))

    antigen_df = pd.DataFrame(antigen_rows, columns=REGISTRY_COLUMNS)
    controls_df = pd.DataFrame([asdict(item) for item in controls], columns=[
        "control_name",
        "antibody_format",
        "chain_role",
        "sequence",
        "sequence_source",
        "target_antigen",
        "notes",
    ])
    return antigen_df, controls_df


def read_csv_or_zip_tables(path: str, low_memory: bool = False) -> dict[str, pd.DataFrame]:
    """
    读取单个 CSV 或 csv.zip。

    参数：
      path: .csv 或 .csv.zip 路径。
      low_memory: 传给 pandas.read_csv。

    返回：
      dict，key 为表名，value 为 DataFrame。

    改进思路：
      旧读取逻辑只读 zip 中第一个 CSV。registry stage 在 registry 阶段会读取 zip 内
      所有 CSV，并添加 source_file/source_member，避免压缩包多表时静默漏数。
    """
    tables: dict[str, pd.DataFrame] = {}
    source_file = os.path.basename(path)

    if path.endswith(".csv.zip"):
        with zipfile.ZipFile(path) as archive:
            csv_names = sorted(name for name in archive.namelist() if name.endswith(".csv"))
            if not csv_names:
                raise ValueError(f"{path} 中没有 csv 文件")
            for member in csv_names:
                with archive.open(member) as handle:
                    frame = _read_csv_bytes_with_fallback(
                        handle.read(),
                        low_memory=low_memory,
                    )
                frame["source_file"] = source_file
                frame["source_member"] = member
                key = f"{os.path.splitext(source_file)[0]}::{os.path.splitext(os.path.basename(member))[0]}"
                tables[key] = frame
        return tables

    frame = _read_csv_path_with_fallback(path, low_memory=low_memory)
    frame["source_file"] = source_file
    tables[os.path.splitext(source_file)[0]] = frame
    return tables


def _read_csv_path_with_fallback(path: str, low_memory: bool = False) -> pd.DataFrame:
    """
    读取 CSV 路径，并在编码失败时尝试常见编码。

    参数：
      path: CSV 文件路径。
      low_memory: 传给 pandas.read_csv。

    返回：
      pandas.DataFrame。

    改进思路：
      FLAb/外部数据中可能混有 UTF-8、UTF-8-BOM、GB 系编码或 latin1 文本。
      registry 阶段只需要字段级信息，应该在读取层做编码 fallback。
    """
    last_error: Exception | None = None
    for encoding in CSV_ENCODING_CANDIDATES:
        try:
            return pd.read_csv(path, low_memory=low_memory, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(path, low_memory=low_memory)


def _read_csv_bytes_with_fallback(data: bytes, low_memory: bool = False) -> pd.DataFrame:
    """
    读取 zip member 的 CSV bytes，并在编码失败时尝试常见编码。

    参数：
      data: zip member 原始 bytes。
      low_memory: 传给 pandas.read_csv。

    返回：
      pandas.DataFrame。
    """
    last_error: Exception | None = None
    for encoding in CSV_ENCODING_CANDIDATES:
        try:
            return pd.read_csv(
                io.BytesIO(data),
                low_memory=low_memory,
                encoding=encoding,
            )
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(io.BytesIO(data), low_memory=low_memory)


def load_binding_tables(
    binding_dir: str,
    pattern: str = "*.csv*",
    max_files: int | None = None,
) -> dict[str, pd.DataFrame]:
    """
    读取 FLAb/data/binding 下的 CSV/CSV.zip 表。

    参数：
      binding_dir: binding 数据目录。
      pattern:     glob pattern，默认读取 *.csv*。
      max_files:   可选，只读取前 N 个文件，方便本地 debug。

    返回：
      dict[str, DataFrame]，可直接传给 build_antigen_registry。
    """
    paths = sorted(glob.glob(os.path.join(binding_dir, pattern)))
    if max_files is not None:
        paths = paths[:max_files]

    datasets: dict[str, pd.DataFrame] = {}
    for path in paths:
        if os.path.basename(path).startswith("."):
            continue
        datasets.update(read_csv_or_zip_tables(path))
    return datasets


def load_flab_metadata(metadata_csv: str) -> dict[str, dict[str, Any]]:
    """
    读取 FLAb metadata 并按 filename 建索引。

    参数：
      metadata_csv: FLAb/data/flab_metadata.csv。

    返回：
      dict，key 是 filename，value 是该行 metadata。
    """
    frame = pd.read_csv(metadata_csv, low_memory=False)
    if "filename" not in frame.columns:
        raise ValueError(f"{metadata_csv} 缺少 filename 列")
    return {
        clean_text(row["filename"]): row.to_dict()
        for _, row in frame.iterrows()
    }


def _safe_json_loads(value: Any) -> Any:
    """宽松解析 JSON；失败时返回 None。"""
    text = clean_text(value)
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def build_proteinbase_target_index(path: str) -> pd.DataFrame:
    """
    从 proteinbase 导出的 evaluations JSON 中统计 target 信息。

    参数：
      path: proteinbase_all_data_28_01_2026.csv。

    返回：
      DataFrame，按 target_slug 汇总 n_rows/n_kd/binding 等信息。

    改进思路：
      proteinbase 行本身多是抗体/设计序列，不是抗原表。registry stage 不把这些行混入
      FLAb 标签，而是先抽取 target 索引，告诉后续 registry 哪些 target 在
      proteinbase 中出现过、是否有 Kd/binding 证据。
    """
    frame = pd.read_csv(path, low_memory=False)
    required = {"id", "name", "sequence", "evaluations"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} 缺少列: {sorted(missing)}")

    summary: dict[str, dict[str, Any]] = {}
    row_sets: dict[str, set[str]] = defaultdict(set)

    for _, row in frame.iterrows():
        row_id = clean_text(row.get("id", ""))
        evaluations = _safe_json_loads(row.get("evaluations", ""))
        if not isinstance(evaluations, list):
            continue
        for item in evaluations:
            if not isinstance(item, dict):
                continue
            target = clean_text(item.get("target", ""))
            if not target:
                continue
            target_slug = slugify(target.replace("-", " "))
            record = summary.setdefault(target_slug, {
                "source_file": os.path.basename(path),
                "target_slug": target_slug,
                "target_name": target.replace("-", " "),
                "n_rows": 0,
                "n_experimental": 0,
                "n_kd": 0,
                "n_binding_true": 0,
                "n_binding_false": 0,
                "binding_strength_labels": set(),
                "example_ids": [],
            })
            row_sets[target_slug].add(row_id)
            if item.get("type") == "experimental":
                record["n_experimental"] += 1
            metric = clean_text(item.get("metric", "")).lower()
            if metric == "kd":
                record["n_kd"] += 1
            elif metric == "binding":
                value = item.get("value")
                if value is True:
                    record["n_binding_true"] += 1
                elif value is False:
                    record["n_binding_false"] += 1
            elif metric == "binding_strength":
                label = clean_text(item.get("value", ""))
                if label:
                    record["binding_strength_labels"].add(label)

    rows: list[dict[str, Any]] = []
    for target_slug, record in sorted(summary.items()):
        ids = sorted(row_sets[target_slug])
        rows.append({
            **record,
            "n_rows": len(ids),
            "binding_strength_labels": "|".join(sorted(record["binding_strength_labels"])),
            "example_ids": "|".join(ids[:10]),
        })
    return pd.DataFrame(rows)


def proteinbase_target_registry(path: str) -> pd.DataFrame:
    """
    将 proteinbase target index 转成 registry 形状。

    参数：
      path: proteinbase_all_data_28_01_2026.csv。

    返回：
      REGISTRY_COLUMNS 形状的 DataFrame。多数 target 只有名字，没有序列。
    """
    target_index = build_proteinbase_target_index(path)
    rows: list[dict[str, Any]] = []
    for _, row in target_index.iterrows():
        target_slug = clean_text(row["target_slug"])
        antigen_name = clean_text(row["target_name"])
        compatible_group = f"proteinbase:{target_slug}"
        antigen_type = infer_antigen_type(antigen_name=antigen_name)
        flags = flags_from_antigen_type(antigen_type, "")
        rows.append(ordered_registry_dict({
            "antigen_id": stable_antigen_id(compatible_group, antigen_name),
            "compatible_group": compatible_group,
            "dataset": "proteinbase",
            "source_file": os.path.basename(path),
            "antigen_name": antigen_name,
            "antigen_type": antigen_type,
            "antigen_sequence": "",
            "sequence_source": "proteinbase:evaluations",
            "sequence_accession": target_slug,
            "sequence_confidence": "none",
            "ligand_smiles": "",
            "glycan_info": "",
            "msa_source": "",
            "msa_cache_path": "",
            "notes": (
                f"n_rows={row['n_rows']}; n_kd={row['n_kd']}; "
                f"n_binding_true={row['n_binding_true']}; "
                f"binding_strength={row['binding_strength_labels']}"
            ),
            **flags,
        }))
    return pd.DataFrame(rows, columns=REGISTRY_COLUMNS)


def _find_first_column(columns: list[str], candidates: list[str]) -> str | None:
    """按大小写不敏感规则寻找第一个候选列。"""
    by_lower = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    return None


def andd_antigen_registry(path: str) -> pd.DataFrame:
    """
    从 ANDD.xlsx 中提取抗原名和抗原序列。

    参数：
      path: ANDD.xlsx。

    返回：
      REGISTRY_COLUMNS 形状的 DataFrame，每个 unique antigen name/sequence 一行。

    改进思路：
      ANDD 是纳米抗体数据源，含 Ag_Name/Ag_Seq/Kd 等字段。registry stage 先把抗原
      作为 external registry 资产登记，不把 nanobody 样本直接塞进 MLP 训练。
    """
    try:
        frame = pd.read_excel(path)
    except ImportError as exc:
        raise ImportError(
            "读取 ANDD.xlsx 需要 openpyxl，请安装 openpyxl>=3.1.0"
        ) from exc

    name_col = _find_first_column(
        list(frame.columns),
        ["Ag_Name", "antigen_name", "Antigen", "Target", "target_name"],
    )
    seq_col = _find_first_column(
        list(frame.columns),
        ["Ag_Seq", "antigen_sequence", "antigen_seq", "Target_Seq", "target_sequence"],
    )
    if name_col is None:
        raise ValueError(f"{path} 中找不到抗原名称列")

    kd_col = _find_first_column(
        list(frame.columns),
        ["Affinity_Kd", "Kd", "KD", "Affinity"],
    )
    grouped = frame.groupby([name_col, seq_col] if seq_col else [name_col], dropna=False)
    rows: list[dict[str, Any]] = []
    for group_key, group_df in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        antigen_name = clean_text(group_key[0])
        antigen_sequence = normalize_antigen_sequence(group_key[1] if seq_col else "")
        if not antigen_name and not antigen_sequence:
            continue
        compatible_group = f"andd:{slugify(antigen_name or antigen_sequence[:24])}"
        antigen_type = infer_antigen_type(
            antigen_name=antigen_name,
            antigen_sequence=antigen_sequence,
        )
        mixed_antigen_note = ""
        if antigen_sequence and antigen_type in {"small_molecule", "carbohydrate"}:
            mixed_antigen_note = (
                f"; inferred_nonprotein_from_name={antigen_type}; "
                "Ag_Seq present, treated as protein-like antigen"
            )
            antigen_type = "peptide" if len(antigen_sequence) < 50 else "protein"
        flags = flags_from_antigen_type(antigen_type, antigen_sequence)
        n_kd = int(group_df[kd_col].notna().sum()) if kd_col else 0
        rows.append(ordered_registry_dict({
            "antigen_id": stable_antigen_id(
                compatible_group,
                antigen_name,
                antigen_sequence,
            ),
            "compatible_group": compatible_group,
            "dataset": "ANDD",
            "source_file": os.path.basename(path),
            "antigen_name": antigen_name,
            "antigen_type": antigen_type,
            "antigen_sequence": antigen_sequence,
            "sequence_source": f"andd:{seq_col}" if antigen_sequence else "missing",
            "sequence_accession": "",
            "sequence_confidence": "high" if antigen_sequence else "none",
            "ligand_smiles": "",
            "glycan_info": "",
            "msa_source": "",
            "msa_cache_path": "",
            "notes": (
                f"n_rows={len(group_df)}; n_kd={n_kd}; "
                f"name_col={name_col}; seq_col={seq_col or ''}"
                f"{mixed_antigen_note}"
            ),
            **flags,
        }))
    return pd.DataFrame(rows, columns=REGISTRY_COLUMNS)


def _split_pipe_values(value: Any) -> list[str]:
    """拆分 SAbDab summary 中的 `a | b | c` 字段。"""
    text = clean_text(value)
    if not text:
        return []
    return [clean_text(part) for part in text.split("|")]


def _normalize_sabdab_antigen_type(value: Any, antigen_name: str) -> str:
    """将 SAbDab antigen_type 映射到本项目 antigen_type。"""
    text = clean_text(value).lower()
    if "glycoprotein" in text:
        return "glycoprotein"
    if "protein" in text:
        return "protein"
    if "peptide" in text:
        return "peptide"
    if "carbohydrate" in text or "glycan" in text:
        return "carbohydrate"
    if "hapten" in text or "small" in text or "het" in text:
        return "small_molecule"
    return infer_antigen_type(antigen_name=antigen_name)


def sabdab_antigen_registry(path: str) -> pd.DataFrame:
    """
    从 SAbDab summary tsv 中提取结构抗原索引。

    参数：
      path: sabdab_summary_all.tsv。

    返回：
      REGISTRY_COLUMNS 形状的 DataFrame。SAbDab summary 通常只有链 ID/名字/
      类型，真实序列需要后续从 PDB raw/imgt/chothia 文件提取，因此这里
      sequence_confidence=none。
    """
    frame = pd.read_csv(path, sep="\t", low_memory=False)
    required = {"pdb", "antigen_name", "antigen_type"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} 缺少列: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        pdb_id = clean_text(row.get("pdb", "")).lower()
        if not pdb_id:
            continue
        names = _split_pipe_values(row.get("antigen_name", ""))
        types = _split_pipe_values(row.get("antigen_type", ""))
        chains = _split_pipe_values(row.get("antigen_chain", ""))
        n_items = max(len(names), len(types), len(chains), 1)

        for idx in range(n_items):
            antigen_name = names[idx] if idx < len(names) else ""
            antigen_type_raw = types[idx] if idx < len(types) else ""
            chain = chains[idx] if idx < len(chains) else str(idx + 1)
            if not antigen_name:
                continue
            antigen_type = _normalize_sabdab_antigen_type(antigen_type_raw, antigen_name)
            compatible_group = f"sabdab:{pdb_id}:{slugify(chain or str(idx + 1))}"
            flags = flags_from_antigen_type(antigen_type, "")
            affinity = clean_text(row.get("affinity", ""))
            affinity_method = clean_text(row.get("affinity_method", ""))
            rows.append(ordered_registry_dict({
                "antigen_id": stable_antigen_id(compatible_group, antigen_name),
                "compatible_group": compatible_group,
                "dataset": "SAbDab",
                "source_file": os.path.basename(path),
                "antigen_name": antigen_name,
                "antigen_type": antigen_type,
                "antigen_sequence": "",
                "sequence_source": "sabdab:summary",
                "sequence_accession": pdb_id,
                "sequence_confidence": "none",
                "ligand_smiles": "",
                "glycan_info": "",
                "msa_source": "",
                "msa_cache_path": "",
                "notes": (
                    f"pdb={pdb_id}; antigen_chain={chain}; "
                    f"raw_antigen_type={antigen_type_raw}; "
                    f"affinity={affinity}; affinity_method={affinity_method}"
                ),
                **flags,
            }))
    return pd.DataFrame(rows, columns=REGISTRY_COLUMNS)


def build_external_antigen_registry(
    tasks_md: str | None = None,
    proteinbase_csv: str | None = None,
    andd_xlsx: str | None = None,
    sabdab_summary_tsv: str | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    汇总 外部抗原来源。

    参数：
      tasks_md:          TASKS.md 路径，可为空。
      proteinbase_csv:   proteinbase_all_data_28_01_2026.csv 路径，可为空。
      andd_xlsx:         ANDD.xlsx 路径，可为空。
      sabdab_summary_tsv: SAbDab summary tsv 路径，可为空。

    返回：
      (external_registry, auxiliary_tables)。

    auxiliary_tables 当前包含：
      task_controls:       TASKS.md 对照抗体序列表；
      proteinbase_targets: proteinbase target 统计表。
    """
    registry_parts: list[pd.DataFrame] = []
    auxiliary: dict[str, pd.DataFrame] = {}

    if tasks_md:
        task_registry, task_controls = parse_tasks_markdown(tasks_md)
        registry_parts.append(task_registry)
        auxiliary["task_controls"] = task_controls

    if proteinbase_csv:
        proteinbase_targets = build_proteinbase_target_index(proteinbase_csv)
        registry_parts.append(proteinbase_target_registry(proteinbase_csv))
        auxiliary["proteinbase_targets"] = proteinbase_targets

    if andd_xlsx:
        registry_parts.append(andd_antigen_registry(andd_xlsx))

    if sabdab_summary_tsv:
        registry_parts.append(sabdab_antigen_registry(sabdab_summary_tsv))

    if registry_parts:
        registry = pd.concat(registry_parts, ignore_index=True)
    else:
        registry = pd.DataFrame(columns=REGISTRY_COLUMNS)
    return registry[REGISTRY_COLUMNS], auxiliary
