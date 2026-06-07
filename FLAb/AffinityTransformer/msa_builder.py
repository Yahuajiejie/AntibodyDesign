"""
msa_builder.py — v3 MSA cache 构建与质检工具

本模块只处理本地 FASTA/A3M 文件，不下载数据库，也不在导入时调用外部程序。
外部工具如 MAFFT/HHblits/ColabFold 应作为离线预处理运行。
"""

from __future__ import annotations

import os
import random
from typing import Iterable

from .antigen_schema import normalize_antigen_sequence
from .homolog_search import FastaRecord


def strip_a3m_insertions(sequence: str, remove_gaps: bool = True) -> str:
    """
    把 A3M 序列还原成可比较的 query-like 序列。

    A3M 中小写字母通常表示插入位点；ESM-MSA 读取时可以保留 A3M，
    但质检 query 是否匹配原始序列时需要去掉小写插入。
    """
    chars: list[str] = []
    for char in sequence:
        if char.islower() or char == ".":
            continue
        if remove_gaps and char == "-":
            continue
        chars.append(char)
    return normalize_antigen_sequence("".join(chars))


def read_a3m(path: str) -> list[FastaRecord]:
    """
    读取 A3M 文件。

    A3M 是 FASTA-like 格式，但小写字母表示插入位点，不能像普通蛋白 FASTA
    那样统一转大写。因此这里单独解析，保留原始序列字符。
    """
    records: list[FastaRecord] = []
    header: str | None = None
    chunks: list[str] = []

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    sequence = "".join(chunks)
                    if sequence:
                        records.append(FastaRecord(header, sequence))
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line)

    if header is not None:
        sequence = "".join(chunks)
        if sequence:
            records.append(FastaRecord(header, sequence))
    return records


def write_a3m(records: Iterable[FastaRecord], path: str) -> None:
    """
    写出 A3M/FASTA-like MSA 文件。

    注意：
      如果 records 是未比对的 FASTA，这个函数不会自动比对；真正比对应由
      MAFFT/HHblits 等外部工具完成。
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            if not record.sequence:
                continue
            handle.write(f">{record.header}\n")
            sequence = record.sequence
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start:start + 80] + "\n")


def sample_msa_depth(
    records: list[FastaRecord],
    max_depth: int = 128,
    seed: int = 42,
    keep_first: bool = True,
) -> list[FastaRecord]:
    """
    对超深 MSA 做深度采样。

    参数：
      records:    MSA 记录，通常第一条是 query
      max_depth:  最多保留多少条
      seed:       随机种子
      keep_first: 是否固定保留第一条 query

    返回：
      采样后的记录列表。
    """
    if len(records) <= max_depth:
        return list(records)
    if max_depth <= 0:
        raise ValueError("max_depth 必须为正数")

    rng = random.Random(seed)
    if keep_first:
        query = records[0]
        pool = records[1:]
        chosen = rng.sample(pool, k=max_depth - 1)
        return [query] + chosen
    return rng.sample(records, k=max_depth)


def validate_msa(
    msa_path: str,
    query_sequence: str,
    min_depth: int = 2,
) -> dict[str, object]:
    """
    质检 MSA 文件。

    参数：
      msa_path:        A3M/FASTA-like MSA 路径
      query_sequence:  原始抗原序列
      min_depth:       至少多少条序列才认为有 MSA 信息

    返回：
      dict，包含 exists/depth/query_matches/has_msa 等统计。
    """
    query = normalize_antigen_sequence(query_sequence)
    result: dict[str, object] = {
        "msa_path": msa_path,
        "exists": os.path.exists(msa_path),
        "depth": 0,
        "query_length": len(query),
        "first_sequence_length": 0,
        "query_matches": False,
        "has_msa": False,
        "message": "",
    }
    if not result["exists"]:
        result["message"] = "MSA file does not exist"
        return result

    records = read_a3m(msa_path)
    result["depth"] = len(records)
    if not records:
        result["message"] = "MSA file is empty"
        return result

    first = strip_a3m_insertions(records[0].sequence)
    result["first_sequence_length"] = len(first)
    result["query_matches"] = bool(query and first == query)
    result["has_msa"] = bool(result["query_matches"] and len(records) >= min_depth)

    if not result["query_matches"]:
        result["message"] = "first MSA sequence does not match query_sequence"
    elif len(records) < min_depth:
        result["message"] = "MSA depth is below min_depth"
    else:
        result["message"] = "ok"
    return result


def build_mafft_command(
    input_fasta: str,
    threads: int = 8,
    auto: bool = True,
) -> list[str]:
    """
    构造 MAFFT 命令。

    返回：
      list[str]。MAFFT 通常把结果写到 stdout，调用方负责重定向到 .a3m/.fasta。
    """
    command = ["mafft"]
    if auto:
        command.append("--auto")
    command.extend(["--thread", str(threads), input_fasta])
    return command
