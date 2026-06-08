"""
msa/homolog_search.py — 同源序列搜索辅助函数

这个模块不假设训练时联网，也不在导入时运行 BLAST/MMseqs2。它只提供：
  - FASTA 读写；
  - 去重和长度过滤；
  - 外部搜索命令的构造；
  - 可选的 subprocess 执行入口。

真正的大数据库搜索应在离线预处理阶段完成，训练阶段只读 cache。
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import subprocess
from typing import Iterable

from ..antigen_schema import clean_text, normalize_antigen_sequence


@dataclass(frozen=True)
class FastaRecord:
    """FASTA 记录。header 不含开头的 '>'。"""

    header: str
    sequence: str

    def normalized(self) -> "FastaRecord":
        """返回序列标准化后的记录。"""
        return FastaRecord(
            header=clean_text(self.header),
            sequence=normalize_antigen_sequence(self.sequence),
        )


def read_fasta(path: str) -> list[FastaRecord]:
    """
    读取 FASTA 文件。

    参数：
      path: FASTA 路径

    返回：
      list[FastaRecord]。空序列会被跳过。
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
                    seq = normalize_antigen_sequence("".join(chunks))
                    if seq:
                        records.append(FastaRecord(header, seq))
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line)

    if header is not None:
        seq = normalize_antigen_sequence("".join(chunks))
        if seq:
            records.append(FastaRecord(header, seq))
    return records


def write_fasta(records: Iterable[FastaRecord], path: str, line_width: int = 80) -> None:
    """
    写出 FASTA 文件。

    参数：
      records: FASTA 记录
      path: 输出路径
      line_width: 每行序列字符数
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            normalized = record.normalized()
            if not normalized.sequence:
                continue
            handle.write(f">{normalized.header}\n")
            seq = normalized.sequence
            for start in range(0, len(seq), line_width):
                handle.write(seq[start:start + line_width] + "\n")


def deduplicate_records(records: Iterable[FastaRecord]) -> list[FastaRecord]:
    """
    按序列去重，保留第一次出现的 header。

    返回：
      list[FastaRecord]。
    """
    seen: set[str] = set()
    unique: list[FastaRecord] = []
    for record in records:
        normalized = record.normalized()
        if not normalized.sequence or normalized.sequence in seen:
            continue
        seen.add(normalized.sequence)
        unique.append(normalized)
    return unique


def filter_homologs(
    records: Iterable[FastaRecord],
    query_sequence: str,
    min_length_ratio: float = 0.5,
    max_length_ratio: float = 1.5,
    max_records: int = 256,
) -> list[FastaRecord]:
    """
    对搜索到的同源序列做轻量过滤。

    参数：
      records:           搜索结果 FASTA 记录
      query_sequence:    原始抗原序列
      min_length_ratio:  homolog 长度至少为 query 的多少倍
      max_length_ratio:  homolog 长度至多为 query 的多少倍
      max_records:       最多保留多少条

    返回：
      去重、长度过滤后的记录。identity 过滤应由 BLAST/MMseqs2 阶段完成。
    """
    query = normalize_antigen_sequence(query_sequence)
    if not query:
        raise ValueError("query_sequence 为空，无法过滤 homolog")

    min_len = int(len(query) * min_length_ratio)
    max_len = int(len(query) * max_length_ratio)
    filtered: list[FastaRecord] = []

    for record in deduplicate_records(records):
        length = len(record.sequence)
        if min_len <= length <= max_len:
            filtered.append(record)
        if len(filtered) >= max_records:
            break
    return filtered


def write_homolog_fasta(
    query_id: str,
    query_sequence: str,
    homologs: Iterable[FastaRecord],
    path: str,
    max_homologs: int = 255,
) -> None:
    """
    写出 query-first homolog FASTA。

    ESM-MSA 和后续 MSA 质检都要求 query 序列在第一条。
    """
    query = FastaRecord(query_id, normalize_antigen_sequence(query_sequence))
    if not query.sequence:
        raise ValueError("query_sequence 为空，无法写 homolog FASTA")
    homolog_list = deduplicate_records(homologs)[:max_homologs]
    write_fasta([query] + homolog_list, path)


def build_blastp_command(
    query_fasta: str,
    database: str,
    output_tsv: str,
    evalue: str = "1e-5",
    max_target_seqs: int = 512,
    threads: int = 8,
) -> list[str]:
    """
    构造 blastp 命令。

    返回：
      list[str]，可直接交给 subprocess.run(command)。
    """
    return [
        "blastp",
        "-query", query_fasta,
        "-db", database,
        "-out", output_tsv,
        "-evalue", str(evalue),
        "-max_target_seqs", str(max_target_seqs),
        "-num_threads", str(threads),
        "-outfmt", "6 qseqid sseqid pident length evalue bitscore qseq sseq",
    ]


def build_mmseqs_easy_search_command(
    query_fasta: str,
    target_database: str,
    output_tsv: str,
    tmp_dir: str,
    min_seq_id: float = 0.3,
    threads: int = 8,
) -> list[str]:
    """
    构造 mmseqs easy-search 命令。

    min_seq_id=0.3 对应 v3 方案中的 identity > 30% 粗过滤。
    """
    return [
        "mmseqs",
        "easy-search",
        query_fasta,
        target_database,
        output_tsv,
        tmp_dir,
        "--min-seq-id", str(min_seq_id),
        "--threads", str(threads),
        "--format-output", "query,target,pident,alnlen,evalue,bits,qseq,tseq",
    ]


def run_search_command(
    command: list[str],
    dry_run: bool = True,
    check: bool = True,
) -> subprocess.CompletedProcess | list[str]:
    """
    运行外部 homolog search 命令。

    参数：
      command: subprocess 命令列表
      dry_run: True 时只返回 command，不执行
      check: subprocess.run 的 check 参数

    返回：
      dry_run=True 返回 command；否则返回 CompletedProcess。
    """
    if dry_run:
        return command
    return subprocess.run(command, check=check)
