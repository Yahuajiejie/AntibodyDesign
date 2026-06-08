"""
antigen_schema.py — 抗原上下文数据结构

这个模块只定义 抗原编码器 需要的通用 schema 和小工具函数，不读取磁盘，也不依赖
训练流程。设计目标是让 antigen_registry、embedding cache、v3 feature matrix
使用同一套字段名和类型约定。
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import re
from typing import Any


REGISTRY_COLUMNS = [
    "antigen_id",
    "compatible_group",
    "dataset",
    "source_file",
    "antigen_name",
    "antigen_type",
    "antigen_sequence",
    "sequence_source",
    "sequence_accession",
    "sequence_confidence",
    "has_antigen_sequence",
    "is_protein",
    "is_glycoprotein",
    "is_peptide",
    "is_small_molecule",
    "is_carbohydrate",
    "ligand_smiles",
    "glycan_info",
    "msa_source",
    "msa_cache_path",
    "notes",
]

BOOL_COLUMNS = [
    "has_antigen_sequence",
    "is_protein",
    "is_glycoprotein",
    "is_peptide",
    "is_small_molecule",
    "is_carbohydrate",
]

ANTIGEN_TYPES = {
    "protein",
    "glycoprotein",
    "peptide",
    "small_molecule",
    "carbohydrate",
    "unknown",
}

PROTEIN_LIKE_TYPES = {"protein", "glycoprotein", "peptide"}
SEQUENCE_CONFIDENCES = {"high", "medium", "low", "none"}
SEQUENCE_SOURCES = {
    "csv",
    "metadata",
    "tasks",
    "uniprot",
    "pdb",
    "sabdab", # SAbDab SAbDab: Structural Antibody Database，牛津 OPIG 做的结构抗体数据库。它专门整理 PDB 中所有可用的抗体结构，并统一注释。SAbDab 会标注抗体结构、重轻链配对、抗原信息、实验信息、基因信息，有些条目还包括 antibody-antigen binding affinity
    "andd",
    "proteinbase",
    "paper",
    "manual",
    "missing",
    "unknown",
}

EMBEDDING_TYPES = {
    "single_esm2",
    "msa_esm1b",
    "ligand",
    "glycan",
}

MISSING_TOKENS = {"", "nan", "none", "null", "na", "n/a", "unknown", "-"}


@dataclass
class AntigenRecord:
    """
    antigen_registry 表格 每一行数据的实例

    参数：
      antigen_id:            稳定 ID，用于 cache 文件名；不要用抗原名字直接做文件名
      compatible_group:      与训练数据中的 compatible_group 对齐
      dataset/source_file:   追踪该抗原来自哪个数据集
      antigen_name:          至少应存在的抗原有效名字
      antigen_type:          protein/glycoprotein/peptide/small_molecule/...
      antigen_sequence:      仅蛋白或肽类抗原使用，非蛋白抗原必须留空
      sequence_source:       序列来源；缺失时写 missing，而不是空字符串
      sequence_accession:    UniProt/PDB/其它数据库 accession，可为空
      sequence_confidence:   high/medium/low/none
      ligand_smiles:         小分子抗原使用
      glycan_info:           糖抗原或糖基化信息使用
      msa_source/path:       MSA 预处理产物的来源和路径
      notes:                 人工质检备注
    """

    antigen_id: str
    compatible_group: str
    dataset: str = ""
    source_file: str = ""
    antigen_name: str = ""
    antigen_type: str = "unknown"
    antigen_sequence: str = ""
    sequence_source: str = "missing"
    sequence_accession: str = ""
    sequence_confidence: str = "none"
    has_antigen_sequence: bool = False
    is_protein: bool = False
    is_glycoprotein: bool = False
    is_peptide: bool = False
    is_small_molecule: bool = False
    is_carbohydrate: bool = False
    ligand_smiles: str = ""
    glycan_info: str = ""
    msa_source: str = ""
    msa_cache_path: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """返回可直接写入 pandas DataFrame 的 dict。"""
        return {field.name: getattr(self, field.name) for field in fields(self)}
        # fields()返回的是变量名及其他基本信息（数据类型），getattr则是返回变量名存放的数据

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> "AntigenRecord":
        """
        从 dict / pandas row 创建 AntigenRecord。

        self 代表的是当前对象

        cls 代表的则是抽象的类

        未提供的字段会使用 dataclass 默认值；bool 字段会做宽松转换。
        """
        values: dict[str, Any] = {}
        for field in fields(cls):
            value = row.get(field.name, field.default)
            if field.name in BOOL_COLUMNS:
                value = coerce_bool(value)
            values[field.name] = value
        values["antigen_sequence"] = normalize_antigen_sequence(
            values.get("antigen_sequence", "")
        )
        return cls(**values)


def clean_text(value: Any) -> str:
    """
    把 CSV/DataFrame 中的任意值整理成普通字符串。

    输入：
      value: 可能是 None、NaN、数字、字符串等。

    返回：
      str。明显缺失值返回空字符串；其它值去掉首尾空格。
    """
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in MISSING_TOKENS:
        return ""
    return text


def normalize_antigen_sequence(value: Any) -> str:
    """
    标准化蛋白/肽抗原序列。

    实现：
      1. 缺失值返回空字符串；
      2. 去掉空格、换行、数字和常见 FASTA 间隔符；
      3. 转成大写。

    返回：
      str。该函数不强行拒绝 X/U/O/B/Z 等非标准残基，避免误删真实数据。
    """
    text = clean_text(value)
    if not text:
        return ""
    text = re.sub(r"[\s0-9\-_.]", "", text)
    return text.upper()


def has_antigen_sequence(value: Any) -> bool:
    """判断一个 antigen_sequence 字段是否真的有序列。"""
    return bool(normalize_antigen_sequence(value))


def sequence_hash(sequence: str) -> str:
    """
    返回序列 SHA1 hash。

    用途：
      cache manifest 记录输入序列，防止序列更新后误读旧 embedding。
    """
    normalized = normalize_antigen_sequence(sequence)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def stable_antigen_id(
    compatible_group: str,
    antigen_name: str = "",
    antigen_sequence: str = "",
    sequence_accession: str = "",
) -> str:
    """
    生成不易重复的 antigen_id。

    输入：
      compatible_group: 训练数据中的可比较组
      antigen_name:     抗原名字
      antigen_sequence: 抗原序列，可为空
      sequence_accession: 外部数据库 accession，可为空

    返回：
      形如 ag_xxxxxxxxxxxx 的短 ID。它不是生物学 accession，只是本项目 cache key。
    """
    key = "|".join([
        clean_text(compatible_group),
        clean_text(antigen_name).lower(),
        normalize_antigen_sequence(antigen_sequence),
        clean_text(sequence_accession).lower(),
    ])
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"ag_{digest}"


def infer_antigen_type(
    antigen_name: str = "",
    antigen_sequence: str = "",
    ligand_smiles: str = "",
    glycan_info: str = "",
) -> str:
    """
    根据名字、序列和非蛋白字段粗略推断 antigen_type。

    这是保守推断，只用于 registry 初稿；人工质检仍然是必须的。
    """
    name = clean_text(antigen_name).lower()
    seq = normalize_antigen_sequence(antigen_sequence)
    smiles = clean_text(ligand_smiles)
    glycan = clean_text(glycan_info)

    if smiles:
        return "small_molecule"
    if glycan and not seq:
        return "carbohydrate"

    small_molecule_tokens = [
        "fluorescein",
        "biotin",
        "digoxigenin",
        "hapten",
        "dansyl",
        "nitrophenyl",
        "np-bsa",
    ]
    if any(token in name for token in small_molecule_tokens):
        return "small_molecule"

    carbohydrate_tokens = ["glycan", "carbohydrate", "polysaccharide", "sialyl"] # sialyl 唾液酸
    if any(token in name for token in carbohydrate_tokens):
        return "carbohydrate"

    if seq:
        return "peptide" if len(seq) < 50 else "protein"

    glycoprotein_tokens = ["glycoprotein"]
    if any(token in name for token in glycoprotein_tokens):
        return "glycoprotein"

    protein_tokens = [
        "protein",
        "receptor",
        "enzyme",
        "cytokine",
        "interleukin",
        "spike",
        "hemagglutinin",
        "antigen",
    ]
    if any(token in name for token in protein_tokens):
        return "protein"

    return "unknown"


def flags_from_antigen_type(antigen_type: str, antigen_sequence: str = "") -> dict[str, bool]:
    """
    根据 antigen_type 和 sequence 生成 registry 的布尔标记。

    返回：
      dict，包含 is_protein/is_glycoprotein/is_peptide/is_small_molecule/
      is_carbohydrate/has_antigen_sequence。
    """
    antigen_type = clean_text(antigen_type).lower() or "unknown"
    seq_exists = has_antigen_sequence(antigen_sequence)
    return {
        "has_antigen_sequence": seq_exists,
        "is_protein": antigen_type == "protein",
        "is_glycoprotein": antigen_type == "glycoprotein",
        "is_peptide": antigen_type == "peptide",
        "is_small_molecule": antigen_type == "small_molecule",
        "is_carbohydrate": antigen_type == "carbohydrate",
    }


def coerce_bool(value: Any) -> bool:
    """把 CSV 中常见的 bool 写法转成 Python bool。"""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "T", "yes", "y"}


def clean_registry_enum(value: Any, default: str) -> str:
    """
    标准化 registry 枚举字段，同时保留 unknown/none 这类合法枚举值。
    枚举字段指的我们的代码开头注册的enum类型变量，处理枚举字段 最重要的就是清除不符合要求的字段

    参数：
      value:   原始枚举值。
      default: 缺失时使用的默认值。

    返回：
      小写字符串。

    注意事项：
      之前定义的clean_text 会把 "unknown" 和 "none" 当作缺失值；但在 antigen_type 和
      sequence_confidence 中，它们是合法枚举。registry 需要单独处理。
    """
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"", "nan", "null", "na", "n/a", "-"}:
        return default
    return text


def ordered_registry_dict(row: dict[str, Any]) -> dict[str, Any]:
    """
    将任意 registry row 整理成 REGISTRY_COLUMNS 顺序。

    缺失字段补空字符串；布尔字段补 False。
    """
    ordered: dict[str, Any] = {}
    for col in REGISTRY_COLUMNS: # 在现在的表格中依次找到标准表格对应的行，组装成最终表格
        if col in BOOL_COLUMNS:
            ordered[col] = coerce_bool(row.get(col, False))
        elif col == "antigen_type":
            ordered[col] = clean_registry_enum(row.get(col, "unknown"), "unknown")
        elif col == "sequence_confidence":
            ordered[col] = clean_registry_enum(row.get(col, "none"), "none")
        elif col == "sequence_source":
            ordered[col] = clean_text(row.get(col, "")) or "missing"
        else:
            ordered[col] = clean_text(row.get(col, ""))
    return ordered
