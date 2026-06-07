"""
config.py — AffinityTransformer(v3) 独立配置

v3 的目标是 score = f(antibody, antigen_context)。这里的配置只服务 v3，
不读取也不覆盖 AffinityMLP(v1/v2) 的 config。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SequenceEncoderSpec:
    """
    序列编码模型的配置。

    参数：
      alias:             本项目内部短名
      model_name:        Hugging Face 或外部模型名
      embedding_dim:     输出 hidden size；None 表示加载模型后从 config 自动读取
      tokenizer_style:   raw / space / paired_t5
      architecture:      auto / t5_encoder
      max_length:        tokenizer 最大长度
      trust_remote_code: 是否允许 Hugging Face remote code
      notes:             人工说明
    """

    alias: str
    model_name: str
    embedding_dim: int | None
    tokenizer_style: str = "raw"
    architecture: str = "auto"
    max_length: int = 512
    trust_remote_code: bool = False
    notes: str = ""


ENCODER_PRESETS: dict[str, SequenceEncoderSpec] = {
    "esm2_650m": SequenceEncoderSpec(
        alias="esm2_650m",
        model_name="facebook/esm2_t33_650M_UR50D",
        embedding_dim=1280,
        tokenizer_style="raw",
        architecture="auto",
        max_length=512,
        notes="general protein language model; default antigen single-sequence encoder",
    ),
    "esm2_150m": SequenceEncoderSpec(
        alias="esm2_150m",
        model_name="facebook/esm2_t30_150M_UR50D",
        embedding_dim=640,
        tokenizer_style="raw",
        architecture="auto",
        max_length=512,
        notes="smaller ESM2 option for quick ablations",
    ),
    "igbert": SequenceEncoderSpec(
        alias="igbert",
        model_name="Exscientia/IgBert",
        embedding_dim=1024,
        tokenizer_style="space",
        architecture="auto",
        max_length=512,
        notes="antibody language model; hidden_size=1024 in Hugging Face config",
    ),
    "igt5": SequenceEncoderSpec(
        alias="igt5",
        model_name="Exscientia/IgT5",
        embedding_dim=None,
        tokenizer_style="paired_t5",
        architecture="t5_encoder",
        max_length=512,
        notes="paired antibody language model; use heavy </s> light token format",
    ),
}


class V3Config:
    """
    v3 默认配置。

    关键策略：
      - 官方有抗原序列：single antigen embedding + MSA embedding；
      - 官方没有抗原序列：不伪造 single embedding，只使用 MSA embedding；
      - 抗体 encoder 可从 ESM2 切到 IgBert/IgT5 或自定义 HF 模型。
    """

    MODEL_VERSION = "v3"

    # Antibody encoder.
    ANTIBODY_ENCODER = "esm2_650m"
    # separate_chains: heavy/light 分别编码后拼接；
    # paired_chains:   适合 IgT5 这类 paired antibody encoder。
    ANTIBODY_ENCODER_LAYOUT = "separate_chains"

    # Antigen encoders.
    ANTIGEN_SINGLE_ENCODER = "esm2_650m"
    ANTIGEN_MSA_MODEL_NAME = "esm_msa1b_t12_100M_UR50S"
    ANTIGEN_MSA_EMBEDDING_DIM = 768

    # Feature policy.
    ANTIGEN_CONTEXT_POLICY = "official_sequence_then_msa"
    INCLUDE_ANTIGEN_TYPE_FLAGS = True
    ALLOW_MISSING_ANTIGEN_CONTEXT = False

    # Cache paths. These are local artifacts and should not be committed.
    ANTIBODY_CACHE_DIR = "cache/v3/antibody_embeddings"
    ANTIGEN_CACHE_DIR = "cache/v3/antigen_embeddings"
    MSA_CACHE_DIR = "cache/v3/msa"

    # Model dimensions.
    TRANSFORMER_TOKEN_DIM = 256
    TRANSFORMER_LAYERS = 2
    TRANSFORMER_HEADS = 8
    TRANSFORMER_FF_DIM = 512
    DROPOUT = 0.2

    # Training defaults. v3 trainer can reuse these later without touching v1/v2.
    EPOCHS = 100
    LR = 1e-4
    BATCH_SIZE = 64
    WEIGHT_DECAY = 1e-4
    SEED = 42


cfg = V3Config()


def get_encoder_spec(alias_or_model_name: str) -> SequenceEncoderSpec:
    """
    获取 encoder 配置。

    如果传入的是预设 alias，返回预设；否则按 Hugging Face model name 构造
    custom_hf 配置，embedding_dim 会在加载后自动读取。
    """
    if alias_or_model_name in ENCODER_PRESETS:
        return ENCODER_PRESETS[alias_or_model_name]
    return SequenceEncoderSpec(
        alias="custom_hf",
        model_name=alias_or_model_name,
        embedding_dim=None,
        tokenizer_style="raw",
        architecture="auto",
        max_length=512,
        notes="custom Hugging Face encoder",
    )

