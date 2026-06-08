"""
encoders/sequence.py — 可插拔序列 encoder

这个模块负责把氨基酸序列转成固定长度 embedding。它支持：
  - ESM2 这类通用蛋白模型；
  - IgBert 这类 antibody BERT 模型；
  - IgT5 这类 paired antibody T5 encoder；
  - 任意 Hugging Face AutoModel/T5EncoderModel 自定义模型。

注意：本模块只在被调用时加载 torch/transformers，不影响轻量数据质检。
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from typing import Iterable

import numpy as np

from ..antigen_schema import clean_text, normalize_antigen_sequence
from ..config import SequenceEncoderSpec, get_encoder_spec


def sequence_hash(sequence: str, encoder_name: str) -> str:
    """
    根据序列和 encoder 名称生成 cache key。

    同一条序列用不同 encoder 计算出的 embedding 不能混用，所以 hash 中包含
    encoder_name。
    """
    normalized = normalize_antigen_sequence(sequence)
    key = f"{encoder_name}|{normalized}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def paired_sequence_hash(heavy: str, light: str, encoder_name: str) -> str:
    """为 heavy/light paired encoder 生成 cache key。"""
    key = "|".join([
        encoder_name,
        normalize_antigen_sequence(heavy),
        normalize_antigen_sequence(light),
    ])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def format_single_sequence(sequence: str, tokenizer_style: str) -> str:
    """
    按 tokenizer 约定格式化单条序列。

    raw:
      "EVQL..."

    space:
      "E V Q L ..."
    """
    seq = normalize_antigen_sequence(sequence)
    if tokenizer_style == "raw":
        return seq
    if tokenizer_style in {"space", "paired_t5"}:
        return " ".join(seq)
    raise ValueError(f"未知 tokenizer_style={tokenizer_style!r}")


def format_paired_antibody_sequence(
    heavy: str,
    light: str,
    tokenizer_style: str,
) -> str:
    """
    格式化 heavy/light paired antibody 输入。

    IgT5 model card 使用：
      "V Q ... S </s> E V ... K"

    对 raw tokenizer，则用 linker 拼接，作为兜底。
    """
    heavy_seq = normalize_antigen_sequence(heavy)
    light_seq = normalize_antigen_sequence(light)
    if tokenizer_style == "paired_t5":
        if light_seq:
            return f"{' '.join(heavy_seq)} </s> {' '.join(light_seq)}"
        return " ".join(heavy_seq)
    if tokenizer_style == "space":
        if light_seq:
            return f"{' '.join(heavy_seq)} {' '.join(light_seq)}"
        return " ".join(heavy_seq)
    if tokenizer_style == "raw":
        linker = "GGGGSGGGGSGGGGS"
        return heavy_seq + (linker + light_seq if light_seq else "")
    raise ValueError(f"未知 tokenizer_style={tokenizer_style!r}")


class HuggingFaceSequenceEncoder:
    """
    Hugging Face 序列 encoder 包装。

    参数：
      spec:   SequenceEncoderSpec
      device: cuda/cpu；None 时自动选择

    输出：
      encode_texts() 返回 [N, hidden_dim] 的 numpy float32 矩阵。
    """

    def __init__(self, spec: SequenceEncoderSpec | str, device: str | None = None):
        self.spec = get_encoder_spec(spec) if isinstance(spec, str) else spec
        self.device = device
        self.tokenizer = None
        self.model = None
        self.hidden_size: int | None = self.spec.embedding_dim

    def load(self) -> None:
        """懒加载 tokenizer/model。"""
        if self.model is not None:
            return

        import torch
        from transformers import AutoModel, AutoTokenizer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.spec.architecture == "t5_encoder":
            from transformers import T5EncoderModel, T5Tokenizer

            self.tokenizer = T5Tokenizer.from_pretrained(
                self.spec.model_name,
                do_lower_case=False,
            )
            self.model = T5EncoderModel.from_pretrained(self.spec.model_name)
        elif self.spec.architecture == "auto":
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.spec.model_name,
                trust_remote_code=self.spec.trust_remote_code,
            )
            self.model = AutoModel.from_pretrained(
                self.spec.model_name,
                trust_remote_code=self.spec.trust_remote_code,
            )
        else:
            raise ValueError(f"未知 architecture={self.spec.architecture!r}")

        self.model.to(self.device)
        self.model.eval()
        self.hidden_size = int(getattr(self.model.config, "hidden_size", 0) or 0)
        if self.hidden_size <= 0 and hasattr(self.model.config, "d_model"):
            self.hidden_size = int(self.model.config.d_model)

    def encode_texts(
        self,
        texts: list[str],
        batch_size: int = 16,
    ) -> np.ndarray:
        """
        编码已经格式化好的文本序列。

        实现：
          1. tokenizer 返回 input_ids/attention_mask/special_tokens_mask；
          2. model 输出 last_hidden_state；
          3. 对非特殊 token、非 padding token 做 mean pooling。
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        self.load()

        import torch

        assert self.tokenizer is not None
        assert self.model is not None
        outputs: list[np.ndarray] = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            tokens = self.tokenizer(
                batch_texts,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=self.spec.max_length,
                return_tensors="pt",
                return_special_tokens_mask=True,
            )
            tokens = {key: value.to(self.device) for key, value in tokens.items()}
            with torch.no_grad():
                model_inputs = {
                    key: value
                    for key, value in tokens.items()
                    if key != "special_tokens_mask"
                }
                model_output = self.model(**model_inputs)
                hidden = model_output.last_hidden_state

            attention_mask = tokens["attention_mask"].bool()
            special_mask = tokens.get("special_tokens_mask")
            if special_mask is not None:
                valid_mask = attention_mask & ~special_mask.bool()
            else:
                valid_mask = attention_mask

            valid_mask_f = valid_mask.unsqueeze(-1).to(hidden.dtype)
            summed = (hidden * valid_mask_f).sum(dim=1)
            lengths = valid_mask_f.sum(dim=1).clamp(min=1.0)
            pooled = summed / lengths
            outputs.append(pooled.cpu().float().numpy())

        return np.concatenate(outputs, axis=0).astype(np.float32)

    def encode_sequences(
        self,
        sequences: Iterable[str],
        batch_size: int = 16,
    ) -> np.ndarray:
        """编码单链/单蛋白序列。"""
        texts = [
            format_single_sequence(seq, self.spec.tokenizer_style)
            for seq in sequences
        ]
        return self.encode_texts(texts, batch_size=batch_size)

    def encode_paired_antibodies(
        self,
        heavy_sequences: Iterable[str],
        light_sequences: Iterable[str],
        batch_size: int = 16,
    ) -> np.ndarray:
        """编码 heavy/light paired antibody 序列。"""
        texts = [
            format_paired_antibody_sequence(heavy, light, self.spec.tokenizer_style)
            for heavy, light in zip(heavy_sequences, light_sequences)
        ]
        return self.encode_texts(texts, batch_size=batch_size)

    def metadata(self) -> dict[str, object]:
        """返回 encoder 元数据，供 cache manifest 使用。"""
        return {
            **asdict(self.spec),
            "resolved_hidden_size": self.hidden_size,
            "device": self.device,
        }


def cache_paths(cache_dir: str, cache_key: str) -> tuple[str, str]:
    """返回 embedding npy 和 manifest json 路径。"""
    return (
        os.path.join(cache_dir, f"{cache_key}.npy"),
        os.path.join(cache_dir, f"{cache_key}.manifest.json"),
    )


def save_sequence_embedding(
    embedding: np.ndarray,
    cache_dir: str,
    cache_key: str,
    encoder: HuggingFaceSequenceEncoder,
    source: str,
) -> None:
    """保存序列 embedding 及其 manifest。"""
    os.makedirs(cache_dir, exist_ok=True)
    emb_path, manifest_path = cache_paths(cache_dir, cache_key)
    np.save(emb_path, np.asarray(embedding, dtype=np.float32))
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "cache_key": cache_key,
                "source": source,
                "encoder": encoder.metadata(),
                "embedding_dim": int(np.asarray(embedding).shape[-1]),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


def load_cached_sequence_embedding(
    cache_dir: str,
    cache_key: str,
) -> np.ndarray | None:
    """读取单条序列 embedding cache；不存在时返回 None。"""
    emb_path, manifest_path = cache_paths(cache_dir, cache_key)
    if not (os.path.exists(emb_path) and os.path.exists(manifest_path)):
        return None
    return np.load(emb_path).astype(np.float32)


def get_or_compute_sequence_embeddings(
    sequences: list[str],
    encoder: HuggingFaceSequenceEncoder,
    cache_dir: str,
    batch_size: int = 16,
) -> dict[str, np.ndarray]:
    """
    对一组单序列做缓存化编码。

    返回：
      normalized_sequence -> embedding。
    """
    unique = sorted({normalize_antigen_sequence(seq) for seq in sequences if clean_text(seq)})
    result: dict[str, np.ndarray] = {}
    missing: list[str] = []

    for seq in unique:
        key = sequence_hash(seq, encoder.spec.model_name)
        cached = load_cached_sequence_embedding(cache_dir, key)
        if cached is None:
            missing.append(seq)
        else:
            result[seq] = cached

    if missing:
        computed = encoder.encode_sequences(missing, batch_size=batch_size)
        for seq, emb in zip(missing, computed):
            key = sequence_hash(seq, encoder.spec.model_name)
            save_sequence_embedding(emb, cache_dir, key, encoder, source=seq)
            result[seq] = emb

    return result
