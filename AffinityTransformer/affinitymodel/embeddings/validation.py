"""Strict preflight validation for frozen token-embedding caches."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd
import yaml

from ..config import EncoderConfig
from ..utils import hash_text
from .schema import SequenceType
from .store import MANIFEST_COLUMNS, ShardedEmbeddingStore


@dataclass(frozen=True)
class CacheDescriptor:
    """Validated cache facts safe to use for model construction/checkpoints."""

    cache_dir: Path
    manifest_path: Path
    metadata_path: Path
    sequence_type: SequenceType
    encoder_name: str
    encoder_revision: str
    tokenizer_revision: str
    embedding_dim: int
    dtype: str
    metadata_hash: str
    n_items: int
    required_count: int
    covered_count: int
    sequence_length_summary: Mapping[str, float] = field(default_factory=dict)
    embedding_length_summary: Mapping[str, float] = field(default_factory=dict)
    truncated_count: int = 0
    truncation_rate: float = 0.0

    @property
    def coverage(self) -> float:
        return 1.0 if self.required_count == 0 else self.covered_count / self.required_count


def validate_embedding_cache(
    cache_dir: Path,
    encoder_config: EncoderConfig,
    sequence_type: SequenceType,
    required_sequence_hashes: Iterable[str] = (),
) -> CacheDescriptor:
    """Validate metadata, manifest consistency, coverage, and required shard items.

    This function is intended to run before model/optimizer construction. Every
    required item is loaded once through ``ShardedEmbeddingStore`` so manifest
    versus shard shape/dtype mismatches fail during preflight rather than in a
    later training batch.
    """
    if encoder_config.mode != "frozen_cached":
        raise ValueError("embedding cache validation requires encoder mode='frozen_cached'")
    cache_dir = Path(cache_dir)
    if encoder_config.cache_dir is None or cache_dir.resolve() != encoder_config.cache_dir.resolve():
        raise ValueError("cache_dir does not match EncoderConfig.cache_dir")
    metadata_path = cache_dir / "metadata.yaml"
    manifest_path = _resolve_manifest_path(cache_dir)
    if not metadata_path.exists():
        raise FileNotFoundError(f"embedding metadata not found: {metadata_path}")

    metadata = _read_metadata(metadata_path)
    extraction = metadata.get("extraction")
    if not isinstance(extraction, Mapping):
        raise ValueError("embedding metadata field 'extraction' must be a mapping")
    actual_name = _required_text(metadata, "encoder_name")
    actual_revision = _required_text(metadata, "encoder_revision")
    tokenizer_revision = _required_text(metadata, "tokenizer_revision")
    _require_equal("encoder_name", actual_name, encoder_config.name)
    _require_equal("encoder_revision", actual_revision, encoder_config.revision)
    _require_equal("tokenizer_revision", tokenizer_revision, encoder_config.tokenizer_revision)
    _require_equal(
        "embedding_layer", _required_value(extraction, "embedding_layer"), encoder_config.embedding_layer
    )
    _require_equal("max_length", _required_value(extraction, "max_length"), encoder_config.max_length)
    _require_equal(
        "long_sequence_strategy",
        _required_value(extraction, "long_sequence_strategy"),
        encoder_config.long_sequence_strategy,
    )

    manifest = _read_manifest(manifest_path)
    missing_columns = [column for column in MANIFEST_COLUMNS if column not in manifest.columns]
    if missing_columns:
        raise ValueError(f"embedding manifest is missing required column(s): {missing_columns}")
    selected = manifest[manifest["sequence_type"].astype(str) == sequence_type].copy()
    if selected.empty:
        raise ValueError(f"embedding manifest contains no {sequence_type} items")
    if selected["sequence_hash"].astype(str).duplicated().any():
        raise ValueError(f"embedding manifest contains duplicate {sequence_type} sequence hashes")
    _require_single_manifest_value(selected, "encoder_name", encoder_config.name)
    _require_single_manifest_value(selected, "encoder_revision", encoder_config.revision)
    embedding_dim = _require_positive_single_int(selected, "embedding_dim")
    dtype = _require_single_text(selected, "dtype")
    sequence_lengths = pd.to_numeric(selected["sequence_length"], errors="coerce")
    embedding_lengths = pd.to_numeric(selected["embedding_length"], errors="coerce")
    if sequence_lengths.isna().any() or (sequence_lengths < 1).any():
        raise ValueError("embedding manifest contains invalid sequence_length")
    if embedding_lengths.isna().any() or (embedding_lengths < 1).any():
        raise ValueError("embedding manifest contains invalid embedding_length")

    available = set(selected["sequence_hash"].astype(str))
    required = set(str(value) for value in required_sequence_hashes)
    missing = sorted(required - available)
    if missing:
        preview = missing[:10]
        raise ValueError(
            f"embedding cache coverage failure for {sequence_type}: missing "
            f"{len(missing)}/{len(required)} required sequence hash(es); first={preview}"
        )

    store = ShardedEmbeddingStore(manifest_path)
    for sequence_hash in sorted(required):
        item = store.get(sequence_hash, sequence_type)
        if item.values.shape[1] != embedding_dim:
            raise ValueError(
                f"embedding dimension mismatch for {sequence_type}/{sequence_hash}: "
                f"{item.values.shape[1]} != {embedding_dim}"
            )
        actual_dtype = str(item.values.dtype).removeprefix("torch.")
        if actual_dtype != dtype:
            raise ValueError(
                f"embedding dtype mismatch for {sequence_type}/{sequence_hash}: "
                f"{actual_dtype} != {dtype}"
            )

    measured = (
        selected
        if not required
        else selected[selected["sequence_hash"].astype(str).isin(required)]
    )
    measured_sequence_lengths = pd.to_numeric(measured["sequence_length"])
    measured_embedding_lengths = pd.to_numeric(measured["embedding_length"])
    truncated_count = int((measured_embedding_lengths < measured_sequence_lengths).sum())

    canonical_metadata = json.dumps(metadata, sort_keys=True, separators=(",", ":"), default=str)
    return CacheDescriptor(
        cache_dir=cache_dir,
        manifest_path=manifest_path,
        metadata_path=metadata_path,
        sequence_type=sequence_type,
        encoder_name=actual_name,
        encoder_revision=actual_revision,
        tokenizer_revision=tokenizer_revision,
        embedding_dim=embedding_dim,
        dtype=dtype,
        metadata_hash=hash_text(canonical_metadata),
        n_items=len(selected),
        required_count=len(required),
        covered_count=len(required),
        sequence_length_summary=_length_summary(measured_sequence_lengths),
        embedding_length_summary=_length_summary(measured_embedding_lengths),
        truncated_count=truncated_count,
        truncation_rate=truncated_count / len(measured),
    )


def _resolve_manifest_path(cache_dir: Path) -> Path:
    candidates = [cache_dir / "manifest.parquet", cache_dir / "manifest.csv"]
    existing = [path for path in candidates if path.exists()]
    if len(existing) != 1:
        raise FileNotFoundError(
            f"expected exactly one embedding manifest in {cache_dir}, found {existing}"
        )
    return existing[0]


def _read_metadata(path: Path) -> dict[str, object]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"embedding metadata must contain a mapping: {path}")
    return raw


def _read_manifest(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _required_text(mapping: Mapping[str, object], key: str) -> str:
    value = _required_value(mapping, key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"embedding metadata field {key!r} must be a non-empty string")
    return value


def _required_value(mapping: Mapping[str, object], key: str) -> object:
    if key not in mapping:
        raise ValueError(f"embedding metadata is missing required field {key!r}")
    return mapping[key]


def _require_equal(field: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise ValueError(f"embedding metadata mismatch for {field}: {actual!r} != {expected!r}")


def _require_single_manifest_value(table: pd.DataFrame, field: str, expected: str) -> None:
    values = set(table[field].astype(str))
    if values != {expected}:
        raise ValueError(f"embedding manifest mismatch for {field}: {sorted(values)} != {[expected]}")


def _require_positive_single_int(table: pd.DataFrame, field: str) -> int:
    numeric = pd.to_numeric(table[field], errors="coerce")
    if numeric.notna().any() and (numeric.dropna() % 1 != 0).any():
        raise ValueError(f"embedding manifest field {field} must contain integers")
    values = set(numeric.dropna().astype(int))
    if numeric.isna().any() or len(values) != 1 or next(iter(values)) < 1:
        raise ValueError(f"embedding manifest requires one positive {field}, got {sorted(values)}")
    return next(iter(values))


def _require_single_text(table: pd.DataFrame, field: str) -> str:
    values = set(table[field].astype(str))
    if len(values) != 1:
        raise ValueError(f"embedding manifest requires one {field}, got {sorted(values)}")
    return next(iter(values))


def _length_summary(values: pd.Series) -> dict[str, float]:
    return {
        "p50": float(values.quantile(0.50)),
        "p90": float(values.quantile(0.90)),
        "p95": float(values.quantile(0.95)),
        "max": float(values.max()),
    }
