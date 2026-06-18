"""Embedding cache lookup interfaces and implementations."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Mapping, Protocol

import pandas as pd
import torch

from .schema import EmbeddingItem, SequenceType

MANIFEST_COLUMNS = (
    "sequence_hash",
    "sequence_type",
    "encoder_name",
    "encoder_revision",
    "shard_path",
    "item_key",
    "sequence_length",
    "embedding_length",
    "embedding_dim",
    "dtype",
)


class EmbeddingNotFoundError(KeyError):
    """Raised when a required sequence is absent from an embedding store."""


class EmbeddingStore(Protocol):
    """Minimal cache interface consumed by embedding dataloaders."""

    def get(self, sequence_hash: str, sequence_type: SequenceType) -> EmbeddingItem:
        """Return one cached item or raise ``EmbeddingNotFoundError``."""
        ...


class InMemoryEmbeddingStore:
    """Small store used by tests and programmatic callers."""

    def __init__(
        self,
        items: Mapping[tuple[SequenceType, str], EmbeddingItem] | None = None,
    ) -> None:
        self._items = dict(items or {})

    def put(
        self,
        sequence_hash: str,
        sequence_type: SequenceType,
        item: EmbeddingItem,
    ) -> None:
        """Insert or replace one item."""
        self._items[(sequence_type, sequence_hash)] = item

    def get(self, sequence_hash: str, sequence_type: SequenceType) -> EmbeddingItem:
        """Return one item from memory."""
        try:
            return self._items[(sequence_type, sequence_hash)]
        except KeyError as exc:
            raise EmbeddingNotFoundError(
                f"missing {sequence_type} embedding for sequence_hash={sequence_hash}"
            ) from exc


class ShardedEmbeddingStore:
    """Lazy reader for manifest-indexed ``torch.save`` embedding shards."""

    def __init__(self, manifest_path: Path, max_cached_shards: int = 2) -> None:
        if max_cached_shards < 1:
            raise ValueError("max_cached_shards must be >= 1")
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"embedding manifest not found: {self.manifest_path}")
        manifest = _read_manifest(self.manifest_path)
        missing = [column for column in MANIFEST_COLUMNS if column not in manifest.columns]
        if missing:
            raise ValueError(f"embedding manifest is missing required column(s): {missing}")

        self._rows: dict[tuple[str, str], dict[str, object]] = {}
        for row in manifest.to_dict(orient="records"):
            sequence_type = str(row["sequence_type"])
            if sequence_type not in {"antibody", "antigen"}:
                raise ValueError(f"invalid manifest sequence_type: {sequence_type!r}")
            key = (sequence_type, str(row["sequence_hash"]))
            if key in self._rows:
                raise ValueError(f"duplicate embedding manifest key: {key}")
            self._rows[key] = row

        self._max_cached_shards = max_cached_shards
        self._shards: OrderedDict[Path, Mapping[str, object]] = OrderedDict()

    def get(self, sequence_hash: str, sequence_type: SequenceType) -> EmbeddingItem:
        """Load one item, retaining a small per-process shard LRU cache."""
        key = (sequence_type, sequence_hash)
        row = self._rows.get(key)
        if row is None:
            raise EmbeddingNotFoundError(
                f"missing {sequence_type} embedding for sequence_hash={sequence_hash} "
                f"in {self.manifest_path}"
            )

        shard_path = Path(str(row["shard_path"]))
        if not shard_path.is_absolute():
            shard_path = self.manifest_path.parent / shard_path
        shard = self._load_shard(shard_path)
        item_key = str(row["item_key"])
        if item_key not in shard:
            raise EmbeddingNotFoundError(
                f"item_key {item_key!r} is absent from embedding shard {shard_path}"
            )
        item = _coerce_item(shard[item_key])
        expected_dim = int(row["embedding_dim"])
        expected_length = int(row["embedding_length"])
        if item.values.shape != (expected_length, expected_dim):
            raise ValueError(
                f"embedding shape mismatch for {key}: manifest "
                f"({expected_length}, {expected_dim}), shard {tuple(item.values.shape)}"
            )
        expected_dtype = str(row["dtype"])
        actual_dtype = str(item.values.dtype).removeprefix("torch.")
        if actual_dtype != expected_dtype:
            raise ValueError(
                f"embedding dtype mismatch for {key}: manifest {expected_dtype}, "
                f"shard {actual_dtype}"
            )
        return item

    def _load_shard(self, path: Path) -> Mapping[str, object]:
        if path in self._shards:
            shard = self._shards.pop(path)
            self._shards[path] = shard
            return shard
        if not path.exists():
            raise FileNotFoundError(f"embedding shard not found: {path}")
        try:
            shard = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            shard = torch.load(path, map_location="cpu")
        if not isinstance(shard, Mapping):
            raise ValueError(f"embedding shard must contain a mapping: {path}")
        self._shards[path] = shard
        while len(self._shards) > self._max_cached_shards:
            self._shards.popitem(last=False)
        return shard


def _read_manifest(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"embedding manifest must be .parquet or .csv: {path}")


def _coerce_item(raw: object) -> EmbeddingItem:
    if isinstance(raw, EmbeddingItem):
        return raw
    if isinstance(raw, torch.Tensor):
        return EmbeddingItem.from_values(raw)
    if isinstance(raw, Mapping) and "values" in raw:
        values = raw["values"]
        mask = raw.get("mask")
        if not isinstance(values, torch.Tensor):
            raise ValueError("embedding shard item 'values' must be a tensor")
        if mask is not None and not isinstance(mask, torch.Tensor):
            raise ValueError("embedding shard item 'mask' must be a tensor")
        return EmbeddingItem.from_values(values, mask)
    raise ValueError(f"unsupported embedding shard item type: {type(raw).__name__}")
