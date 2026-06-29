"""Tests for strict frozen-cache preflight validation."""

from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from affinity_transformer.config import EncoderConfig
from affinity_transformer.embeddings import validate_embedding_cache


def _encoder(cache_dir: Path, *, revision: str = "encoder-rev") -> EncoderConfig:
    return EncoderConfig(
        name="fake-encoder",
        revision=revision,
        tokenizer_revision="tokenizer-rev",
        mode="frozen_cached",
        embedding_layer=-1,
        cache_dir=cache_dir,
        max_length=128,
        long_sequence_strategy="truncate",
    )


def _write_cache(cache_dir: Path, *, shape: tuple[int, int] = (3, 5)) -> None:
    cache_dir.mkdir()
    torch.save({"item": {"values": torch.ones(*shape)}}, cache_dir / "shard.pt")
    pd.DataFrame([
        {
            "sequence_hash": "required-hash",
            "sequence_type": "antibody",
            "encoder_name": "fake-encoder",
            "encoder_revision": "encoder-rev",
            "shard_path": "shard.pt",
            "item_key": "item",
            "sequence_length": 3,
            "embedding_length": 3,
            "embedding_dim": 5,
            "dtype": "float32",
        }
    ]).to_csv(cache_dir / "manifest.csv", index=False)
    metadata = {
        "encoder_name": "fake-encoder",
        "encoder_revision": "encoder-rev",
        "tokenizer_revision": "tokenizer-rev",
        "extraction": {
            "embedding_layer": -1,
            "max_length": 128,
            "long_sequence_strategy": "truncate",
        },
    }
    (cache_dir / "metadata.yaml").write_text(yaml.safe_dump(metadata), encoding="utf-8")


def test_validate_embedding_cache_returns_model_construction_facts(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    _write_cache(cache_dir)

    descriptor = validate_embedding_cache(
        cache_dir,
        _encoder(cache_dir),
        "antibody",
        ["required-hash"],
    )

    assert descriptor.embedding_dim == 5
    assert descriptor.dtype == "float32"
    assert descriptor.coverage == 1.0
    assert len(descriptor.metadata_hash) == 64
    assert descriptor.sequence_length_summary == {
        "p50": 3.0, "p90": 3.0, "p95": 3.0, "max": 3.0
    }
    assert descriptor.truncation_rate == 0.0


def test_validate_embedding_cache_rejects_revision_mismatch(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    _write_cache(cache_dir)

    with pytest.raises(ValueError, match="encoder_revision"):
        validate_embedding_cache(cache_dir, _encoder(cache_dir, revision="wrong"), "antibody")


def test_validate_embedding_cache_rejects_missing_required_hash(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    _write_cache(cache_dir)

    with pytest.raises(ValueError, match="coverage failure"):
        validate_embedding_cache(
            cache_dir,
            _encoder(cache_dir),
            "antibody",
            ["missing-hash"],
        )


def test_validate_embedding_cache_reads_required_shards_before_training(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    _write_cache(cache_dir, shape=(2, 5))

    with pytest.raises(ValueError, match="shape mismatch"):
        validate_embedding_cache(
            cache_dir,
            _encoder(cache_dir),
            "antibody",
            ["required-hash"],
        )


def test_validate_embedding_cache_reports_truncation_for_required_items(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    _write_cache(cache_dir, shape=(2, 5))
    manifest_path = cache_dir / "manifest.csv"
    manifest = pd.read_csv(manifest_path)
    manifest["embedding_length"] = 2
    manifest.to_csv(manifest_path, index=False)

    descriptor = validate_embedding_cache(
        cache_dir,
        _encoder(cache_dir),
        "antibody",
        ["required-hash"],
    )

    assert descriptor.embedding_length_summary["max"] == 2.0
    assert descriptor.truncated_count == 1
    assert descriptor.truncation_rate == 1.0
