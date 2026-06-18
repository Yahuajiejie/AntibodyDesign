"""Offline embedding request collection and sharded cache generation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import torch
import yaml

from ..dataset import AffinityExample
from .extractors import EmbeddingExtractor
from .schema import (
    AntibodySequenceInput,
    EmbeddingRequest,
    antibody_embedding_request,
    antigen_embedding_request,
)


def collect_embedding_requests(
    examples: Iterable[AffinityExample],
    *,
    include_antibodies: bool = True,
    include_antigens: bool = True,
) -> list[EmbeddingRequest]:
    """Collect unique structured sequence requests in deterministic order."""
    if not include_antibodies and not include_antigens:
        raise ValueError("at least one request type must be enabled")
    requests: dict[tuple[str, str], EmbeddingRequest] = {}
    for example in examples:
        if include_antibodies:
            antibody = AntibodySequenceInput(
                heavy_chain=example.heavy_chain,
                light_chain=example.light_chain,
                single_chain_sequence=example.single_chain_sequence,
                antibody_type=example.antibody_type,
            )
            request = antibody_embedding_request(antibody)
            requests[(request.sequence_type, request.sequence_hash)] = request
        if include_antigens and example.antigen_sequence is not None:
            request = antigen_embedding_request(example.antigen_sequence)
            requests[(request.sequence_type, request.sequence_hash)] = request
    return [requests[key] for key in sorted(requests)]


def write_embedding_cache(
    requests: Sequence[EmbeddingRequest],
    extractor: EmbeddingExtractor,
    output_dir: Path,
    *,
    shard_size: int = 256,
) -> Path:
    """Encode requests and write manifest-indexed tensor shards.

    Existing manifests are rejected so a cache cannot silently mix encoder
    revisions or extraction rules.
    """
    if shard_size < 1:
        raise ValueError("shard_size must be >= 1")
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.parquet"
    metadata_path = output_dir / "metadata.yaml"
    if manifest_path.exists() or metadata_path.exists():
        raise FileExistsError(
            f"embedding cache already exists in {output_dir}; use a new revision directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(exist_ok=True)

    unique = _unique_requests(requests)
    manifest_rows: list[dict[str, object]] = []
    for shard_index, start in enumerate(range(0, len(unique), shard_size)):
        chunk = unique[start : start + shard_size]
        encoded = extractor.encode(chunk)
        expected = {request.sequence_hash for request in chunk}
        actual = set(encoded)
        if actual != expected:
            raise ValueError(
                "extractor output keys do not match requests: "
                f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
            )

        shard_name = f"shard_{shard_index:05d}.pt"
        shard_payload: dict[str, dict[str, torch.Tensor]] = {}
        for request in chunk:
            item = encoded[request.sequence_hash]
            item_key = request.sequence_hash
            shard_payload[item_key] = {
                "values": item.values,
                "mask": item.mask,
            }
            manifest_rows.append({
                "sequence_hash": request.sequence_hash,
                "sequence_type": request.sequence_type,
                "encoder_name": extractor.encoder_name,
                "encoder_revision": extractor.encoder_revision,
                "shard_path": f"shards/{shard_name}",
                "item_key": item_key,
                "sequence_length": _request_sequence_length(request),
                "embedding_length": item.values.shape[0],
                "embedding_dim": item.values.shape[1],
                "dtype": str(item.values.dtype).removeprefix("torch."),
            })
        torch.save(shard_payload, shard_dir / shard_name)

    pd.DataFrame(manifest_rows, columns=[
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
    ]).to_parquet(manifest_path, index=False)
    metadata = {
        "cache_schema": "token_embedding_cache_v1",
        "encoder_name": extractor.encoder_name,
        "encoder_revision": extractor.encoder_revision,
        "tokenizer_revision": getattr(
            extractor, "tokenizer_revision", extractor.encoder_revision
        ),
        "n_items": len(unique),
        "shard_size": shard_size,
        "sequence_type_counts": {
            sequence_type: sum(request.sequence_type == sequence_type for request in unique)
            for sequence_type in ("antibody", "antigen")
        },
        "extraction": dict(extractor.metadata()),
    }
    metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=True), encoding="utf-8")
    return manifest_path


def _unique_requests(requests: Sequence[EmbeddingRequest]) -> list[EmbeddingRequest]:
    unique: dict[tuple[str, str], EmbeddingRequest] = {}
    for request in requests:
        key = (request.sequence_type, request.sequence_hash)
        existing = unique.get(key)
        if existing is not None and existing != request:
            raise ValueError(f"conflicting requests share key={key}")
        unique[key] = request
    return [unique[key] for key in sorted(unique)]


def _request_sequence_length(request: EmbeddingRequest) -> int:
    if request.sequence_type == "antigen":
        assert request.antigen_sequence is not None
        return len(request.antigen_sequence)
    assert request.antibody is not None
    antibody = request.antibody
    if antibody.single_chain_sequence is not None:
        return len(antibody.single_chain_sequence)
    return sum(len(chain) for chain in (antibody.heavy_chain, antibody.light_chain) if chain)
