#!/usr/bin/env python3
"""Build and validate formal group-holdout IgBERT/ESM-2 embedding caches.

This command never creates or modifies experiment YAML files.  Training
configuration is human-owned under ``configs/``.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Iterable

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from affinity_transformer.config import EncoderConfig
from affinity_transformer.dataset import AffinityRecordDataset, filter_trainable_records, load_records
from affinity_transformer.embeddings import (
    EmbeddingRequest,
    build_embedding_extractor,
    collect_embedding_requests,
    validate_embedding_cache,
    write_embedding_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--revision-file", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--antibody-max-length", type=int, default=256)
    parser.add_argument("--antigen-max-length", type=int, default=512)
    parser.add_argument("--antibody-shard-size", type=int, default=32)
    parser.add_argument("--antigen-shard-size", type=int, default=4)
    args = parser.parse_args()

    split_paths = {
        split: args.split_dir / f"{split}.parquet"
        for split in ("train", "valid", "test")
    }
    missing = [str(path) for path in split_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing split file(s): {missing}")
    revisions = _load_revisions(args.revision_file)
    requests = _collect_requests(split_paths.values())
    antibody_requests = [r for r in requests if r.sequence_type == "antibody"]
    antigen_requests = [r for r in requests if r.sequence_type == "antigen"]
    if not antibody_requests or not antigen_requests:
        raise ValueError("formal dual-cache training requires antibody and antigen requests")

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    antibody_info = revisions["antibody"]
    antigen_info = revisions["antigen"]
    antibody_cache = args.cache_root / "igbert" / antibody_info["model_revision"]
    antigen_cache = args.cache_root / "esm2" / antigen_info["model_revision"]

    _build_cache_if_missing(
        requests=antibody_requests,
        extractor_name="igbert",
        model_info=antibody_info,
        output_dir=antibody_cache,
        device=args.device,
        dtype=dtype,
        max_length=args.antibody_max_length,
        shard_size=args.antibody_shard_size,
    )
    _release_device_cache()
    _build_cache_if_missing(
        requests=antigen_requests,
        extractor_name="esm2",
        model_info=antigen_info,
        output_dir=antigen_cache,
        device=args.device,
        dtype=dtype,
        max_length=args.antigen_max_length,
        shard_size=args.antigen_shard_size,
    )
    _release_device_cache()

    antibody_config = _encoder_config(antibody_info, antibody_cache, args.antibody_max_length)
    antigen_config = _encoder_config(antigen_info, antigen_cache, args.antigen_max_length)
    antibody_descriptor = validate_embedding_cache(
        antibody_cache,
        antibody_config,
        "antibody",
        [request.sequence_hash for request in antibody_requests],
    )
    antigen_descriptor = validate_embedding_cache(
        antigen_cache,
        antigen_config,
        "antigen",
        [request.sequence_hash for request in antigen_requests],
    )
    print(
        f"validated antibody cache: n={antibody_descriptor.n_items}, "
        f"dim={antibody_descriptor.embedding_dim}, coverage={antibody_descriptor.coverage:.3f}"
    )
    print(
        f"validated antigen cache: n={antigen_descriptor.n_items}, "
        f"dim={antigen_descriptor.embedding_dim}, coverage={antigen_descriptor.coverage:.3f}"
    )

def _load_revisions(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            "model revision file not found: "
            f"{path}; run download_group_holdout_models_login.sh first"
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"revision file must contain a mapping: {path}")
    result = {}
    for role in ("antibody", "antigen"):
        value = raw.get(role)
        if not isinstance(value, dict):
            raise ValueError(f"revision file is missing mapping {role!r}")
        required = ("model_name", "model_revision", "tokenizer_revision")
        missing = [key for key in required if not value.get(key)]
        if missing:
            raise ValueError(f"revision file {role!r} is missing {missing}")
        result[role] = {key: str(value[key]) for key in required}
    return result


def _collect_requests(paths: Iterable[Path]) -> list[EmbeddingRequest]:
    unique: dict[tuple[str, str], EmbeddingRequest] = {}
    for path in paths:
        records = filter_trainable_records(load_records(path))
        dataset = AffinityRecordDataset(records)
        requests = collect_embedding_requests(dataset[index] for index in range(len(dataset)))
        for request in requests:
            unique[(request.sequence_type, request.sequence_hash)] = request
        print(f"collected requests from {path}: records={len(dataset)}, unique_total={len(unique)}")
    return [unique[key] for key in sorted(unique)]


def _build_cache_if_missing(
    *,
    requests: list[EmbeddingRequest],
    extractor_name: str,
    model_info: dict[str, str],
    output_dir: Path,
    device: str,
    dtype: torch.dtype,
    max_length: int,
    shard_size: int,
) -> None:
    manifest_path = output_dir / "manifest.parquet"
    metadata_path = output_dir / "metadata.yaml"
    if manifest_path.exists() and metadata_path.exists():
        print(f"reuse existing cache: {output_dir}")
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"partial cache directory exists: {output_dir}; move it aside and resubmit"
        )
    print(
        f"building {extractor_name} cache: requests={len(requests)}, "
        f"max_length={max_length}, shard_size={shard_size}, output={output_dir}"
    )
    extractor = build_embedding_extractor(
        extractor_name,
        model_name=model_info["model_name"],
        revision=model_info["model_revision"],
        tokenizer_revision=model_info["tokenizer_revision"],
        device=device,
        output_dtype=dtype,
        max_length=max_length,
        long_sequence_strategy="truncate",
    )
    write_embedding_cache(requests, extractor, output_dir, shard_size=shard_size)
    del extractor


def _encoder_config(
    model_info: dict[str, str], cache_dir: Path, max_length: int
) -> EncoderConfig:
    return EncoderConfig(
        name=model_info["model_name"],
        revision=model_info["model_revision"],
        tokenizer_revision=model_info["tokenizer_revision"],
        mode="frozen_cached",
        embedding_layer=-1,
        cache_dir=cache_dir,
        max_length=max_length,
        long_sequence_strategy="truncate",
    )


def _release_device_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
