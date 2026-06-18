#!/usr/bin/env python3
"""Build formal v0.65 IgBERT/ESM-2 caches and generated training configs."""

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
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--antibody-max-length", type=int, default=256)
    parser.add_argument("--antigen-max-length", type=int, default=512)
    parser.add_argument("--antibody-shard-size", type=int, default=32)
    parser.add_argument("--antigen-shard-size", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--max-pairs-per-group", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true", default=True)
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

    write_training_configs(
        config_dir=args.config_dir,
        split_paths=split_paths,
        antibody_info=antibody_info,
        antigen_info=antigen_info,
        antibody_cache=antibody_cache,
        antigen_cache=antigen_cache,
        antibody_max_length=args.antibody_max_length,
        antigen_max_length=args.antigen_max_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        max_pairs_per_group=args.max_pairs_per_group,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )


def write_training_configs(
    *,
    config_dir: Path,
    split_paths: dict[str, Path],
    antibody_info: dict[str, str],
    antigen_info: dict[str, str],
    antibody_cache: Path,
    antigen_cache: Path,
    antibody_max_length: int,
    antigen_max_length: int,
    d_model: int,
    num_heads: int,
    batch_size: int,
    epochs: int,
    lr: float,
    max_pairs_per_group: int,
    seed: int,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> list[Path]:
    """Write one Concat and three fixed-depth Deep RankNet configs."""
    if d_model < 1 or num_heads < 1 or d_model % num_heads != 0:
        raise ValueError("d_model must be positive and divisible by num_heads")
    config_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "data": {
            "all_records_path": None,
            "train_path": str(split_paths["train"]),
            "valid_path": str(split_paths["valid"]),
            "test_path": str(split_paths["test"]),
            "split_strategy": "none",
            "split_dir": None,
            "valid_fraction": 0.1,
            "test_fraction": 0.1,
            "max_pairs_per_group": max_pairs_per_group,
            "pair_sample_strategy": "absolute_cap",
            "pair_fraction": None,
            "min_pairs_per_group": 1,
            "seed": seed,
        },
        "model": {
            "antibody_encoder": _encoder_mapping(
                antibody_info, antibody_cache, antibody_max_length
            ),
            "antigen_encoder": _encoder_mapping(
                antigen_info, antigen_cache, antigen_max_length
            ),
            "objective": {
                "name": "pairwise_ranknet",
                "temperature": 1.0,
                "sigma": 1.0,
                "pointwise_loss": "huber",
            },
        },
        "train": {
            "batch_size": batch_size,
            "lr": lr,
            "epochs": epochs,
            "device": "cuda",
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        },
    }
    variants = [("concat", 0), *[("deep_cross_attention", n) for n in (4, 8, 16)]]
    written = []
    for kind, num_layers in variants:
        payload = yaml.safe_load(yaml.safe_dump(common))
        payload["model"]["interaction"] = {
            "kind": kind,
            "d_model": d_model,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "ffn_multiplier": 4.0,
            "dropout": 0.1,
            "pooling": "masked_mean",
            "bidirectional": True,
        }
        name = "concat" if kind == "concat" else f"deep{num_layers}"
        path = config_dir / f"v065_{name}_ranknet.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        written.append(path)
        print(f"wrote config: {path}")
    return written


def _load_revisions(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"model revision file not found: {path}; run download_v065_models_login.sh first"
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


def _encoder_mapping(
    model_info: dict[str, str], cache_dir: Path, max_length: int
) -> dict[str, object]:
    return {
        "name": model_info["model_name"],
        "revision": model_info["model_revision"],
        "tokenizer_revision": model_info["tokenizer_revision"],
        "mode": "frozen_cached",
        "embedding_layer": -1,
        "cache_dir": str(cache_dir),
        "max_length": max_length,
        "long_sequence_strategy": "truncate",
        "lora_rank": None,
        "lora_alpha": None,
        "lora_dropout": None,
    }


def _release_device_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
