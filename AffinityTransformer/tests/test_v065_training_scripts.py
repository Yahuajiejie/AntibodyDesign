"""Tests for the generated v0.65 cache/training Slurm workflow."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import yaml

from affinity_transformer.config import load_config

ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts/embeddings/build_v065_cache_and_configs.py"


def _load_builder_module():
    spec = importlib.util.spec_from_file_location("v065_cache_builder", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_write_training_configs_generates_concat_and_fixed_deep_depths(tmp_path: Path):
    builder = _load_builder_module()
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    split_paths = {}
    for split in ("train", "valid", "test"):
        path = split_dir / f"{split}.parquet"
        path.write_bytes(b"")
        split_paths[split] = path
    antibody_cache = tmp_path / "ab-cache"
    antigen_cache = tmp_path / "ag-cache"
    antibody_cache.mkdir()
    antigen_cache.mkdir()
    config_dir = tmp_path / "configs"

    paths = builder.write_training_configs(
        config_dir=config_dir,
        split_paths=split_paths,
        antibody_info={
            "model_name": "fake-ab",
            "model_revision": "ab-revision-1",
            "tokenizer_revision": "ab-tokenizer-1",
        },
        antigen_info={
            "model_name": "fake-ag",
            "model_revision": "ag-revision-1",
            "tokenizer_revision": "ag-tokenizer-1",
        },
        antibody_cache=antibody_cache,
        antigen_cache=antigen_cache,
        antibody_max_length=256,
        antigen_max_length=512,
        d_model=256,
        num_heads=8,
        batch_size=2,
        epochs=10,
        lr=1.0e-4,
        max_pairs_per_group=200,
        seed=0,
    )

    assert [path.name for path in paths] == [
        "v065_concat_ranknet.yaml",
        "v065_deep4_ranknet.yaml",
        "v065_deep8_ranknet.yaml",
        "v065_deep16_ranknet.yaml",
    ]
    expected = [("concat", 0), ("deep_cross_attention", 4),
                ("deep_cross_attention", 8), ("deep_cross_attention", 16)]
    for path, (kind, depth) in zip(paths, expected):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert raw["model"]["interaction"]["kind"] == kind
        assert raw["model"]["interaction"]["num_layers"] == depth
        assert raw["model"]["antibody_encoder"]["mode"] == "frozen_cached"
        assert raw["train"]["device"] == "cuda"
        config = load_config(path)
        assert config.model.interaction.num_layers == depth


def test_new_slurm_shell_scripts_pass_bash_syntax_check():
    paths = [
        ROOT / "scripts/slurm/download_v065_models_login.sh",
        ROOT / "scripts/slurm/check_v065_models.sbatch",
        ROOT / "scripts/slurm/build_v065_embedding_cache.sbatch",
        ROOT / "scripts/slurm/submit_v065_training_chain.sh",
    ]

    for path in paths:
        completed = subprocess.run(
            ["bash", "-n", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
