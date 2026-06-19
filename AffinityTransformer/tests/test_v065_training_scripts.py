"""Tests for the human-owned v0.65 config and cache-only Slurm workflow."""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts/embeddings/build_v065_cache.py"
CONFIG_DIR = ROOT / "configs/v065"


def test_v065_configs_are_static_human_owned_files():
    paths = sorted(CONFIG_DIR.glob("v065_*_ranknet.yaml"))
    assert [path.name for path in paths] == [
        "v065_concat_ranknet.yaml",
        "v065_deep16_ranknet.yaml",
        "v065_deep4_ranknet.yaml",
        "v065_deep8_ranknet.yaml",
    ]
    expected = {
        "v065_concat_ranknet.yaml": ("concat", 0),
        "v065_deep4_ranknet.yaml": ("deep_cross_attention", 4),
        "v065_deep8_ranknet.yaml": ("deep_cross_attention", 8),
        "v065_deep16_ranknet.yaml": ("deep_cross_attention", 16),
    }
    for path in paths:
        kind, depth = expected[path.name]
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert raw["model"]["interaction"]["kind"] == kind
        assert raw["model"]["interaction"]["num_layers"] == depth
        assert raw["model"]["antibody_encoder"]["mode"] == "frozen_cached"
        assert raw["train"]["device"] == "cuda"
        assert raw["train"]["batch_size"] > 1, "batch_size must be > 1; check for debug placeholder"
        assert raw["train"]["num_workers"] >= 1
        assert raw["train"]["pin_memory"] is True


def test_cache_builder_has_no_training_config_write_path():
    builder = BUILDER_PATH.read_text(encoding="utf-8")
    sbatch = (ROOT / "scripts/slurm/build_v065_embedding_cache.sbatch").read_text(
        encoding="utf-8"
    )

    assert "write_training_configs" not in builder
    assert "--config-dir" not in builder
    assert "--batch-size" not in builder
    assert "CONFIG_DIR" not in sbatch
    assert "generated configs" not in sbatch


def test_submit_chain_uses_static_config_directory():
    script = (ROOT / "scripts/slurm/submit_v065_training_chain.sh").read_text(
        encoding="utf-8"
    )
    assert 'CONFIG_DIR="${CONFIG_DIR:-configs/v065}"' in script
    assert "human-owned training config" in script


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
