"""Tests for the human-owned group-holdout config and cache-only Slurm workflow."""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts/embeddings/build_group_holdout_cache.py"
CONFIG_DIR = ROOT / "configs/group_holdout"


def test_group_holdout_configs_are_static_human_owned_files():
    paths = sorted(CONFIG_DIR.glob("*.yaml"))
    assert [path.name for path in paths] == [
        "architecture_concat.yaml",
        "architecture_cross_attention_l16.yaml",
        "architecture_cross_attention_l4.yaml",
        "architecture_cross_attention_l8.yaml",
        "sampler_balanced_tree_l4.yaml",
        "sampler_noise_aware_multiscale_l4.yaml",
        "sampler_randomized_bst_l4.yaml",
    ]
    expected = {
        "architecture_concat.yaml": ("concat", 0, "capped_proportional"),
        "architecture_cross_attention_l4.yaml": ("deep_cross_attention", 4, "capped_proportional"),
        "architecture_cross_attention_l8.yaml": ("deep_cross_attention", 8, "capped_proportional"),
        "architecture_cross_attention_l16.yaml": ("deep_cross_attention", 16, "capped_proportional"),
        "sampler_balanced_tree_l4.yaml": ("deep_cross_attention", 4, "balanced_tree"),
        "sampler_randomized_bst_l4.yaml": ("deep_cross_attention", 4, "randomized_bst"),
        "sampler_noise_aware_multiscale_l4.yaml": (
            "deep_cross_attention",
            4,
            "noise_aware_multiscale",
        ),
    }
    for path in paths:
        kind, depth, sampler = expected[path.name]
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert raw["model"]["interaction"]["kind"] == kind
        assert raw["model"]["interaction"]["num_layers"] == depth
        assert raw["data"]["pair_sample_strategy"] == sampler
        assert raw["model"]["antibody_encoder"]["mode"] == "frozen_cached"
        assert "processed/embeddings/group_holdout" in raw["model"]["antibody_encoder"]["cache_dir"]
        assert "processed/embeddings/group_holdout" in raw["model"]["antigen_encoder"]["cache_dir"]
        assert raw["train"]["device"] == "cuda"
        assert raw["train"]["batch_size"] == 128
        assert raw["train"]["num_workers"] >= 1
        assert raw["train"]["pin_memory"] is True


def test_cache_builder_has_no_training_config_write_path():
    builder = BUILDER_PATH.read_text(encoding="utf-8")
    sbatch = (
        ROOT / "scripts/slurm/build_group_holdout_embedding_cache.sbatch"
    ).read_text(encoding="utf-8")

    assert "write_training_configs" not in builder
    assert "--config-dir" not in builder
    assert "--batch-size" not in builder
    assert "CONFIG_DIR" not in sbatch
    assert "generated configs" not in sbatch


def test_submit_chain_uses_static_config_directory():
    script = (
        ROOT / "scripts/slurm/submit_group_holdout_training_chain.sh"
    ).read_text(encoding="utf-8")
    assert 'CONFIG_DIR="${CONFIG_DIR:-configs/group_holdout}"' in script
    assert "human-owned training config" in script


def test_new_slurm_shell_scripts_pass_bash_syntax_check():
    paths = [
        ROOT / "scripts/slurm/download_group_holdout_models_login.sh",
        ROOT / "scripts/slurm/check_group_holdout_models.sbatch",
        ROOT / "scripts/slurm/build_group_holdout_embedding_cache.sbatch",
        ROOT / "scripts/slurm/submit_group_holdout_training_chain.sh",
        ROOT / "scripts/runs/group_holdout_formal_controls.sh",
    ]

    for path in paths:
        completed = subprocess.run(
            ["bash", "-n", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
