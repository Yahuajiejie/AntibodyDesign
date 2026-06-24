"""Tests for the formal group-holdout control scripts and configs."""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/v065"
BUILDER_PATH = ROOT / "scripts/embeddings/build_v065_cache.py"


def _load_yaml(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))


def test_group_holdout_control_configs_cover_expected_experiments():
    expected = {
        "v065_concat_ranknet.yaml": ("concat", 0, "capped_proportional"),
        "v065_deep4_ranknet.yaml": ("deep_cross_attention", 4, "capped_proportional"),
        "v065_deep8_ranknet.yaml": ("deep_cross_attention", 8, "capped_proportional"),
        "v065_deep16_ranknet.yaml": ("deep_cross_attention", 16, "capped_proportional"),
        "v065_concat_randomized_bst_no_redundancy.yaml": ("concat", 0, "randomized_bst"),
        "v065_deep4_randomized_bst_no_redundancy.yaml": (
            "deep_cross_attention",
            4,
            "randomized_bst",
        ),
        "v065_deep8_randomized_bst_no_redundancy.yaml": (
            "deep_cross_attention",
            8,
            "randomized_bst",
        ),
        "v065_deep16_randomized_bst_no_redundancy.yaml": (
            "deep_cross_attention",
            16,
            "randomized_bst",
        ),
        "v065_concat_noise_aware_multiscale.yaml": (
            "concat",
            0,
            "noise_aware_multiscale",
        ),
        "v065_deep4_noise_aware_multiscale.yaml": (
            "deep_cross_attention",
            4,
            "noise_aware_multiscale",
        ),
        "v065_deep8_noise_aware_multiscale.yaml": (
            "deep_cross_attention",
            8,
            "noise_aware_multiscale",
        ),
        "v065_deep16_noise_aware_multiscale.yaml": (
            "deep_cross_attention",
            16,
            "noise_aware_multiscale",
        ),
        "v065_deep4_balanced_tree.yaml": ("deep_cross_attention", 4, "balanced_tree"),
    }

    for name, (kind, depth, sampler) in expected.items():
        raw = _load_yaml(name)
        assert raw["model"]["interaction"]["kind"] == kind
        assert raw["model"]["interaction"]["num_layers"] == depth
        assert raw["data"]["pair_sample_strategy"] == sampler
        assert raw["model"]["antibody_encoder"]["mode"] == "frozen_cached"
        assert raw["model"]["antigen_encoder"]["mode"] == "frozen_cached"
        assert raw["train"]["device"] == "cuda"
        assert raw["train"]["num_workers"] >= 1
        assert raw["train"]["pin_memory"] is True


def test_group_holdout_batch_sizes_are_fixed_by_model_depth():
    expected_batch_by_depth = {
        "concat": 512,
        "deep4": 512,
        "deep8": 128,
        "deep16": 128,
    }
    depth_by_name = {
        "concat": [
            "v065_concat_ranknet.yaml",
            "v065_concat_randomized_bst_no_redundancy.yaml",
            "v065_concat_noise_aware_multiscale.yaml",
        ],
        "deep4": [
            "v065_deep4_ranknet.yaml",
            "v065_deep4_randomized_bst_no_redundancy.yaml",
            "v065_deep4_noise_aware_multiscale.yaml",
            "v065_deep4_balanced_tree.yaml",
        ],
        "deep8": [
            "v065_deep8_ranknet.yaml",
            "v065_deep8_randomized_bst_no_redundancy.yaml",
            "v065_deep8_noise_aware_multiscale.yaml",
        ],
        "deep16": [
            "v065_deep16_ranknet.yaml",
            "v065_deep16_randomized_bst_no_redundancy.yaml",
            "v065_deep16_noise_aware_multiscale.yaml",
        ],
    }

    for depth_name, names in depth_by_name.items():
        for name in names:
            raw = _load_yaml(name)
            assert raw["train"]["batch_size"] == expected_batch_by_depth[depth_name]


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


def test_group_holdout_all_controls_submits_all_families_from_one_cache():
    script = (
        ROOT / "scripts/slurm/submit_group_holdout_all_controls.sh"
    ).read_text(encoding="utf-8")

    assert "affinity_group_holdout_submit_prerequisites" in script
    assert "ranknet_configs" in script
    assert "randomized_bst_configs" in script
    assert "noise_aware_configs" in script
    assert "balanced_tree_config" in script
    assert "GROUP_HOLDOUT_CACHE_JOB" in script


def test_deprecated_v065_scripts_forward_to_group_holdout_scripts():
    expected = {
        "submit_v065_training_chain.sh": "submit_group_holdout_ranknet.sh",
        "submit_randomized_bst_no_redundancy.sh": (
            "submit_group_holdout_randomized_bst.sh"
        ),
        "submit_noise_aware_multiscale.sh": (
            "submit_group_holdout_noise_aware_multiscale.sh"
        ),
    }
    for old_name, new_name in expected.items():
        script = (ROOT / "scripts/slurm" / old_name).read_text(encoding="utf-8")
        assert "deprecated" in script
        assert new_name in script


def test_group_holdout_slurm_shell_scripts_pass_bash_syntax_check():
    paths = [
        ROOT / "scripts/slurm/group_holdout_submit_common.sh",
        ROOT / "scripts/slurm/submit_group_holdout_ranknet.sh",
        ROOT / "scripts/slurm/submit_group_holdout_randomized_bst.sh",
        ROOT / "scripts/slurm/submit_group_holdout_noise_aware_multiscale.sh",
        ROOT / "scripts/slurm/submit_group_holdout_balanced_tree.sh",
        ROOT / "scripts/slurm/submit_group_holdout_all_controls.sh",
        ROOT / "scripts/slurm/submit_v065_training_chain.sh",
        ROOT / "scripts/slurm/submit_randomized_bst_no_redundancy.sh",
        ROOT / "scripts/slurm/submit_noise_aware_multiscale.sh",
        ROOT / "scripts/slurm/download_v065_models_login.sh",
        ROOT / "scripts/slurm/check_v065_models.sbatch",
        ROOT / "scripts/slurm/build_v065_embedding_cache.sbatch",
    ]

    for path in paths:
        completed = subprocess.run(
            ["bash", "-n", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
