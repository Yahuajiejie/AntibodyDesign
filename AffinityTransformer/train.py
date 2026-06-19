#!/usr/bin/env python3
"""Training entry point for AffinityTransformer."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from affinity_transformer.config import load_config
from affinity_transformer.training import (
    resolve_data_paths,
    run_cached_ranknet,
    run_group_kfold_cross_validation,
    run_online_training,
)
from affinity_transformer.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or _default_output_dir(args.config)
    run_training(args.config, output_dir)
    print(f"wrote run outputs -> {output_dir}")


def run_training(config_path: Path, output_dir: Path) -> dict[str, float]:
    """Load one experiment and dispatch to its explicit execution mode."""
    config = load_config(config_path)
    output_dir = ensure_dir(output_dir)

    runner = (
        run_cached_ranknet
        if config.model.antibody_encoder.mode == "frozen_cached"
        else run_online_training
    )
    if config.cross_validation.enabled:
        return run_group_kfold_cross_validation(
            config_path,
            config,
            output_dir,
            runner,
        )

    train_path, valid_path, test_path = resolve_data_paths(config)
    return runner(
        config_path,
        config,
        output_dir,
        train_path,
        valid_path,
        test_path,
    )


def _default_output_dir(config_path: Path) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return Path("outputs") / f"{config_path.stem}-{stamp}"


if __name__ == "__main__":
    main()
