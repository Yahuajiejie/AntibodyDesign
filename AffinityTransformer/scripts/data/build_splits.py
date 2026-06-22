#!/usr/bin/env python3
"""Build fixed train/valid/test splits from a processed records table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from affinity_transformer.antigen_clustering import compute_antigen_clusters
from affinity_transformer.dataset import load_records
from affinity_transformer.record_filter import (
    filter_records,
    load_record_filter_config,
    write_filter_outputs,
)
from affinity_transformer.splits import (
    AUXILIARY_STRATEGIES,
    ENTITY_COLD_START_STRATEGIES,
    VALID_STRATEGIES,
    build_antibody_cold_start_split,
    build_antigen_cold_start_split,
    build_splits,
    build_within_antigen_split,
    write_entity_cold_start_split,
    write_splits,
    write_within_antigen_split,
)

_ALL_STRATEGIES = sorted(
    VALID_STRATEGIES | AUXILIARY_STRATEGIES | ENTITY_COLD_START_STRATEGIES
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input records.parquet/csv")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for split files")
    parser.add_argument("--strategy", required=True, choices=_ALL_STRATEGIES)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min-eval-records",
        type=int,
        default=5,
        help=(
            "within_antigen_split and strict entity cold-start protocols: "
            "minimum protocol-eligible records required per evaluation group."
        ),
    )
    parser.add_argument(
        "--allow-new-experimental-group",
        action="store_true",
        help=(
            "antibody_cold_start_split only: require a known exact antigen "
            "but allow a holdout group_id absent from train. By default both "
            "the exact antigen and group must occur in train."
        ),
    )
    parser.add_argument(
        "--antigen-similarity-threshold",
        type=float,
        default=0.9,
        help=(
            "antigen_cluster_holdout_split only: minimum fraction of "
            "identical residues (same-length sequences only) for two "
            "antigens to be merged into one cluster. There is no universal "
            "default -- AbRank's published benchmark uses 0.75 for a "
            "highly diverse viral panel (HIV Env); inspect "
            "antigen_clusters.csv's cluster sizes at a few thresholds "
            "before committing to one. See antigen_clustering.py."
        ),
    )
    parser.add_argument(
        "--antigen-cluster-linkage",
        choices=["average", "complete"],
        default="average",
        help=(
            "antigen_cluster_holdout_split only: hierarchical-clustering "
            "linkage method. Deliberately excludes 'single' -- see "
            "antigen_clustering.py's module docstring for why."
        ),
    )
    parser.add_argument(
        "--antigen-clusters-cache",
        type=Path,
        default=None,
        help=(
            "antigen_cluster_holdout_split only: path to a previously "
            "written antigen_clusters.csv to reuse instead of recomputing "
            "clusters (useful for trying several valid/test fractions or "
            "seeds against the same clustering)."
        ),
    )
    parser.add_argument(
        "--filter-config",
        type=Path,
        default=None,
        help="Optional YAML filter applied before splitting.",
    )
    args = parser.parse_args()

    records = load_records(args.input)
    if args.filter_config is not None:
        config = load_record_filter_config(args.filter_config)
        filtered = filter_records(records, config)
        if filtered.empty:
            raise ValueError(f"filter produced no rows: {args.filter_config}")
        write_filter_outputs(
            records,
            filtered,
            config,
            args.output_dir / "filtered_records.parquet",
            args.output_dir / "filter_summary.csv",
        )
        records = filtered

    if args.strategy == "antigen_cluster_holdout_split":
        if args.antigen_clusters_cache is not None:
            antigen_clusters = pd.read_csv(args.antigen_clusters_cache)
        else:
            antigen_clusters = compute_antigen_clusters(
                records,
                similarity_threshold=args.antigen_similarity_threshold,
                linkage_method=args.antigen_cluster_linkage,
            )
            antigen_clusters.to_csv(args.output_dir / "antigen_clusters.csv", index=False)

        result = build_splits(
            records,
            strategy=args.strategy,
            valid_fraction=args.valid_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
            antigen_clusters=antigen_clusters,
        )
        write_splits(result, args.output_dir)

        print(f"strategy={args.strategy}")
        print(f"antigen clusters: {antigen_clusters['antigen_cluster_id'].nunique()} "
              f"from {len(antigen_clusters)} antigen_key values")
        print(f"train rows={len(result.train)}")
        print(f"valid rows={len(result.valid)}")
        print(f"test rows={len(result.test)}")
        print(f"split outputs -> {args.output_dir}")
        return

    if args.strategy in AUXILIARY_STRATEGIES:
        # within_antigen_split: auxiliary "known-antigen, new-antibody"
        # protocol (programming_spec_v1.0.md 3.2). group_id is allowed to
        # repeat across train/valid/test here -- that's by design, not a
        # leak. NOT a substitute for the main protocol; see
        # WithinAntigenSplitResult's docstring before reporting results
        # from this strategy as antigen-generalization evidence.
        result = build_within_antigen_split(
            records,
            valid_fraction=args.valid_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
            min_eval_records=args.min_eval_records,
        )
        write_within_antigen_split(result, args.output_dir)
        print(f"strategy={args.strategy} (auxiliary -- report as within-antigen generalization)")
        print(f"train rows={len(result.train)}")
        print(f"valid rows={len(result.valid)}")
        print(f"test rows={len(result.test)}")
        print(f"groups pinned to train (too small to split): {len(result.pinned_groups)}")
        print(f"split outputs -> {args.output_dir}")
        return

    if args.strategy in ENTITY_COLD_START_STRATEGIES:
        if args.strategy == "antibody_cold_start_split":
            result = build_antibody_cold_start_split(
                records,
                valid_fraction=args.valid_fraction,
                test_fraction=args.test_fraction,
                seed=args.seed,
                min_eval_records=args.min_eval_records,
                require_train_group=not args.allow_new_experimental_group,
            )
        else:
            if args.allow_new_experimental_group:
                raise ValueError(
                    "--allow-new-experimental-group only applies to "
                    "antibody_cold_start_split"
                )
            result = build_antigen_cold_start_split(
                records,
                valid_fraction=args.valid_fraction,
                test_fraction=args.test_fraction,
                seed=args.seed,
                min_eval_records=args.min_eval_records,
            )
        write_entity_cold_start_split(result, args.output_dir)
        print(f"strategy={args.strategy}")
        print(f"train rows={len(result.train)}")
        print(f"valid rows={len(result.valid)}")
        print(f"test rows={len(result.test)}")
        print(f"protocol-excluded rows={len(result.excluded_records)}")
        print(f"split outputs -> {args.output_dir}")
        return

    result = build_splits(
        records,
        strategy=args.strategy,
        valid_fraction=args.valid_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    write_splits(result, args.output_dir)

    print(f"strategy={args.strategy}")
    print(f"train rows={len(result.train)}")
    print(f"valid rows={len(result.valid)}")
    print(f"test rows={len(result.test)}")
    print(f"split outputs -> {args.output_dir}")


if __name__ == "__main__":
    main()
