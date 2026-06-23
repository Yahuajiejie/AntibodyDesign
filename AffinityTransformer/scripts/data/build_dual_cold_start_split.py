#!/usr/bin/env python3
"""Build an offline dual cold-start split (unseen antigen AND unseen antibody).

Implements entity_cold_start_protocols.md section 5.5 as a standalone step.
Records and the entity annotation table are loaded separately; an optional
representation annotation table enables effective-input collision auditing.
Outputs keep the base records schema unless ``--debug-keep-entity-columns``.

Example:
    python scripts/data/build_dual_cold_start_split.py \\
        --records processed/binding/all_records.parquet \\
        --entity-annotations processed/binding/annotations/entity_annotations.parquet \\
        --output-dir processed/binding/splits/dual_cold_start \\
        --valid-fraction 0.1 --test-fraction 0.1 --seed 0 --min-eval-records 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from affinity_transformer.annotations import (
    load_entity_annotations,
    load_representation_annotations,
)
from affinity_transformer.dataset import load_records
from affinity_transformer.splits import (
    build_dual_cold_start_manifest,
    build_dual_cold_start_split,
    frame_hash,
    write_dual_cold_start_split,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--entity-annotations", required=True, type=Path)
    parser.add_argument(
        "--representation-annotations",
        type=Path,
        default=None,
        help="Optional representation table (enables effective-input audit)",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min-eval-records",
        type=int,
        default=5,
        help="Minimum protocol-eligible records required per evaluation group.",
    )
    parser.add_argument(
        "--debug-keep-entity-columns",
        action="store_true",
        help="Persist entity identity columns in train/valid/test parquet.",
    )
    args = parser.parse_args()

    records = load_records(args.records)
    annotations = load_entity_annotations(args.entity_annotations)

    representation = None
    representation_hash = None
    if args.representation_annotations is not None:
        representation = load_representation_annotations(args.representation_annotations)
        representation_hash = frame_hash(representation)

    result = build_dual_cold_start_split(
        records,
        annotations,
        valid_fraction=args.valid_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        min_eval_records=args.min_eval_records,
        representation_annotations=representation,
    )
    manifest = build_dual_cold_start_manifest(
        seed=args.seed,
        valid_fraction=args.valid_fraction,
        test_fraction=args.test_fraction,
        min_eval_records=args.min_eval_records,
        input_records_hash=frame_hash(records),
        entity_annotations_hash=frame_hash(annotations),
        representation_annotations_hash=representation_hash,
        effective_input_audited=representation is not None,
        component_summary=result.component_summary,
    )
    write_dual_cold_start_split(
        result,
        args.output_dir,
        manifest=manifest,
        debug=args.debug_keep_entity_columns,
    )

    print("protocol=dual_cold_start")
    print(f"n_components={len(result.component_summary)}")
    print(f"train rows={len(result.train)}")
    print(f"valid rows={len(result.valid)}")
    print(f"test rows={len(result.test)}")
    print(f"protocol-excluded rows={len(result.excluded_records)}")
    print(f"effective_input_audited={representation is not None}")
    print(f"split outputs -> {args.output_dir}")


if __name__ == "__main__":
    main()
