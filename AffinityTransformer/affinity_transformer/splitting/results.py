"""Result dataclasses and split artifact writers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from ..utils import ensure_dir
from .common import COLD_START_IDENTITY_COLUMNS


@dataclass
class SplitResult:
    """DataFrames and QC reports produced by `build_splits`."""

    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    summary: pd.DataFrame
    leakage_report: pd.DataFrame


@dataclass
class WithinAntigenSplitResult:
    """Output of `build_within_antigen_split` (programming_spec_v1.0.md 3.2,
    "known-antigen, new-antibody").

    Unlike `SplitResult` from `group_holdout_split`, the same `group_id` (and
    the same antibody-sequence identity, as long as it's via a *different*
    group) may legitimately appear in more than one split here -- that is
    the point of this auxiliary protocol. What this guarantees, per group,
    is the only thing that actually matters for `dataset.pairs.build_pairs`
    (which only ever constructs pairs *within* one group_id): no
    `record_id` crosses a split, and within any single group, an antibody
    assigned to train never also shows up in that same group's valid/test
    rows. Whether that exact antibody sequence also appears in some other,
    unrelated group's training data is allowed and is not leakage -- the
    relationship actually being predicted (this antibody vs THIS antigen's
    other candidates) was never trained on regardless.

    `pinned_groups` lists every group that was too small to split reliably
    (fewer than 3 distinct antibody-sequence units, or splitting would leave
    fewer than `min_eval_records` records in valid or test) -- these are
    routed entirely to train rather than forced into an unstable 1-2-point
    split (spec section 3.4).

    Always report results from this split as "within-antigen
    generalization", never as evidence of generalization to unseen
    antigens.
    """

    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    summary: pd.DataFrame
    leakage_report: pd.DataFrame
    pinned_groups: pd.DataFrame


@dataclass(frozen=True)
class GroupFold:
    """One group-isolated cross-validation fold."""

    index: int
    train: pd.DataFrame
    valid: pd.DataFrame


@dataclass
class EntityColdStartSplitResult:
    """Fixed train/valid/test artifacts for one strict entity protocol."""

    protocol: str
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    summary: pd.DataFrame
    leakage_report: pd.DataFrame
    eligibility_report: pd.DataFrame
    excluded_records: pd.DataFrame
    unit_assignments: pd.DataFrame


@dataclass
class EntityColdStartFold:
    """One protocol-aware development fold; final test never rotates here."""

    protocol: str
    index: int
    train: pd.DataFrame
    valid: pd.DataFrame
    leakage_report: pd.DataFrame
    eligibility_report: pd.DataFrame
    excluded_records: pd.DataFrame
    unit_assignments: pd.DataFrame


@dataclass
class DualColdStartSplitResult:
    """Fixed train/valid/test artifacts for the Dual cold-start protocol.

    Same fields as an entity split plus ``component_summary`` -- the preflight
    per-component feasibility statistics (largest-component first).
    """

    protocol: str
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    summary: pd.DataFrame
    leakage_report: pd.DataFrame
    eligibility_report: pd.DataFrame
    excluded_records: pd.DataFrame
    unit_assignments: pd.DataFrame
    component_summary: pd.DataFrame


def write_splits(result: SplitResult, output_dir: Path) -> None:
    """Write split records and QC reports to `output_dir`."""
    output_dir = ensure_dir(Path(output_dir))
    result.train.to_parquet(output_dir / "train.parquet", index=False)
    result.valid.to_parquet(output_dir / "valid.parquet", index=False)
    result.test.to_parquet(output_dir / "test.parquet", index=False)
    result.summary.to_csv(output_dir / "split_summary.csv", index=False)
    result.leakage_report.to_csv(output_dir / "leakage_report.csv", index=False)


def write_within_antigen_split(result: WithinAntigenSplitResult, output_dir: Path) -> None:
    """Write within-antigen split records and QC reports to `output_dir`."""
    output_dir = ensure_dir(Path(output_dir))
    result.train.to_parquet(output_dir / "train.parquet", index=False)
    result.valid.to_parquet(output_dir / "valid.parquet", index=False)
    result.test.to_parquet(output_dir / "test.parquet", index=False)
    result.summary.to_csv(output_dir / "split_summary.csv", index=False)
    result.leakage_report.to_csv(output_dir / "leakage_report.csv", index=False)
    result.pinned_groups.to_csv(output_dir / "pinned_groups.csv", index=False)


def write_entity_cold_start_split(
    result: EntityColdStartSplitResult,
    output_dir: Path,
) -> None:
    """Write one strict entity-protocol split and all audit artifacts."""
    output_dir = ensure_dir(Path(output_dir))
    result.train.to_parquet(output_dir / "train.parquet", index=False)
    result.valid.to_parquet(output_dir / "valid.parquet", index=False)
    result.test.to_parquet(output_dir / "test.parquet", index=False)
    result.summary.to_csv(output_dir / "split_summary.csv", index=False)
    result.leakage_report.to_csv(output_dir / "leakage_report.csv", index=False)
    result.eligibility_report.to_csv(output_dir / "eligibility_report.csv", index=False)
    result.excluded_records.to_parquet(output_dir / "excluded_records.parquet", index=False)
    result.unit_assignments.to_parquet(output_dir / "unit_assignments.parquet", index=False)


def build_antibody_cold_start_manifest(
    *,
    seed: int,
    valid_fraction: float,
    test_fraction: float,
    min_eval_records: int,
    require_train_group: bool,
    input_records_hash: str,
    entity_annotations_hash: str,
) -> dict[str, object]:
    """Build the small ``split_manifest.yaml`` payload for one split."""
    return {
        "protocol": "antibody_cold_start",
        "seed": int(seed),
        "valid_fraction": float(valid_fraction),
        "test_fraction": float(test_fraction),
        "min_eval_records": int(min_eval_records),
        "require_train_group": bool(require_train_group),
        "input_records_hash": str(input_records_hash),
        "entity_annotations_hash": str(entity_annotations_hash),
    }


def _strip_entity_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop cold-start identity columns so base split files keep base schema."""
    drop = [column for column in COLD_START_IDENTITY_COLUMNS if column in frame.columns]
    return frame.drop(columns=drop) if drop else frame


def build_antigen_cold_start_manifest(
    *,
    seed: int,
    valid_fraction: float,
    test_fraction: float,
    min_eval_records: int,
    input_records_hash: str,
    entity_annotations_hash: str,
    representation_annotations_hash: str | None,
    effective_input_audited: bool,
) -> dict[str, object]:
    """Build the ``split_manifest.yaml`` payload for an antigen cold-start split.

    ``effective_input_audited`` records whether effective-input overlap was
    checked (representation annotations supplied); when False the manifest makes
    explicit that this audit was not performed.
    """
    return {
        "protocol": "antigen_cold_start",
        "seed": int(seed),
        "valid_fraction": float(valid_fraction),
        "test_fraction": float(test_fraction),
        "min_eval_records": int(min_eval_records),
        "input_records_hash": str(input_records_hash),
        "entity_annotations_hash": str(entity_annotations_hash),
        "representation_annotations_hash": (
            None if representation_annotations_hash is None
            else str(representation_annotations_hash)
        ),
        "effective_input_audited": bool(effective_input_audited),
    }


def _write_cold_start_artifacts(
    result: EntityColdStartSplitResult,
    output_dir: Path,
    *,
    manifest: dict[str, object],
    debug: bool = False,
) -> Path:
    """Shared writer for entity cold-start splits (antibody and antigen).

    train/valid/test parquet keep the base records columns by default; entity
    identity columns are dropped unless ``debug=True``. Component assignments,
    eligibility, excluded records, the leakage report, a summary and
    ``split_manifest.yaml`` are written alongside.
    """
    output_dir = ensure_dir(Path(output_dir))
    for name, frame in (
        ("train", result.train), ("valid", result.valid), ("test", result.test)
    ):
        persisted = frame if debug else _strip_entity_columns(frame)
        persisted.to_parquet(output_dir / f"{name}.parquet", index=False)
    result.unit_assignments.to_parquet(
        output_dir / "component_assignments.parquet", index=False
    )
    result.eligibility_report.to_csv(output_dir / "eligibility_report.csv", index=False)
    result.excluded_records.to_parquet(
        output_dir / "excluded_records.parquet", index=False
    )
    result.leakage_report.to_csv(output_dir / "leakage_report.csv", index=False)
    result.summary.to_csv(output_dir / "summary.csv", index=False)
    with open(output_dir / "split_manifest.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(manifest), handle, sort_keys=False)
    return output_dir


def write_antibody_cold_start_split(
    result: EntityColdStartSplitResult,
    output_dir: Path,
    *,
    manifest: dict[str, object],
    debug: bool = False,
) -> Path:
    """Write antibody cold-start split files, audits, and a small manifest."""
    if result.protocol != "antibody_cold_start":
        raise ValueError(
            f"write_antibody_cold_start_split expects an antibody_cold_start "
            f"result, got protocol={result.protocol!r}"
        )
    return _write_cold_start_artifacts(
        result, output_dir, manifest=manifest, debug=debug
    )


def write_antigen_cold_start_split(
    result: EntityColdStartSplitResult,
    output_dir: Path,
    *,
    manifest: dict[str, object],
    debug: bool = False,
) -> Path:
    """Write antigen cold-start split files, audits, and a small manifest."""
    if result.protocol != "antigen_cold_start":
        raise ValueError(
            f"write_antigen_cold_start_split expects an antigen_cold_start "
            f"result, got protocol={result.protocol!r}"
        )
    return _write_cold_start_artifacts(
        result, output_dir, manifest=manifest, debug=debug
    )


def build_dual_cold_start_manifest(
    *,
    seed: int,
    valid_fraction: float,
    test_fraction: float,
    min_eval_records: int,
    input_records_hash: str,
    entity_annotations_hash: str,
    representation_annotations_hash: str | None,
    effective_input_audited: bool,
    component_summary: pd.DataFrame,
) -> dict[str, object]:
    """Build the ``split_manifest.yaml`` payload for a dual cold-start split.

    ``component_summary`` (largest-component first) supplies ``n_components`` and
    the largest-component fractions reported in the manifest.
    """
    if component_summary.empty:
        largest = {
            "record_fraction": 0.0, "group_fraction": 0.0,
            "antibody_cluster_fraction": 0.0, "antigen_cluster_fraction": 0.0,
        }
    else:
        largest = component_summary.iloc[0]
    return {
        "protocol": "dual_cold_start",
        "seed": int(seed),
        "valid_fraction": float(valid_fraction),
        "test_fraction": float(test_fraction),
        "min_eval_records": int(min_eval_records),
        "input_records_hash": str(input_records_hash),
        "entity_annotations_hash": str(entity_annotations_hash),
        "representation_annotations_hash": (
            None if representation_annotations_hash is None
            else str(representation_annotations_hash)
        ),
        "effective_input_audited": bool(effective_input_audited),
        "n_components": int(len(component_summary)),
        "largest_component_record_fraction": float(largest["record_fraction"]),
        "largest_component_group_fraction": float(largest["group_fraction"]),
        "largest_component_antibody_cluster_fraction": float(
            largest["antibody_cluster_fraction"]
        ),
        "largest_component_antigen_cluster_fraction": float(
            largest["antigen_cluster_fraction"]
        ),
    }


def write_dual_cold_start_split(
    result: DualColdStartSplitResult,
    output_dir: Path,
    *,
    manifest: dict[str, object],
    debug: bool = False,
) -> Path:
    """Write dual cold-start split files, audits, manifest, and component summary.

    Reuses the shared cold-start artifact writer and adds ``component_summary.csv``
    (the preflight feasibility statistics).
    """
    if result.protocol != "dual_cold_start":
        raise ValueError(
            f"write_dual_cold_start_split expects a dual_cold_start result, "
            f"got protocol={result.protocol!r}"
        )
    output_dir = _write_cold_start_artifacts(
        result, output_dir, manifest=manifest, debug=debug
    )
    result.component_summary.to_csv(
        output_dir / "component_summary.csv", index=False
    )
    return output_dir
