"""Load, validate, and join the entity annotation table.

The entity annotation is a narrow table keyed by ``record_id``. It carries the
cold-start identity fields that must NOT be added to the base records schema
(``dataset/schema.py::REQUIRED_COLUMNS`` is intentionally left untouched).

Public interface:

    load_entity_annotations(path) -> pd.DataFrame
    validate_entity_annotations(records, annotations) -> None
    join_entity_annotations(records, annotations) -> pd.DataFrame
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Columns the antibody cold-start protocol relies on. ``antigen_cluster_id`` is
# accepted when present but is NOT required by the antibody cold-start logic
# (see entity_cold_start_protocols.md section 5.3).
REQUIRED_ENTITY_COLUMNS = (
    "record_id",
    "measurement_family_id",
    "antibody_sequence_key",
    "antibody_cluster_id",
    "antigen_sequence_key",
    "interaction_key",
)
OPTIONAL_ENTITY_COLUMNS = ("antigen_cluster_id",)
ENTITY_ANNOTATION_COLUMNS = REQUIRED_ENTITY_COLUMNS + OPTIONAL_ENTITY_COLUMNS

_IDENTITY_VALUE_COLUMNS = tuple(
    column for column in REQUIRED_ENTITY_COLUMNS if column != "record_id"
)


def load_entity_annotations(path: Path) -> pd.DataFrame:
    """Read an entity annotation table (parquet/csv/tsv) keyed by record_id.

    Args:
        path: Path to ``.parquet``/``.pq``/``.csv``/``.tsv`` annotation file.

    Returns:
        The annotation DataFrame with the index reset.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the format is unsupported or a required column is absent.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"entity annotations not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        annotations = pd.read_parquet(path)
    elif suffix in {".csv", ".tsv"}:
        annotations = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
    else:
        raise ValueError(
            f"unsupported entity annotation format: {path.suffix!r} "
            "(expected .parquet, .csv, or .tsv)"
        )
    missing = [c for c in REQUIRED_ENTITY_COLUMNS if c not in annotations.columns]
    if missing:
        raise ValueError(
            f"entity annotations missing required column(s): {missing}"
        )
    return annotations.reset_index(drop=True)


def validate_entity_annotations(
    records: pd.DataFrame,
    annotations: pd.DataFrame,
) -> None:
    """Validate that annotations cover the records and are internally consistent.

    Checks:
        * required entity columns are present;
        * ``record_id`` is non-null, non-empty and unique in the annotation;
        * every base record has a matching annotation row (no missing rows);
        * required identity values are non-null and non-empty;
        * each ``antibody_sequence_key`` maps to exactly one
          ``antibody_cluster_id`` (many-to-one: one cluster may hold many keys).

    Raises:
        ValueError: If any check fails.
    """
    if "record_id" not in records.columns:
        raise ValueError("records must contain a 'record_id' column")
    missing_columns = [c for c in REQUIRED_ENTITY_COLUMNS if c not in annotations.columns]
    if missing_columns:
        raise ValueError(
            f"entity annotations missing required column(s): {missing_columns}"
        )

    annotation_ids = annotations["record_id"].astype(str)
    if annotations["record_id"].isna().any() or annotation_ids.str.strip().eq("").any():
        raise ValueError("entity annotations require non-null, non-empty record_id")
    duplicates = annotation_ids[annotation_ids.duplicated()].unique().tolist()
    if duplicates:
        raise ValueError(
            f"entity annotations contain duplicate record_id values: {duplicates[:10]}"
        )

    record_ids = set(records["record_id"].astype(str))
    annotated_ids = set(annotation_ids)
    missing_rows = sorted(record_ids - annotated_ids)
    if missing_rows:
        raise ValueError(
            f"entity annotations missing rows for {len(missing_rows)} record(s); "
            f"first={missing_rows[:10]}"
        )

    relevant = annotations.loc[annotation_ids.isin(record_ids)]
    for column in _IDENTITY_VALUE_COLUMNS:
        values = relevant[column].astype(str)
        invalid = relevant[column].isna() | values.str.strip().eq("")
        if invalid.any():
            bad = relevant.loc[invalid, "record_id"].astype(str).tolist()
            raise ValueError(
                f"entity annotation column {column!r} must be non-null and "
                f"non-empty; record_id={bad[:10]}"
            )

    # Many-to-one only: each antibody_sequence_key -> exactly one
    # antibody_cluster_id. One antibody_cluster_id MAY contain multiple
    # antibody_sequence_key values, which is allowed.
    cluster_counts = (
        relevant.groupby("antibody_sequence_key", sort=False)["antibody_cluster_id"]
        .nunique()
    )
    conflicts = cluster_counts[cluster_counts != 1]
    if not conflicts.empty:
        raise ValueError(
            "antibody_sequence_key must map to exactly one antibody_cluster_id; "
            f"conflicts={conflicts.index.astype(str).tolist()[:10]}"
        )


def join_entity_annotations(
    records: pd.DataFrame,
    annotations: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join entity columns onto records for transient split construction.

    The returned frame preserves the row order and ``record_id`` dtype of
    ``records`` and never persists into base split files (the split writer
    strips entity columns by default).

    Raises:
        ValueError: If ``validate_entity_annotations`` fails.
    """
    validate_entity_annotations(records, annotations)
    entity_columns = [
        column
        for column in ENTITY_ANNOTATION_COLUMNS
        if column != "record_id" and column in annotations.columns
    ]
    narrow = (
        annotations[["record_id", *entity_columns]]
        .drop_duplicates(subset="record_id")
        .copy()
    )
    narrow["_join_key"] = narrow["record_id"].astype(str)
    narrow = narrow.drop(columns=["record_id"])

    base = records.copy()
    overlap = [column for column in entity_columns if column in base.columns]
    if overlap:
        base = base.drop(columns=overlap)
    base["_join_key"] = base["record_id"].astype(str)

    joined = base.merge(narrow, on="_join_key", how="left", validate="many_to_one")
    return joined.drop(columns=["_join_key"])


# ---------------------------------------------------------------------------
# Representation annotations (optional; used for effective-input audits only).
#
# Keyed by (sequence_type, sequence_key); supplies the per-sequence effective
# model-input hash. This table is OPTIONAL: ``effective_input_hash`` must never
# become an unconditional field in base records or the entity annotation.
# ---------------------------------------------------------------------------
REQUIRED_REPRESENTATION_COLUMNS = (
    "sequence_type",
    "sequence_key",
    "effective_input_hash",
)


def load_representation_annotations(path: Path) -> pd.DataFrame:
    """Read a representation annotation table (parquet/csv/tsv).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the format is unsupported or a required column is absent.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"representation annotations not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        annotations = pd.read_parquet(path)
    elif suffix in {".csv", ".tsv"}:
        annotations = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
    else:
        raise ValueError(
            f"unsupported representation annotation format: {path.suffix!r} "
            "(expected .parquet, .csv, or .tsv)"
        )
    missing = [c for c in REQUIRED_REPRESENTATION_COLUMNS if c not in annotations.columns]
    if missing:
        raise ValueError(
            f"representation annotations missing required column(s): {missing}"
        )
    return annotations.reset_index(drop=True)


def validate_representation_annotations(
    records: pd.DataFrame,
    representation_annotations: pd.DataFrame,
    *,
    sequence_type: str,
    sequence_key_column: str,
) -> None:
    """Validate representation rows for one ``sequence_type`` against records.

    Checks:
        * required representation columns present;
        * records carry ``sequence_key_column``;
        * each ``sequence_key`` maps to exactly one ``effective_input_hash``;
        * every record's sequence key has a representation row.

    Raises:
        ValueError: If any check fails.
    """
    missing = [
        c for c in REQUIRED_REPRESENTATION_COLUMNS
        if c not in representation_annotations.columns
    ]
    if missing:
        raise ValueError(
            f"representation annotations missing required column(s): {missing}"
        )
    if sequence_key_column not in records.columns:
        raise ValueError(
            f"records must contain {sequence_key_column!r} to join representation "
            f"annotations for sequence_type={sequence_type!r}"
        )

    rows = representation_annotations.loc[
        representation_annotations["sequence_type"].astype(str) == sequence_type
    ]
    keys = rows["sequence_key"].astype(str)
    if rows["sequence_key"].isna().any() or keys.str.strip().eq("").any():
        raise ValueError(
            f"representation sequence_key must be non-null and non-empty for "
            f"sequence_type={sequence_type!r}"
        )
    hash_counts = rows.groupby("sequence_key", sort=False)["effective_input_hash"].nunique()
    conflicts = hash_counts[hash_counts != 1]
    if not conflicts.empty:
        raise ValueError(
            "sequence_key must map to exactly one effective_input_hash; "
            f"conflicts={conflicts.index.astype(str).tolist()[:10]}"
        )

    needed = set(records[sequence_key_column].astype(str))
    available = set(keys)
    missing_keys = sorted(needed - available)
    if missing_keys:
        raise ValueError(
            f"representation annotations missing {sequence_type} rows for "
            f"{len(missing_keys)} sequence_key(s); first={missing_keys[:10]}"
        )


def join_representation_annotations(
    records: pd.DataFrame,
    representation_annotations: pd.DataFrame,
    *,
    sequence_type: str,
    sequence_key_column: str,
    out_column: str,
) -> pd.DataFrame:
    """Materialize ``out_column`` (effective input hash) onto records.

    Uses only rows of the given ``sequence_type``; row order and dtypes of
    ``records`` are preserved. Validates before joining.
    """
    validate_representation_annotations(
        records,
        representation_annotations,
        sequence_type=sequence_type,
        sequence_key_column=sequence_key_column,
    )
    rows = representation_annotations.loc[
        representation_annotations["sequence_type"].astype(str) == sequence_type
    ]
    mapping = dict(zip(
        rows["sequence_key"].astype(str),
        rows["effective_input_hash"].astype(str),
    ))
    out = records.copy()
    out[out_column] = out[sequence_key_column].astype(str).map(mapping)
    return out
