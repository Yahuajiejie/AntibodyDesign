"""Reusable filters for selecting processed training records.

This module operates only on the standard processed table. It does not read
raw CSVs, does not transform labels, and does not build train/valid/test
splits. Its job is to select a reproducible subset before splitting or
training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .utils import hash_text


@dataclass(frozen=True)
class AntigenAntibodyPair:
    """One exact antigen-antibody selector.

    The pair matches records whose `antigen_key` and `antibody_id` both match.
    If a dataset does not populate `antibody_id`, use `include_record_ids` or
    `include_antibody_sequence_hashes` instead.
    """

    antigen_key: str
    antibody_id: str


@dataclass(frozen=True)
class RecordFilterConfig:
    """Config for selecting a subset of standard processed records.

    Include filters are conjunctive across fields and disjunctive within one
    field. For example, `include_dataset_ids=["A", "B"]` and
    `include_antigen_keys=["X"]` keeps `(dataset_id in {A,B}) AND
    (antigen_key == X)`. Exclude filters are applied after include filters.
    """

    include_dataset_ids: tuple[str, ...] = ()
    exclude_dataset_ids: tuple[str, ...] = ()
    include_study_ids: tuple[str, ...] = ()
    exclude_study_ids: tuple[str, ...] = ()
    include_table_ids: tuple[str, ...] = ()
    exclude_table_ids: tuple[str, ...] = ()
    include_antigen_keys: tuple[str, ...] = ()
    exclude_antigen_keys: tuple[str, ...] = ()
    include_antibody_ids: tuple[str, ...] = ()
    exclude_antibody_ids: tuple[str, ...] = ()
    include_antibody_sequence_hashes: tuple[str, ...] = ()
    exclude_antibody_sequence_hashes: tuple[str, ...] = ()
    include_group_ids: tuple[str, ...] = ()
    exclude_group_ids: tuple[str, ...] = ()
    include_record_ids: tuple[str, ...] = ()
    exclude_record_ids: tuple[str, ...] = ()
    include_label_kinds: tuple[str, ...] = ()
    exclude_label_kinds: tuple[str, ...] = ()
    include_antibody_types: tuple[str, ...] = ()
    exclude_antibody_types: tuple[str, ...] = ()
    include_antigen_sources: tuple[str, ...] = ()
    exclude_antigen_sources: tuple[str, ...] = ()
    include_antigen_antibody_pairs: tuple[AntigenAntibodyPair, ...] = ()
    require_antigen_sequence: bool = False
    require_antibody_id: bool = False
    min_records_per_group: int | None = None
    min_trainable_records_per_group: int | None = None
    min_unique_labels_per_group: int | None = None

    def is_empty(self) -> bool:
        """Return True when this config does not remove any records."""
        return self == RecordFilterConfig()


_FIELD_FILTERS = (
    ("dataset_id", "include_dataset_ids", "exclude_dataset_ids"),
    ("study_id", "include_study_ids", "exclude_study_ids"),
    ("table_id", "include_table_ids", "exclude_table_ids"),
    ("antigen_key", "include_antigen_keys", "exclude_antigen_keys"),
    ("antibody_id", "include_antibody_ids", "exclude_antibody_ids"),
    ("group_id", "include_group_ids", "exclude_group_ids"),
    ("record_id", "include_record_ids", "exclude_record_ids"),
    ("label_kind", "include_label_kinds", "exclude_label_kinds"),
    ("antibody_type", "include_antibody_types", "exclude_antibody_types"),
    ("antigen_source", "include_antigen_sources", "exclude_antigen_sources"),
)

_LIST_FIELDS = {
    "include_dataset_ids",
    "exclude_dataset_ids",
    "include_study_ids",
    "exclude_study_ids",
    "include_table_ids",
    "exclude_table_ids",
    "include_antigen_keys",
    "exclude_antigen_keys",
    "include_antibody_ids",
    "exclude_antibody_ids",
    "include_antibody_sequence_hashes",
    "exclude_antibody_sequence_hashes",
    "include_group_ids",
    "exclude_group_ids",
    "include_record_ids",
    "exclude_record_ids",
    "include_label_kinds",
    "exclude_label_kinds",
    "include_antibody_types",
    "exclude_antibody_types",
    "include_antigen_sources",
    "exclude_antigen_sources",
}

_BOOL_FIELDS = {"require_antigen_sequence", "require_antibody_id"}
_INT_OR_NONE_FIELDS = {
    "min_records_per_group",
    "min_trainable_records_per_group",
    "min_unique_labels_per_group",
}
_PAIR_FIELD = "include_antigen_antibody_pairs"
_VALID_FIELDS = _LIST_FIELDS | _BOOL_FIELDS | _INT_OR_NONE_FIELDS | {_PAIR_FIELD}


def load_record_filter_config(path: Path) -> RecordFilterConfig:
    """Load a record filter config from YAML.

    The file may contain the filter directly, under `filter`, or under
    `data.filter` so the same YAML can be reused with `train.py`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Filter config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Filter config must be a YAML mapping: {path}")

    if isinstance(raw.get("data"), dict) and "filter" in raw["data"]:
        raw = raw["data"]["filter"]
    elif "filter" in raw:
        raw = raw["filter"]
    return build_record_filter_config(raw)


def build_record_filter_config(raw: Any) -> RecordFilterConfig:
    """Build `RecordFilterConfig` from a YAML-style mapping."""
    if raw is None:
        return RecordFilterConfig()
    if not isinstance(raw, dict):
        raise ValueError(f"record filter must be a mapping or null, got {raw!r}")

    unknown = sorted(set(raw) - _VALID_FIELDS)
    if unknown:
        raise ValueError(f"record filter contains unknown field(s): {unknown}")

    kwargs: dict[str, Any] = {}
    for field_name in _LIST_FIELDS:
        kwargs[field_name] = _tuple_of_str(raw.get(field_name), field_name)
    for field_name in _BOOL_FIELDS:
        kwargs[field_name] = bool(raw.get(field_name, False))
    for field_name in _INT_OR_NONE_FIELDS:
        kwargs[field_name] = _int_or_none(raw.get(field_name), field_name)
    kwargs[_PAIR_FIELD] = _pairs(raw.get(_PAIR_FIELD))
    return RecordFilterConfig(**kwargs)


def filter_records(records: pd.DataFrame, config: RecordFilterConfig) -> pd.DataFrame:
    """Apply a reproducible subset filter to a standard processed table."""
    _require_columns(records, ["record_id", "group_id"])
    filtered = records.copy()

    for column, include_field, exclude_field in _FIELD_FILTERS:
        include_values = getattr(config, include_field)
        exclude_values = getattr(config, exclude_field)
        if include_values or exclude_values:
            _require_columns(filtered, [column])
        if include_values:
            filtered = filtered.loc[_as_str(filtered[column]).isin(include_values)]
        if exclude_values:
            filtered = filtered.loc[~_as_str(filtered[column]).isin(exclude_values)]

    if config.include_antibody_sequence_hashes or config.exclude_antibody_sequence_hashes:
        hashes = antibody_sequence_hashes(filtered)
        if config.include_antibody_sequence_hashes:
            filtered = filtered.loc[hashes.isin(config.include_antibody_sequence_hashes)]
            hashes = hashes.loc[filtered.index]
        if config.exclude_antibody_sequence_hashes:
            filtered = filtered.loc[~hashes.isin(config.exclude_antibody_sequence_hashes)]

    if config.include_antigen_antibody_pairs:
        _require_columns(filtered, ["antigen_key", "antibody_id"])
        pair_mask = pd.Series(False, index=filtered.index)
        antigen = _as_str(filtered["antigen_key"])
        antibody = _as_str(filtered["antibody_id"])
        for pair in config.include_antigen_antibody_pairs:
            pair_mask |= antigen.eq(pair.antigen_key) & antibody.eq(pair.antibody_id)
        filtered = filtered.loc[pair_mask]

    if config.require_antigen_sequence:
        _require_columns(filtered, ["antigen_sequence"])
        filtered = filtered.loc[_present(filtered["antigen_sequence"])]

    if config.require_antibody_id:
        _require_columns(filtered, ["antibody_id"])
        filtered = filtered.loc[_present(filtered["antibody_id"])]

    filtered = _apply_group_thresholds(filtered, config)
    return filtered.sort_values(["dataset_id", "record_id"]).reset_index(drop=True)


def antibody_sequence_hashes(records: pd.DataFrame) -> pd.Series:
    """Return stable hashes for heavy/light/single-chain antibody identity."""
    _require_columns(records, ["heavy_chain", "light_chain", "single_chain_sequence"])
    values = (
        records["heavy_chain"].map(_nullable_text)
        + "|"
        + records["light_chain"].map(_nullable_text)
        + "|"
        + records["single_chain_sequence"].map(_nullable_text)
    )
    return values.map(hash_text)


def build_filter_summary(
    before: pd.DataFrame,
    after: pd.DataFrame,
    config: RecordFilterConfig,
) -> pd.DataFrame:
    """Build a small audit table for a filtering operation."""
    rows = [
        _summary_row("input", before),
        _summary_row("output", after),
    ]
    rows.append({
        "stage": "removed",
        "n_records": len(before) - len(after),
        "n_trainable_records": _n_trainable(before) - _n_trainable(after),
        "n_groups": _nunique(before, "group_id") - _nunique(after, "group_id"),
        "n_dataset_ids": _nunique(before, "dataset_id") - _nunique(after, "dataset_id"),
        "label_kind_counts": "",
        "antigen_source_counts": "",
        "filter_active": not config.is_empty(),
    })
    return pd.DataFrame(rows)


def write_filter_outputs(
    before: pd.DataFrame,
    after: pd.DataFrame,
    config: RecordFilterConfig,
    output_path: Path,
    summary_path: Path,
) -> None:
    """Write a filtered records file and its audit summary."""
    output_path = Path(output_path)
    summary_path = Path(summary_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        after.to_parquet(output_path, index=False)
    elif output_path.suffix == ".csv":
        after.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Filtered output must be .parquet or .csv, got {output_path}")
    build_filter_summary(before, after, config).to_csv(summary_path, index=False)


def _apply_group_thresholds(records: pd.DataFrame, config: RecordFilterConfig) -> pd.DataFrame:
    filtered = records
    if config.min_records_per_group is not None:
        filtered = _keep_groups(filtered, _group_sizes(filtered), config.min_records_per_group)
    if config.min_trainable_records_per_group is not None:
        filtered = _keep_groups(
            filtered,
            _trainable_records(filtered).groupby("group_id").size(),
            config.min_trainable_records_per_group,
        )
    if config.min_unique_labels_per_group is not None:
        labels = _trainable_records(filtered).copy()
        labels["rank_label_numeric"] = pd.to_numeric(labels["rank_label"], errors="coerce")
        counts = labels.groupby("group_id")["rank_label_numeric"].nunique()
        filtered = _keep_groups(filtered, counts, config.min_unique_labels_per_group)
    return filtered


def _keep_groups(records: pd.DataFrame, counts: pd.Series, minimum: int) -> pd.DataFrame:
    if minimum < 1:
        raise ValueError(f"group minimum must be >= 1, got {minimum}")
    keep_groups = set(counts[counts >= minimum].index.astype(str))
    return records.loc[_as_str(records["group_id"]).isin(keep_groups)]


def _group_sizes(records: pd.DataFrame) -> pd.Series:
    return records.groupby("group_id").size()


def _trainable_records(records: pd.DataFrame) -> pd.DataFrame:
    _require_columns(records, ["keep_for_training", "rank_label"])
    keep = records["keep_for_training"].map(_parse_bool)
    labels = pd.to_numeric(records["rank_label"], errors="coerce")
    finite = labels.notna() & labels.map(lambda value: value not in (float("inf"), float("-inf")))
    return records.loc[keep & finite]


def _summary_row(stage: str, records: pd.DataFrame) -> dict[str, Any]:
    return {
        "stage": stage,
        "n_records": len(records),
        "n_trainable_records": _n_trainable(records),
        "n_groups": _nunique(records, "group_id"),
        "n_dataset_ids": _nunique(records, "dataset_id"),
        "label_kind_counts": _value_counts_json(records, "label_kind"),
        "antigen_source_counts": _value_counts_json(records, "antigen_source"),
        "filter_active": "",
    }


def _n_trainable(records: pd.DataFrame) -> int:
    if not {"keep_for_training", "rank_label"} <= set(records.columns):
        return 0
    return len(_trainable_records(records))


def _nunique(records: pd.DataFrame, column: str) -> int:
    return int(records[column].nunique()) if column in records else 0


def _value_counts_json(records: pd.DataFrame, column: str) -> str:
    if column not in records:
        return "{}"
    counts = records[column].fillna("<NA>").astype(str).value_counts().sort_index().to_dict()
    return json.dumps(counts, sort_keys=True)


def _tuple_of_str(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"record filter field {field_name!r} must be a string/list/null")
    return tuple(str(item) for item in value)


def _int_or_none(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    result = int(value)
    if result < 1:
        raise ValueError(f"record filter field {field_name!r} must be >= 1")
    return result


def _pairs(value: Any) -> tuple[AntigenAntibodyPair, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{_PAIR_FIELD} must be a list of mappings")
    pairs: list[AntigenAntibodyPair] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"{_PAIR_FIELD} entries must be mappings")
        missing = [key for key in ("antigen_key", "antibody_id") if key not in item]
        if missing:
            raise ValueError(f"{_PAIR_FIELD} entry is missing field(s): {missing}")
        pairs.append(
            AntigenAntibodyPair(
                antigen_key=str(item["antigen_key"]),
                antibody_id=str(item["antibody_id"]),
            )
        )
    return tuple(pairs)


def _present(values: pd.Series) -> pd.Series:
    normalized = values.map(_nullable_text)
    return ~normalized.isin({"", "nan", "none", "null"})


def _nullable_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _as_str(values: pd.Series) -> pd.Series:
    return values.map(_nullable_text)


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, float) and value in (0.0, 1.0):
        return bool(int(value))
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "t", "1", "yes", "y"}:
            return True
        if normalized in {"false", "f", "0", "no", "n"}:
            return False
    raise ValueError(f"Invalid boolean value for keep_for_training: {value!r}")


def _require_columns(records: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in records.columns]
    if missing:
        raise ValueError(f"records is missing required column(s): {missing}")
