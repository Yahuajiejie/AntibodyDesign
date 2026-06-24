"""Standard table schema and sampler defaults."""

from __future__ import annotations


REQUIRED_COLUMNS: tuple[str, ...] = (
    "record_id", "dataset_id", "study_id", "table_id",
    "source_file", "source_row",
    "antibody_id", "antibody_type",
    "heavy_chain", "light_chain", "single_chain_sequence",
    "antigen_key", "antigen_name", "antigen_sequence", "antigen_source",
    "assay_name", "assay_type",
    "metric_name", "metric_value_raw", "metric_value_numeric",
    "metric_unit", "metric_direction", "transform_rule",
    "rank_label", "label_kind",
    "group_id", "keep_for_training", "drop_reason",
)

_EXAMPLE_COLUMNS: tuple[str, ...] = (
    "record_id", "dataset_id",
    "heavy_chain", "light_chain", "single_chain_sequence", "antibody_type",
    "antigen_sequence", "antigen_key",
    "rank_label", "label_kind", "group_id",
)

PAIR_COLUMNS: tuple[str, ...] = (
    "pair_id", "group_id", "record_id_i", "record_id_j",
    "label_i", "label_j", "y_ij",
)

GROUP_COLUMNS: tuple[str, ...] = (
    "group_id", "record_id", "rank_label", "label_kind",
)

_BINARY_LABEL_KIND = "binary"
_DEFAULT_LARGE_GROUP_THRESHOLD = 10_000
_DEFAULT_PAIR_ENUMERATION_LIMIT = 100_000
_DEFAULT_LABEL_BLOCK_COUNT = 5
_DEFAULT_INTRA_BLOCK_PAIRS_PER_LARGE_GROUP = 50
_DEFAULT_DISCRETE_LABEL_UNIQUE_THRESHOLD = 32
_DEFAULT_DISCRETE_LABEL_RATIO_THRESHOLD = 0.05
