"""Separate entity-annotation tables for cold-start splitting.

Entity identity fields (sequence keys, cluster ids, measurement families,
interaction keys) live OUTSIDE the base records schema -- see
``docs/future/entity_cold_start_protocols.md`` sections 2-4. The base records
table keeps its standard columns; this package supplies the entity annotation
as an independent narrow table keyed by ``record_id`` and joins it only
transiently for split construction.
"""

from __future__ import annotations

from .io import (
    ENTITY_ANNOTATION_COLUMNS,
    OPTIONAL_ENTITY_COLUMNS,
    REQUIRED_ENTITY_COLUMNS,
    REQUIRED_REPRESENTATION_COLUMNS,
    join_entity_annotations,
    join_representation_annotations,
    load_entity_annotations,
    load_representation_annotations,
    validate_entity_annotations,
    validate_representation_annotations,
)

__all__ = [
    "ENTITY_ANNOTATION_COLUMNS",
    "OPTIONAL_ENTITY_COLUMNS",
    "REQUIRED_ENTITY_COLUMNS",
    "REQUIRED_REPRESENTATION_COLUMNS",
    "join_entity_annotations",
    "join_representation_annotations",
    "load_entity_annotations",
    "load_representation_annotations",
    "validate_entity_annotations",
    "validate_representation_annotations",
]
