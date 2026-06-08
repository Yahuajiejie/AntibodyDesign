"""
Data feature-building subpackage.
"""

from .context import (
    antigen_context_dim,
    antigen_type_one_hot,
    build_antibody_feature_matrix,
    build_antigen_context_feature_matrix,
    build_antigen_context_matrix,
    build_antigen_flags,
    has_official_antigen_sequence,
    sequence_source_prefix,
)

__all__ = [
    "antigen_context_dim",
    "antigen_type_one_hot",
    "build_antibody_feature_matrix",
    "build_antigen_context_feature_matrix",
    "build_antigen_context_matrix",
    "build_antigen_flags",
    "has_official_antigen_sequence",
    "sequence_source_prefix",
]
