"""
Sequence encoder subpackage.

This package wraps external protein/antibody language models.
"""

from .sequence import (
    HuggingFaceSequenceEncoder,
    cache_paths,
    format_paired_antibody_sequence,
    format_single_sequence,
    get_or_compute_sequence_embeddings,
    load_cached_sequence_embedding,
    paired_sequence_hash,
    save_sequence_embedding,
    sequence_hash,
)

__all__ = [
    "HuggingFaceSequenceEncoder",
    "cache_paths",
    "format_paired_antibody_sequence",
    "format_single_sequence",
    "get_or_compute_sequence_embeddings",
    "load_cached_sequence_embedding",
    "paired_sequence_hash",
    "save_sequence_embedding",
    "sequence_hash",
]
