"""
Embedding cache subpackage.

This package computes and reads antibody, antigen, and MSA-aware embeddings.
"""

from .antibody import embed_antibody_dataframe, build_antibody_feature_matrix
from .antigen import (
    AntigenEmbeddingManifest,
    embed_antigen_msa,
    embed_antigen_single,
    embed_ligand,
    embedding_cache_paths,
    has_cached_embedding,
    load_antigen_embedding_cache,
    read_embedding_manifest,
    save_embedding_with_manifest,
    zero_embedding,
)

__all__ = [
    "AntigenEmbeddingManifest",
    "build_antibody_feature_matrix",
    "embed_antibody_dataframe",
    "embed_antigen_msa",
    "embed_antigen_single",
    "embed_ligand",
    "embedding_cache_paths",
    "has_cached_embedding",
    "load_antigen_embedding_cache",
    "read_embedding_manifest",
    "save_embedding_with_manifest",
    "zero_embedding",
]
