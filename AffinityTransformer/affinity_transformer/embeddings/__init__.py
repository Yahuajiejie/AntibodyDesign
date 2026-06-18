"""Foundation-model embedding extraction and cache interfaces."""

from .collate import (
    EmbeddingBatch,
    PairEmbeddingBatch,
    collate_embedding_batch,
    collate_pair_embedding_batch,
)
from .extractors import (
    EmbeddingExtractor,
    build_embedding_extractor,
    register_embedding_extractor,
    registered_embedding_extractors,
)
from .schema import (
    AntibodySequenceInput,
    EmbeddingItem,
    EmbeddingRequest,
    antibody_embedding_request,
    antibody_sequence_hash,
    antigen_embedding_request,
    antigen_sequence_hash,
)
from .huggingface import Esm2EmbeddingExtractor, IgBertEmbeddingExtractor
from .pipeline import collect_embedding_requests, write_embedding_cache
from .store import (
    EmbeddingNotFoundError,
    EmbeddingStore,
    InMemoryEmbeddingStore,
    ShardedEmbeddingStore,
)

__all__ = [
    "AntibodySequenceInput",
    "EmbeddingBatch",
    "EmbeddingExtractor",
    "EmbeddingItem",
    "EmbeddingNotFoundError",
    "EmbeddingRequest",
    "EmbeddingStore",
    "Esm2EmbeddingExtractor",
    "IgBertEmbeddingExtractor",
    "InMemoryEmbeddingStore",
    "PairEmbeddingBatch",
    "ShardedEmbeddingStore",
    "antibody_embedding_request",
    "antibody_sequence_hash",
    "antigen_embedding_request",
    "antigen_sequence_hash",
    "build_embedding_extractor",
    "collect_embedding_requests",
    "collate_embedding_batch",
    "collate_pair_embedding_batch",
    "register_embedding_extractor",
    "registered_embedding_extractors",
    "write_embedding_cache",
]
