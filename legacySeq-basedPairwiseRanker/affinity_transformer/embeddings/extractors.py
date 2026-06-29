"""Base-model embedding adapter protocol and registry.

Each foundation model owns its tokenization and chain-formatting rules. The
registry provides one common construction API without pretending those rules
are interchangeable.
"""

from __future__ import annotations

from typing import Callable, Mapping, Protocol, Sequence, runtime_checkable

from .schema import EmbeddingItem, EmbeddingRequest


@runtime_checkable
class EmbeddingExtractor(Protocol):
    """Interface implemented by ESM-2, IgBERT, AbLang, and future adapters."""

    encoder_name: str
    encoder_revision: str

    def encode(
        self,
        requests: Sequence[EmbeddingRequest],
    ) -> Mapping[str, EmbeddingItem]:
        """Encode requests and return items keyed by ``sequence_hash``."""
        ...

    def metadata(self) -> Mapping[str, object]:
        """Return serializable extraction settings written beside the cache."""
        ...


ExtractorFactory = Callable[..., EmbeddingExtractor]
_EXTRACTOR_FACTORIES: dict[str, ExtractorFactory] = {}


def register_embedding_extractor(
    name: str,
    factory: ExtractorFactory,
    *,
    replace: bool = False,
) -> None:
    """Register one explicit base-model adapter factory."""
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("embedding extractor name must be non-empty")
    if normalized in _EXTRACTOR_FACTORIES and not replace:
        raise ValueError(f"embedding extractor already registered: {normalized}")
    _EXTRACTOR_FACTORIES[normalized] = factory


def build_embedding_extractor(name: str, **kwargs: object) -> EmbeddingExtractor:
    """Build a registered adapter without silently falling back."""
    normalized = name.strip().lower()
    factory = _EXTRACTOR_FACTORIES.get(normalized)
    if factory is None:
        raise ValueError(
            f"unsupported embedding extractor {name!r}; "
            f"registered: {sorted(_EXTRACTOR_FACTORIES)}"
        )
    extractor = factory(**kwargs)
    if (
        not isinstance(getattr(extractor, "encoder_name", None), str)
        or not isinstance(getattr(extractor, "encoder_revision", None), str)
        or not callable(getattr(extractor, "encode", None))
        or not callable(getattr(extractor, "metadata", None))
    ):
        raise TypeError(f"factory {normalized!r} did not return an EmbeddingExtractor")
    return extractor


def registered_embedding_extractors() -> tuple[str, ...]:
    """Return registered adapter names in deterministic order."""
    return tuple(sorted(_EXTRACTOR_FACTORIES))


def _build_esm2(**kwargs: object) -> EmbeddingExtractor:
    from .huggingface import Esm2EmbeddingExtractor

    return Esm2EmbeddingExtractor.from_pretrained(**kwargs)


def _build_igbert(**kwargs: object) -> EmbeddingExtractor:
    from .huggingface import IgBertEmbeddingExtractor

    return IgBertEmbeddingExtractor.from_pretrained(**kwargs)


register_embedding_extractor("esm2", _build_esm2)
register_embedding_extractor("igbert", _build_igbert)
