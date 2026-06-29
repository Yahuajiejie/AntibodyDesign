"""Tests for the explicit base-model adapter registry."""

from affinity_transformer.embeddings import (
    build_embedding_extractor,
    register_embedding_extractor,
    registered_embedding_extractors,
)
from affinity_transformer.embeddings import extractors as extractor_module


class DummyExtractor:
    encoder_name = "dummy"
    encoder_revision = "revision"

    def encode(self, requests):
        return {}

    def metadata(self):
        return {"kind": "dummy"}


def test_extractor_registry_builds_explicit_adapter(monkeypatch):
    monkeypatch.setattr(extractor_module, "_EXTRACTOR_FACTORIES", {})
    register_embedding_extractor("Dummy", lambda **_: DummyExtractor())

    extractor = build_embedding_extractor("dummy")

    assert extractor.encoder_name == "dummy"
    assert registered_embedding_extractors() == ("dummy",)


def test_extractor_registry_never_silently_falls_back(monkeypatch):
    monkeypatch.setattr(extractor_module, "_EXTRACTOR_FACTORIES", {})

    try:
        build_embedding_extractor("unknown")
    except ValueError as exc:
        assert "registered: []" in str(exc)
    else:
        raise AssertionError("unknown extractor should fail")
