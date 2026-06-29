"""Network-free tests for Hugging Face embedding adapter behavior."""

from types import SimpleNamespace

import pytest
import torch

from affinity_transformer.embeddings import (
    AntibodySequenceInput,
    Esm2EmbeddingExtractor,
    IgBertEmbeddingExtractor,
    antibody_embedding_request,
    antigen_embedding_request,
)


class RecordingTokenizer:
    def __init__(self) -> None:
        self.sequences: list[str] = []

    def __call__(self, sequences, **kwargs):
        self.sequences = list(sequences)
        token_rows: list[list[str]] = []
        for sequence in sequences:
            residues = sequence.split() if " " in sequence else list(sequence)
            token_rows.append(["[CLS]", *residues, "[SEP]"])
        width = max(len(row) for row in token_rows)
        input_ids = torch.zeros(len(token_rows), width, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        special_tokens_mask = torch.ones_like(input_ids)
        for row_index, tokens in enumerate(token_rows):
            for token_index, token in enumerate(tokens):
                input_ids[row_index, token_index] = token_index + 1
                attention_mask[row_index, token_index] = 1
                special_tokens_mask[row_index, token_index] = int(token in {"[CLS]", "[SEP]"})
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "special_tokens_mask": special_tokens_mask,
        }


class FakeHuggingFaceModel(torch.nn.Module):
    def __init__(self, hidden_dim: int = 4) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_dim))

    def forward(self, input_ids, attention_mask, output_hidden_states=False, **kwargs):
        del attention_mask, kwargs
        hidden = input_ids.unsqueeze(-1).float() * self.weight
        hidden_states = (hidden * 0.5, hidden) if output_hidden_states else None
        return SimpleNamespace(last_hidden_state=hidden, hidden_states=hidden_states)


def _paired_antibody_request():
    return antibody_embedding_request(AntibodySequenceInput(
        heavy_chain="QVL",
        light_chain="DI",
        single_chain_sequence=None,
        antibody_type="Fv",
    ))


def test_esm2_encodes_heavy_and_light_as_separate_proteins_then_concatenates():
    tokenizer = RecordingTokenizer()
    model = FakeHuggingFaceModel()
    extractor = Esm2EmbeddingExtractor(
        encoder_name="fake-esm2",
        encoder_revision="rev",
        model=model,
        tokenizer=tokenizer,
        output_dtype=torch.float32,
    )
    request = _paired_antibody_request()

    item = extractor.encode([request])[request.sequence_hash]

    assert tokenizer.sequences == ["QVL", "DI"]
    assert item.values.shape == (5, 4)
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_igbert_uses_documented_spaced_paired_chain_format():
    tokenizer = RecordingTokenizer()
    extractor = IgBertEmbeddingExtractor(
        encoder_name="fake-igbert",
        encoder_revision="rev",
        model=FakeHuggingFaceModel(),
        tokenizer=tokenizer,
        output_dtype=torch.float32,
    )
    request = _paired_antibody_request()

    item = extractor.encode([request])[request.sequence_hash]

    assert tokenizer.sequences == ["Q V L [SEP] D I"]
    assert item.values.shape == (5, 4)


def test_igbert_rejects_antigen_requests():
    extractor = IgBertEmbeddingExtractor(
        encoder_name="fake-igbert",
        encoder_revision="rev",
        model=FakeHuggingFaceModel(),
        tokenizer=RecordingTokenizer(),
    )

    with pytest.raises(ValueError, match="antibody"):
        extractor.encode([antigen_embedding_request("MKT")])


def test_extractor_deduplicates_identical_requests_before_forward():
    tokenizer = RecordingTokenizer()
    extractor = Esm2EmbeddingExtractor(
        encoder_name="fake-esm2",
        encoder_revision="rev",
        model=FakeHuggingFaceModel(),
        tokenizer=tokenizer,
        output_dtype=torch.float32,
    )
    request = antigen_embedding_request("MKT")

    output = extractor.encode([request, request])

    assert list(output) == [request.sequence_hash]
    assert tokenizer.sequences == ["MKT"]
