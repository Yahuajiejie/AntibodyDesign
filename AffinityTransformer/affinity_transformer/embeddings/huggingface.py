"""Hugging Face token-embedding adapters for supported foundation models."""

from __future__ import annotations

from collections import OrderedDict
from typing import Mapping, Sequence

import torch

from .schema import EmbeddingItem, EmbeddingRequest


class _HuggingFaceTokenExtractor:
    """Shared frozen-forward implementation; subclasses own input formatting."""

    def __init__(
        self,
        *,
        encoder_name: str,
        encoder_revision: str,
        model: torch.nn.Module,
        tokenizer: object,
        device: str | torch.device = "cpu",
        embedding_layer: int = -1,
        output_dtype: torch.dtype = torch.float16,
        max_length: int | None = None,
    ) -> None:
        if max_length is not None and max_length < 1:
            raise ValueError("max_length must be None or >= 1")
        if not output_dtype.is_floating_point:
            raise ValueError("output_dtype must be floating point")
        self.encoder_name = encoder_name
        self.encoder_revision = encoder_revision
        self.model = model.to(torch.device(device))
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.embedding_layer = embedding_layer
        self.output_dtype = output_dtype
        self.max_length = max_length

    def encode(
        self,
        requests: Sequence[EmbeddingRequest],
    ) -> Mapping[str, EmbeddingItem]:
        """Encode unique requests and return unpadded, non-special token states."""
        unique = _unique_requests(requests)
        if not unique:
            return {}

        sequences: list[str] = []
        spans: list[tuple[int, int]] = []
        for request in unique:
            formatted = self._format_request(request)
            if not formatted:
                raise ValueError(f"request produced no model inputs: {request.sequence_hash}")
            start = len(sequences)
            sequences.extend(formatted)
            spans.append((start, len(sequences)))

        encoded = self._tokenize(sequences)
        model_inputs = {
            key: value.to(self.device)
            for key, value in encoded.items()
            if key in {"input_ids", "attention_mask", "token_type_ids"}
        }
        with torch.inference_mode():
            outputs = self.model(
                **model_inputs,
                output_hidden_states=self.embedding_layer != -1,
            )
        hidden = (
            outputs.last_hidden_state
            if self.embedding_layer == -1
            else outputs.hidden_states[self.embedding_layer]
        )
        attention_mask = encoded["attention_mask"].bool()
        special_tokens_mask = encoded["special_tokens_mask"].bool()

        result: dict[str, EmbeddingItem] = {}
        for request, (start, end) in zip(unique, spans):
            pieces: list[torch.Tensor] = []
            for index in range(start, end):
                valid = attention_mask[index] & ~special_tokens_mask[index]
                piece = hidden[index, valid.to(hidden.device)]
                if piece.shape[0] == 0:
                    raise ValueError(
                        f"all tokens were removed as padding/special tokens for "
                        f"sequence_hash={request.sequence_hash}"
                    )
                pieces.append(piece)
            values = torch.cat(pieces, dim=0).to(dtype=self.output_dtype, device="cpu")
            result[request.sequence_hash] = EmbeddingItem.from_values(values)
        return result

    def metadata(self) -> Mapping[str, object]:
        """Describe the frozen token-embedding extraction contract."""
        return {
            "adapter_class": type(self).__name__,
            "embedding_layer": self.embedding_layer,
            "output_dtype": str(self.output_dtype).removeprefix("torch."),
            "max_length": self.max_length,
            "base_model_frozen": True,
            "padding_removed": True,
            "special_tokens_removed": True,
        }

    def _tokenize(self, sequences: list[str]) -> Mapping[str, torch.Tensor]:
        kwargs: dict[str, object] = {
            "padding": True,
            "return_tensors": "pt",
            "return_special_tokens_mask": True,
        }
        if self.max_length is not None:
            kwargs.update(truncation=True, max_length=self.max_length)
        encoded = self.tokenizer(sequences, **kwargs)  # type: ignore[operator]
        required = {"input_ids", "attention_mask", "special_tokens_mask"}
        missing = sorted(required.difference(encoded))
        if missing:
            raise ValueError(f"tokenizer output is missing required field(s): {missing}")
        return encoded

    def _format_request(self, request: EmbeddingRequest) -> list[str]:
        raise NotImplementedError


class Esm2EmbeddingExtractor(_HuggingFaceTokenExtractor):
    """Frozen ESM-2 adapter for antibody chains and antigen sequences."""

    def _format_request(self, request: EmbeddingRequest) -> list[str]:
        if request.sequence_type == "antigen":
            assert request.antigen_sequence is not None
            return [request.antigen_sequence]

        assert request.antibody is not None
        antibody = request.antibody
        if antibody.single_chain_sequence is not None:
            return [antibody.single_chain_sequence]
        chains = [chain for chain in (antibody.heavy_chain, antibody.light_chain) if chain]
        return chains

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        revision: str = "main",
        **kwargs: object,
    ) -> "Esm2EmbeddingExtractor":
        """Load ESM-2 lazily; no network/model work happens at module import."""
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        model = AutoModel.from_pretrained(model_name, revision=revision)
        return cls(
            encoder_name=model_name,
            encoder_revision=revision,
            model=model,
            tokenizer=tokenizer,
            **kwargs,
        )


class IgBertEmbeddingExtractor(_HuggingFaceTokenExtractor):
    """Frozen IgBERT adapter using its documented spaced paired-chain input."""

    def _format_request(self, request: EmbeddingRequest) -> list[str]:
        if request.sequence_type != "antibody" or request.antibody is None:
            raise ValueError("IgBERT supports antibody embedding requests only")
        antibody = request.antibody
        if antibody.single_chain_sequence is not None:
            return [_space_residues(antibody.single_chain_sequence)]
        if antibody.heavy_chain and antibody.light_chain:
            return [
                f"{_space_residues(antibody.heavy_chain)} [SEP] "
                f"{_space_residues(antibody.light_chain)}"
            ]
        chain = antibody.heavy_chain or antibody.light_chain
        assert chain is not None
        return [_space_residues(chain)]

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "Exscientia/IgBert",
        *,
        revision: str = "main",
        **kwargs: object,
    ) -> "IgBertEmbeddingExtractor":
        """Load IgBERT lazily with its case-sensitive BERT tokenizer."""
        from transformers import BertModel, BertTokenizer

        tokenizer = BertTokenizer.from_pretrained(
            model_name,
            revision=revision,
            do_lower_case=False,
        )
        model = BertModel.from_pretrained(
            model_name,
            revision=revision,
            add_pooling_layer=False,
        )
        return cls(
            encoder_name=model_name,
            encoder_revision=revision,
            model=model,
            tokenizer=tokenizer,
            **kwargs,
        )


def _space_residues(sequence: str) -> str:
    return " ".join(sequence)


def _unique_requests(requests: Sequence[EmbeddingRequest]) -> list[EmbeddingRequest]:
    unique: OrderedDict[str, EmbeddingRequest] = OrderedDict()
    for request in requests:
        existing = unique.get(request.sequence_hash)
        if existing is not None and existing != request:
            raise ValueError(f"conflicting requests share sequence_hash={request.sequence_hash}")
        unique[request.sequence_hash] = request
    return list(unique.values())
