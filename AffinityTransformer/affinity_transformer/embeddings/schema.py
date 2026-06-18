"""Model-agnostic contracts for cached token-level embeddings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

import torch

from ..utils import hash_text

SequenceType = Literal["antibody", "antigen"]


@dataclass(frozen=True)
class AntibodySequenceInput:
    """Structured antibody sequence supplied to a base-model adapter."""

    heavy_chain: str | None
    light_chain: str | None
    single_chain_sequence: str | None
    antibody_type: str

    def __post_init__(self) -> None:
        if not any((self.heavy_chain, self.light_chain, self.single_chain_sequence)):
            raise ValueError("antibody input must contain at least one sequence")


@dataclass(frozen=True)
class EmbeddingRequest:
    """One structured sequence request passed to an embedding adapter."""

    sequence_hash: str
    sequence_type: SequenceType
    antibody: AntibodySequenceInput | None = None
    antigen_sequence: str | None = None

    def __post_init__(self) -> None:
        if self.sequence_type == "antibody":
            if self.antibody is None or self.antigen_sequence is not None:
                raise ValueError("antibody request must contain only antibody input")
        elif self.sequence_type == "antigen":
            if not self.antigen_sequence or self.antibody is not None:
                raise ValueError("antigen request must contain only antigen_sequence")
        else:
            raise ValueError(f"unsupported sequence_type: {self.sequence_type!r}")


@dataclass(frozen=True)
class EmbeddingItem:
    """Unpadded token-level embedding and validity mask for one sequence."""

    values: torch.Tensor
    mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError(
                f"embedding values must have shape [L, D], got {tuple(self.values.shape)}"
            )
        if not self.values.is_floating_point():
            raise ValueError(f"embedding values must be floating point, got {self.values.dtype}")
        if self.values.device.type != "cpu":
            raise ValueError("cached embedding values must be CPU tensors")
        if self.values.shape[0] == 0 or self.values.shape[1] == 0:
            raise ValueError("embedding values must have non-zero sequence and feature dimensions")
        if self.mask.dtype != torch.bool or self.mask.ndim != 1:
            raise ValueError("embedding mask must be BoolTensor[L]")
        if self.mask.device.type != "cpu":
            raise ValueError("cached embedding mask must be a CPU tensor")
        if self.mask.shape[0] != self.values.shape[0]:
            raise ValueError(
                "embedding mask length must match values sequence length: "
                f"{self.mask.shape[0]} != {self.values.shape[0]}"
            )
        if not torch.isfinite(self.values).all():
            raise ValueError("embedding values contain NaN or infinity")

    @classmethod
    def from_values(
        cls,
        values: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> "EmbeddingItem":
        """Construct an item, defaulting to an all-valid token mask."""
        values = values.detach().cpu()
        if mask is None:
            mask = torch.ones(values.shape[0], dtype=torch.bool)
        else:
            mask = mask.detach().cpu().bool()
        return cls(values=values, mask=mask)


def antibody_sequence_hash(sequence: AntibodySequenceInput) -> str:
    """Return a stable cache key for a structured antibody input."""
    payload = {
        "antibody_type": sequence.antibody_type,
        "heavy_chain": sequence.heavy_chain,
        "light_chain": sequence.light_chain,
        "single_chain_sequence": sequence.single_chain_sequence,
    }
    return hash_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def antigen_sequence_hash(sequence: str) -> str:
    """Return a stable cache key for an antigen amino-acid sequence."""
    if not sequence:
        raise ValueError("antigen sequence must be non-empty")
    return hash_text(sequence)


def antibody_embedding_request(sequence: AntibodySequenceInput) -> EmbeddingRequest:
    """Build an antibody request with its deterministic cache key."""
    return EmbeddingRequest(
        sequence_hash=antibody_sequence_hash(sequence),
        sequence_type="antibody",
        antibody=sequence,
    )


def antigen_embedding_request(sequence: str) -> EmbeddingRequest:
    """Build an antigen request with its deterministic cache key."""
    return EmbeddingRequest(
        sequence_hash=antigen_sequence_hash(sequence),
        sequence_type="antigen",
        antigen_sequence=sequence,
    )
