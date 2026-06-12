"""External user-facing entry points (spec docs/programming_spec.md §5.8).

This is the only module external users are expected to call directly. It
hides `RankBatch`/`PairBatch`/`AffinityExample`, `group_id`/`pair_id`, and
`ranknet_loss` entirely (spec §5.8 rule 1): the public surface is
`AntibodyInput`, `load_model`, `score_antibodies`, and `rank_antibodies`.

Model + tokenizer bundling
---------------------------
`score_antibodies`/`rank_antibodies` need tokenizers to build a `RankBatch`
(spec §5.3), but their spec signature takes only `model: AffinityRanker`.
This module resolves that by attaching the tokenizers `build_model_and_tokenizers`
(spec §5.7) returns as plain (non-`nn.Module`) attributes on the returned
`AffinityRanker` instance:

    model.antibody_tokenizer: Tokenizer
    model.antigen_tokenizer: Tokenizer | None

`load_model` sets these attributes. `score_antibodies`/`rank_antibodies` read
them via `getattr` and raise `ValueError` if `antibody_tokenizer` is missing
-- e.g. for an `AffinityRanker` that was constructed directly rather than via
`load_model` (tests attach `FakeTokenizer` the same way). This keeps
`AffinityRanker` itself (`model.py`, spec §5.4) free of any tokenizer
dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import pandas as pd
import torch

from .config import load_config
from .dataloader import RankBatch, Tokenizer, collate_rank_batch
from .dataset import AffinityExample
from .model import AffinityRanker
from .trainer import build_model_and_tokenizers
from .utils import get_logger, validate_amino_acid_sequence

_logger = get_logger(__name__)

#: Columns of the `pd.DataFrame` returned by `score_antibodies`/`rank_antibodies`
#: (spec §5.8 "输出").
OUTPUT_COLUMNS = ("antibody_id", "score", "rank")

#: `AntibodyInput.antibody_type` values supported by `score_antibodies` (spec
#: §5.8: "Fab" | "IgG" | "unknown" are not yet supported).
SUPPORTED_ANTIBODY_TYPES = frozenset({"Fv", "scFv", "VHH"})


@dataclass
class AntibodyInput:
    """One antibody supplied by an external user (spec §5.8).

    Attributes:
        antibody_id: User-chosen identifier, returned verbatim in the output.
        heavy_chain: Heavy-chain (or VHH) sequence, or `None`.
        light_chain: Light-chain sequence, or `None`.
        single_chain_sequence: Single-chain sequence (e.g. scFv), or `None`.
        antibody_type: One of `SUPPORTED_ANTIBODY_TYPES` (`"Fv"`, `"scFv"`,
            `"VHH"`). `"Fab"`, `"IgG"`, `"unknown"` are not yet supported
            (spec §5.8).
    """

    antibody_id: str
    heavy_chain: str | None
    light_chain: str | None
    single_chain_sequence: str | None
    antibody_type: Literal["Fv", "scFv", "VHH"]


def load_model(checkpoint_path: Path, config_path: Path | None = None) -> AffinityRanker:
    """Load a trained model from a checkpoint (spec §5.8).

    Args:
        checkpoint_path: Path to a checkpoint written by
            `Trainer.save_checkpoint` (spec §5.7 rule 4), containing at least
            `"model_state_dict"` and `"config"`.
        config_path: If given, the model architecture is built from
            `load_config(config_path).model` instead of
            `checkpoint["config"].model`. Use this to load a checkpoint whose
            embedded config is unavailable or out of date; the checkpoint's
            `model_state_dict` must still be compatible with the resulting
            architecture.

    Returns:
        An `AffinityRanker` (spec §5.4) in eval mode, with its trained
        weights loaded, plus two extra attributes used by
        `score_antibodies`/`rank_antibodies` (see module docstring):
            - `antibody_tokenizer`: `Tokenizer` for the antibody encoder.
            - `antigen_tokenizer`: `Tokenizer | None` for the antigen encoder.

    Raises:
        FileNotFoundError: If `checkpoint_path` (or `config_path`) does not
            exist.
        ValueError: Propagated from `build_model_and_tokenizers` (spec §5.7)
            if the resolved `ModelConfig` uses an unsupported encoder.
        RuntimeError: Propagated from `load_state_dict` if `model_state_dict`
            does not match the architecture built from the resolved config.
    """
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")

    if config_path is not None:
        config = load_config(Path(config_path))
    else:
        config = checkpoint["config"]

    model, antibody_tokenizer, antigen_tokenizer = build_model_and_tokenizers(config.model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    model.antibody_tokenizer = antibody_tokenizer  # type: ignore[attr-defined]
    model.antigen_tokenizer = antigen_tokenizer  # type: ignore[attr-defined]
    return model


def score_antibodies(
    antigen_sequence: str | None,
    antibodies: Sequence[AntibodyInput],
    model: AffinityRanker,
) -> pd.DataFrame:
    """Score each antibody against one antigen (spec §5.8).

    Args:
        antigen_sequence: Antigen amino-acid sequence, or `None` if unknown
            (spec §5.8 rule 4: the model's missing-antigen branch is used --
            `collate_rank_batch`/`AffinityRanker.forward` already handle
            `antigen_sequence is None`, including the case where `model` has
            no antigen encoder at all).
        antibodies: Antibodies to score. Must be non-empty.
        model: A model returned by `load_model` (or any `AffinityRanker`
            with `antibody_tokenizer`/`antigen_tokenizer` attributes attached
            the same way, e.g. in tests).

    Returns:
        `pd.DataFrame` with columns `antibody_id, score, rank` (`OUTPUT_COLUMNS`),
        one row per element of `antibodies`, in the same order as `antibodies`.
        `rank` is `1` for the highest `score` (ties share the same rank, spec
        §5.8 rule 2 -- see `rank_antibodies` for a result sorted by `score`).

    Raises:
        ValueError: If `antibodies` is empty; if `antigen_sequence` is not
            `None` and fails `validate_amino_acid_sequence` (spec §5.8 rule
            3); if any `AntibodyInput.antibody_type` is not in
            `SUPPORTED_ANTIBODY_TYPES`; if any `AntibodyInput` has no usable
            antibody sequence, or a provided chain fails
            `validate_amino_acid_sequence` (spec §5.8 rule 3); or if `model`
            has no `antibody_tokenizer` attribute attached.
    """
    if not antibodies:
        raise ValueError("antibodies must be non-empty")

    if antigen_sequence is not None and not validate_amino_acid_sequence(antigen_sequence):
        raise ValueError(f"antigen_sequence is not a valid amino-acid sequence: {antigen_sequence!r}")

    antibody_tokenizer = getattr(model, "antibody_tokenizer", None)
    if antibody_tokenizer is None:
        raise ValueError(
            "model has no 'antibody_tokenizer' attribute attached. Build it with "
            "load_model(), or attach one manually (model.antibody_tokenizer = ...; "
            "see user_entry module docstring)."
        )
    antigen_tokenizer: Tokenizer | None = getattr(model, "antigen_tokenizer", None)

    examples = [_to_affinity_example(antibody, antigen_sequence) for antibody in antibodies]
    batch = collate_rank_batch(examples, antibody_tokenizer, antigen_tokenizer)

    device = next(model.parameters()).device
    batch = _move_rank_batch(batch, device)

    model.eval()
    with torch.no_grad():
        scores = model(batch)

    result = pd.DataFrame({
        "antibody_id": [antibody.antibody_id for antibody in antibodies],
        "score": scores.detach().cpu().tolist(),
    })
    result["rank"] = result["score"].rank(method="min", ascending=False).astype(int)
    return result[list(OUTPUT_COLUMNS)]


def rank_antibodies(
    antigen_sequence: str | None,
    antibodies: Sequence[AntibodyInput],
    model: AffinityRanker,
) -> pd.DataFrame:
    """Rank antibodies against one antigen by descending score (spec §5.8).

    Args:
        antigen_sequence: See `score_antibodies`.
        antibodies: See `score_antibodies`.
        model: See `score_antibodies`.

    Returns:
        The `score_antibodies` result (columns `antibody_id, score, rank`),
        sorted by `score` descending (spec §5.8 rule 2), with the index
        reset. Ties keep their relative input order (stable sort).

    Raises:
        ValueError: Same as `score_antibodies`.
    """
    result = score_antibodies(antigen_sequence, antibodies, model)
    return result.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)


def _to_affinity_example(antibody: AntibodyInput, antigen_sequence: str | None) -> AffinityExample:
    """Build the `AffinityExample` for one `AntibodyInput` (spec §5.8 -> §5.2).

    Args:
        antibody: One user-supplied antibody.
        antigen_sequence: Shared antigen sequence for this scoring call, or
            `None`.

    Returns:
        An `AffinityExample` with placeholder `rank_label=0.0`,
        `label_kind="unknown"`, `dataset_id`/`group_id="user_query"` (not
        used by `collate_rank_batch`/`AffinityRanker.forward`, but required
        fields of `AffinityExample`).

    Raises:
        ValueError: If `antibody.antibody_type` is not in
            `SUPPORTED_ANTIBODY_TYPES`; if `heavy_chain`, `light_chain`, or
            `single_chain_sequence` is set but fails
            `validate_amino_acid_sequence`; or if all three are `None` (no
            usable antibody sequence).
    """
    if antibody.antibody_type not in SUPPORTED_ANTIBODY_TYPES:
        raise ValueError(
            f"AntibodyInput {antibody.antibody_id!r} has unsupported antibody_type "
            f"{antibody.antibody_type!r}. Supported: {sorted(SUPPORTED_ANTIBODY_TYPES)}."
        )

    chains = {
        "heavy_chain": antibody.heavy_chain,
        "light_chain": antibody.light_chain,
        "single_chain_sequence": antibody.single_chain_sequence,
    }
    for chain_name, chain in chains.items():
        if chain is not None and not validate_amino_acid_sequence(chain):
            raise ValueError(
                f"AntibodyInput {antibody.antibody_id!r} has an invalid {chain_name}: {chain!r}"
            )
    if all(chain is None for chain in chains.values()):
        raise ValueError(
            f"AntibodyInput {antibody.antibody_id!r} has no usable antibody sequence "
            "(heavy_chain, light_chain, and single_chain_sequence are all None)"
        )

    return AffinityExample(
        record_id=antibody.antibody_id,
        dataset_id="user_query",
        heavy_chain=antibody.heavy_chain,
        light_chain=antibody.light_chain,
        single_chain_sequence=antibody.single_chain_sequence,
        antibody_type=antibody.antibody_type,
        antigen_sequence=antigen_sequence,
        antigen_key=None,
        rank_label=0.0,
        label_kind="unknown",
        group_id="user_query",
    )


def _move_rank_batch(batch: RankBatch, device: torch.device) -> RankBatch:
    """Return a copy of `batch` with every tensor moved to `device`.

    Args:
        batch: A `RankBatch` (spec §5.3).
        device: Target device, e.g. `next(model.parameters()).device`.

    Returns:
        A new `RankBatch` with all tensor fields moved to `device`;
        `record_ids`/`group_ids` are passed through unchanged.
    """
    return RankBatch(
        antibody_tokens=batch.antibody_tokens.to(device),
        antibody_mask=batch.antibody_mask.to(device),
        antigen_tokens=None if batch.antigen_tokens is None else batch.antigen_tokens.to(device),
        antigen_mask=None if batch.antigen_mask is None else batch.antigen_mask.to(device),
        labels=batch.labels.to(device),
        record_ids=batch.record_ids,
        group_ids=batch.group_ids,
    )
