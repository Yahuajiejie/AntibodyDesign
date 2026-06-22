"""External user-facing prediction entry points.

The public API is built around competition-style inputs: users provide an
antigen sequence, candidate antibody sequences, and an optional `model_name`.
They do not pass `AffinityRanker`, tokenizers, checkpoints, or configs
directly. Those objects are bundled inside `AffinityPredictor`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import pandas as pd
import torch
import torch.nn as nn
import yaml

from .config import Config, load_config
from .dataloader import RankBatch, Tokenizer, collate_rank_batch
from .dataset import AffinityExample
from .embeddings import (
    EmbeddingBatch,
    EmbeddingExtractor,
    InMemoryEmbeddingStore,
    antibody_embedding_request,
    antigen_embedding_request,
    build_embedding_extractor,
    collate_embedding_batch,
)
from .embeddings.schema import AntibodySequenceInput
from .model import AffinityRanker, EmbeddingAffinityRanker
from .trainer import build_model_and_tokenizers
from .utils import validate_amino_acid_sequence

OUTPUT_COLUMNS = ("query_id", "antibody_id", "score", "rank", "model_name")
INPUT_COLUMNS = (
    "query_id",
    "antibody_id",
    "antigen_sequence",
    "heavy_chain",
    "light_chain",
    "single_chain_sequence",
    "antibody_type",
)
SUPPORTED_ANTIBODY_TYPES = frozenset({"Fv", "scFv", "VHH", "Fab", "IgG", "unknown"})


@dataclass
class AntibodyInput:
    """One antibody supplied by an external user."""

    antibody_id: str
    heavy_chain: str | None
    light_chain: str | None
    single_chain_sequence: str | None
    antibody_type: Literal["Fv", "scFv", "VHH", "Fab", "IgG", "unknown"]


@dataclass
class AffinityPredictor:
    """Inference bundle: model + tokenizers + config."""

    model_name: str
    model: nn.Module
    config: Config
    antibody_tokenizer: Tokenizer | None
    antigen_tokenizer: Tokenizer | None
    checkpoint_path: Path
    antibody_extractor: EmbeddingExtractor | None = None
    antigen_extractor: EmbeddingExtractor | None = None


def load_predictor(
    model_name: str = "best",
    registry_path: Path | None = None,
) -> AffinityPredictor:
    """Load a named predictor from `configs/model_registry.yaml`.

    Args:
        model_name: Name under `models` in the registry. The literal
            `"default"` resolves to the registry's `default` entry.
        registry_path: Optional path to a model registry YAML.

    Returns:
        An `AffinityPredictor` ready for inference.

    Raises:
        FileNotFoundError: If the registry, config, or checkpoint is missing.
        ValueError: If `model_name` is unknown or the registry is malformed.
    """
    registry_path = Path("configs/model_registry.yaml") if registry_path is None else Path(registry_path)
    registry = _load_registry(registry_path)
    if model_name == "default":
        model_name = str(registry.get("default", "best"))

    models = registry.get("models")
    if not isinstance(models, dict):
        raise ValueError(f"Model registry {registry_path} must contain a 'models' mapping")
    if model_name not in models:
        raise ValueError(
            f"Unknown model_name {model_name!r}. Available models: {sorted(models)}"
        )

    entry = models[model_name]
    if not isinstance(entry, dict):
        raise ValueError(f"Registry entry for model {model_name!r} must be a mapping")
    checkpoint_path = _registry_path(registry_path, entry.get("checkpoint_path"), "checkpoint_path")
    config_path = _registry_path(registry_path, entry.get("config_path"), "config_path")

    checkpoint = _torch_load_checkpoint(checkpoint_path, map_location="cpu")
    config = load_config(config_path)
    if config.model.antibody_encoder.mode == "frozen_cached":
        (
            model,
            antibody_extractor,
            antigen_extractor,
        ) = _build_cached_predictor_components(config, checkpoint)
        antibody_tokenizer = None
        antigen_tokenizer = None
    else:
        model, antibody_tokenizer, antigen_tokenizer = build_model_and_tokenizers(config.model)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(torch.device(config.train.device))
        model.eval()
        antibody_extractor = None
        antigen_extractor = None

    return AffinityPredictor(
        model_name=model_name,
        model=model,
        config=config,
        antibody_tokenizer=antibody_tokenizer,
        antigen_tokenizer=antigen_tokenizer,
        checkpoint_path=checkpoint_path,
        antibody_extractor=antibody_extractor,
        antigen_extractor=antigen_extractor,
    )


def load_model(checkpoint_path: Path, config_path: Path | None = None) -> nn.Module:
    """Load a bare ranker for internal/developer use.

    This function intentionally returns only the tensor model. It does not
    attach tokenizers and is not the competition-facing entry point.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint = _torch_load_checkpoint(checkpoint_path, map_location="cpu")
    config = load_config(Path(config_path)) if config_path is not None else checkpoint["config"]

    if config.model.antibody_encoder.mode == "frozen_cached":
        return _build_cached_ranker(config, checkpoint)
    model, _, _ = build_model_and_tokenizers(config.model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def score_antibodies(
    antigen_sequence: str | None,
    antibodies: Sequence[AntibodyInput],
    model_name: str = "best",
) -> pd.DataFrame:
    """Score antibodies against one antigen using a named model."""
    predictor = load_predictor(model_name)
    return score_antibodies_with_predictor(antigen_sequence, antibodies, predictor)


def rank_antibodies(
    antigen_sequence: str | None,
    antibodies: Sequence[AntibodyInput],
    model_name: str = "best",
) -> pd.DataFrame:
    """Rank antibodies against one antigen using a named model."""
    predictor = load_predictor(model_name)
    return rank_antibodies_with_predictor(antigen_sequence, antibodies, predictor)


def rank_antibody_table(
    input_table: pd.DataFrame,
    model_name: str = "best",
) -> pd.DataFrame:
    """Rank a batch table, loading the named predictor once."""
    predictor = load_predictor(model_name)
    return rank_antibody_table_with_predictor(input_table, predictor)


def score_antibodies_with_predictor(
    antigen_sequence: str | None,
    antibodies: Sequence[AntibodyInput],
    predictor: AffinityPredictor,
    query_id: str = "query_0",
) -> pd.DataFrame:
    """Score antibodies against one antigen using an already loaded predictor."""
    if not antibodies:
        raise ValueError("antibodies must be non-empty")
    _validate_optional_sequence(antigen_sequence, "antigen_sequence")

    examples = [
        _to_affinity_example(antibody, antigen_sequence, query_id) for antibody in antibodies
    ]
    if predictor.antibody_extractor is None:
        if predictor.antibody_tokenizer is None:
            raise ValueError("online predictor is missing its antibody tokenizer")
        batch = collate_rank_batch(
            examples,
            predictor.antibody_tokenizer,
            predictor.antigen_tokenizer,
        )
        model_batch: RankBatch | EmbeddingBatch = _move_rank_batch(
            batch, next(predictor.model.parameters()).device
        )
    else:
        model_batch = _build_online_embedding_batch(examples, predictor)

    predictor.model.eval()
    with torch.no_grad():
        scores = predictor.model(model_batch)

    result = pd.DataFrame({
        "query_id": query_id,
        "antibody_id": [antibody.antibody_id for antibody in antibodies],
        "score": scores.detach().cpu().tolist(),
        "model_name": predictor.model_name,
    })
    result["rank"] = result["score"].rank(method="min", ascending=False).astype(int)
    return result[list(OUTPUT_COLUMNS)]


def rank_antibodies_with_predictor(
    antigen_sequence: str | None,
    antibodies: Sequence[AntibodyInput],
    predictor: AffinityPredictor,
    query_id: str = "query_0",
) -> pd.DataFrame:
    """Return `score_antibodies_with_predictor` sorted by score descending."""
    result = score_antibodies_with_predictor(antigen_sequence, antibodies, predictor, query_id)
    return result.sort_values(["query_id", "score"], ascending=[True, False], kind="stable").reset_index(drop=True)


def rank_antibody_table_with_predictor(
    input_table: pd.DataFrame,
    predictor: AffinityPredictor,
) -> pd.DataFrame:
    """Rank every `query_id` group in an input table."""
    _validate_input_table(input_table)

    outputs: list[pd.DataFrame] = []
    for query_id, group in input_table.groupby("query_id", sort=False):
        antigen_sequence = _single_antigen_sequence(group, str(query_id))
        antibodies = [_row_to_antibody_input(row) for _, row in group.iterrows()]
        outputs.append(
            rank_antibodies_with_predictor(
                antigen_sequence,
                antibodies,
                predictor,
                query_id=str(query_id),
            )
        )
    if not outputs:
        raise ValueError("input_table must contain at least one row")
    return pd.concat(outputs, ignore_index=True)[list(OUTPUT_COLUMNS)]


def _load_registry(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Model registry not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Model registry must contain a top-level mapping: {path}")
    return raw


def _torch_load_checkpoint(path: Path, map_location: str):
    """Load project checkpoints that may contain Config dataclasses."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _build_cached_predictor_components(
    config: Config,
    checkpoint: dict[str, object],
) -> tuple[EmbeddingAffinityRanker, EmbeddingExtractor, EmbeddingExtractor | None]:
    """Rebuild the embedding-native model and online frozen encoders.

    ``frozen_cached`` describes how embeddings were supplied during training,
    not a requirement that every future query already exists in that cache.
    New user sequences are encoded online with the same frozen adapter
    contract and are then passed to the trained embedding-native ranker.
    """
    model = _build_cached_ranker(config, checkpoint)
    interaction = config.model.interaction

    antibody_extractor = _build_online_extractor(
        config.model.antibody_encoder,
        config.train.device,
    )
    antigen_extractor = (
        None
        if config.model.antigen_encoder is None
        else _build_online_extractor(config.model.antigen_encoder, config.train.device)
    )
    if interaction.kind != "antibody_only" and antigen_extractor is None:
        raise ValueError(f"{interaction.kind} predictor requires an antigen encoder")
    return model, antibody_extractor, antigen_extractor


def _build_cached_ranker(
    config: Config,
    checkpoint: dict[str, object],
) -> EmbeddingAffinityRanker:
    raw_state = checkpoint.get("model_state_dict")
    if not isinstance(raw_state, dict):
        raise ValueError("checkpoint is missing model_state_dict")
    state: dict[str, torch.Tensor] = raw_state  # type: ignore[assignment]
    antibody_dim = _projection_input_dim(state, "antibody_projection")
    interaction = config.model.interaction
    antigen_dim = (
        None
        if interaction.kind == "antibody_only"
        else _projection_input_dim(state, "antigen_projection")
    )
    model = EmbeddingAffinityRanker(
        antibody_input_dim=antibody_dim,
        antigen_input_dim=antigen_dim,
        d_model=interaction.d_model,
        fusion_kind=interaction.kind,  # type: ignore[arg-type]
        num_layers=interaction.num_layers,
        num_heads=interaction.num_heads,
        ffn_multiplier=interaction.ffn_multiplier,
        dropout=interaction.dropout,
        pooling=interaction.pooling,
        bidirectional=interaction.bidirectional,
    )
    model.load_state_dict(state, strict=True)
    model.to(torch.device(config.train.device))
    model.eval()

    return model


def _projection_input_dim(
    state: dict[str, torch.Tensor],
    prefix: str,
) -> int:
    key = f"{prefix}.input_norm.weight"
    value = state.get(key)
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise ValueError(
            f"checkpoint does not contain an embedding-native {prefix}: missing {key!r}"
        )
    return int(value.shape[0])


def _build_online_extractor(encoder_config, device: str) -> EmbeddingExtractor:
    if encoder_config.mode != "frozen_cached":
        raise ValueError(
            "embedding-native inference requires frozen_cached encoder metadata, "
            f"got {encoder_config.mode!r}"
        )
    if encoder_config.long_sequence_strategy == "chunk":
        raise NotImplementedError(
            "online inference for long_sequence_strategy='chunk' is not implemented"
        )
    normalized_name = encoder_config.name.lower()
    if "igbert" in normalized_name:
        extractor_name = "igbert"
    elif "esm2" in normalized_name:
        extractor_name = "esm2"
    else:
        raise ValueError(
            f"cannot infer embedding adapter for encoder {encoder_config.name!r}"
        )
    return build_embedding_extractor(
        extractor_name,
        model_name=encoder_config.name,
        revision=encoder_config.revision,
        tokenizer_revision=encoder_config.tokenizer_revision,
        device=device,
        embedding_layer=encoder_config.embedding_layer,
        output_dtype=torch.float16,
        max_length=encoder_config.max_length,
        long_sequence_strategy=encoder_config.long_sequence_strategy,
    )


def _build_online_embedding_batch(
    examples: Sequence[AffinityExample],
    predictor: AffinityPredictor,
) -> EmbeddingBatch:
    antibody_extractor = predictor.antibody_extractor
    assert antibody_extractor is not None
    antibody_requests = [
        antibody_embedding_request(
            AntibodySequenceInput(
                heavy_chain=example.heavy_chain,
                light_chain=example.light_chain,
                single_chain_sequence=example.single_chain_sequence,
                antibody_type=example.antibody_type,
            )
        )
        for example in examples
    ]
    antibody_items = antibody_extractor.encode(antibody_requests)
    antibody_store = InMemoryEmbeddingStore()
    for request in antibody_requests:
        item = antibody_items.get(request.sequence_hash)
        if item is None:
            raise ValueError(
                "antibody extractor omitted sequence_hash="
                f"{request.sequence_hash}"
            )
        antibody_store.put(request.sequence_hash, "antibody", item)

    antigen_store: InMemoryEmbeddingStore | None = None
    antigen_requests = [
        antigen_embedding_request(example.antigen_sequence)
        for example in examples
        if example.antigen_sequence is not None
    ]
    if antigen_requests:
        if predictor.antigen_extractor is None:
            # Antibody-only models deliberately ignore antigen input.
            if predictor.config.model.interaction.kind != "antibody_only":
                raise ValueError("predictor is missing its antigen embedding extractor")
        else:
            antigen_items = predictor.antigen_extractor.encode(antigen_requests)
            antigen_store = InMemoryEmbeddingStore()
            for request in antigen_requests:
                item = antigen_items.get(request.sequence_hash)
                if item is None:
                    raise ValueError(
                        "antigen extractor omitted sequence_hash="
                        f"{request.sequence_hash}"
                    )
                antigen_store.put(request.sequence_hash, "antigen", item)

    batch = collate_embedding_batch(examples, antibody_store, antigen_store)
    return _move_embedding_batch(batch, next(predictor.model.parameters()).device)


def _registry_path(registry_path: Path, value: object, field_name: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"Registry field {field_name!r} must be a path string, got {value!r}")
    path = Path(value)
    if not path.is_absolute():
        path = registry_path.parent.parent / path if registry_path.parent.name == "configs" else registry_path.parent / path
    if not path.exists():
        raise FileNotFoundError(f"Registry field {field_name!r} points to a missing path: {path}")
    return path


def _validate_input_table(input_table: pd.DataFrame) -> None:
    missing = [column for column in INPUT_COLUMNS if column not in input_table.columns]
    if missing:
        raise ValueError(f"input_table is missing required column(s): {missing}")
    if input_table.empty:
        raise ValueError("input_table must contain at least one row")
    if input_table[["query_id", "antibody_id"]].isna().any().any():
        raise ValueError("query_id and antibody_id must not contain missing values")
    duplicated = input_table.duplicated(subset=["query_id", "antibody_id"])
    if duplicated.any():
        rows = input_table.loc[duplicated, ["query_id", "antibody_id"]].to_dict(orient="records")
        raise ValueError(f"(query_id, antibody_id) must be unique; duplicates: {rows[:10]}")


def _single_antigen_sequence(group: pd.DataFrame, query_id: str) -> str | None:
    values = {_optional_str(value) for value in group["antigen_sequence"].tolist()}
    if len(values) > 1:
        raise ValueError(
            f"query_id {query_id!r} has inconsistent antigen_sequence values"
        )
    antigen_sequence = next(iter(values))
    _validate_optional_sequence(antigen_sequence, "antigen_sequence")
    return antigen_sequence


def _row_to_antibody_input(row: pd.Series) -> AntibodyInput:
    antibody_type = _optional_str(row["antibody_type"])
    if antibody_type is None:
        raise ValueError(f"antibody_id {row['antibody_id']!r} has missing antibody_type")
    return AntibodyInput(
        antibody_id=str(row["antibody_id"]),
        heavy_chain=_optional_str(row["heavy_chain"]),
        light_chain=_optional_str(row["light_chain"]),
        single_chain_sequence=_optional_str(row["single_chain_sequence"]),
        antibody_type=antibody_type,  # type: ignore[arg-type]
    )


def _to_affinity_example(
    antibody: AntibodyInput,
    antigen_sequence: str | None,
    query_id: str,
) -> AffinityExample:
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
        _validate_optional_sequence(chain, f"{antibody.antibody_id}.{chain_name}")
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
        group_id=query_id,
    )


def _optional_str(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _validate_optional_sequence(sequence: str | None, field_name: str) -> None:
    if sequence is not None and not validate_amino_acid_sequence(sequence):
        raise ValueError(f"{field_name} is not a valid amino-acid sequence: {sequence!r}")


def _move_rank_batch(batch: RankBatch, device: torch.device) -> RankBatch:
    return RankBatch(
        antibody_tokens=batch.antibody_tokens.to(device),
        antibody_mask=batch.antibody_mask.to(device),
        antigen_tokens=None if batch.antigen_tokens is None else batch.antigen_tokens.to(device),
        antigen_mask=None if batch.antigen_mask is None else batch.antigen_mask.to(device),
        labels=batch.labels.to(device),
        record_ids=batch.record_ids,
        group_ids=batch.group_ids,
    )


def _move_embedding_batch(batch: EmbeddingBatch, device: torch.device) -> EmbeddingBatch:
    return EmbeddingBatch(
        antibody_embeddings=batch.antibody_embeddings.to(device),
        antibody_mask=batch.antibody_mask.to(device),
        antigen_embeddings=(
            None if batch.antigen_embeddings is None else batch.antigen_embeddings.to(device)
        ),
        antigen_mask=None if batch.antigen_mask is None else batch.antigen_mask.to(device),
        labels=batch.labels.to(device),
        record_ids=batch.record_ids,
        group_ids=batch.group_ids,
    )
