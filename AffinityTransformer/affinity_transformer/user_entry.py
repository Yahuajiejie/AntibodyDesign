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
import yaml

from .config import Config, load_config
from .dataloader import RankBatch, Tokenizer, collate_rank_batch
from .dataset import AffinityExample
from .model import AffinityRanker
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
    model: AffinityRanker
    config: Config
    antibody_tokenizer: Tokenizer
    antigen_tokenizer: Tokenizer | None
    checkpoint_path: Path


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

    config = load_config(config_path)
    model, antibody_tokenizer, antigen_tokenizer = build_model_and_tokenizers(config.model)
    checkpoint = _torch_load_checkpoint(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(torch.device(config.train.device))
    model.eval()

    return AffinityPredictor(
        model_name=model_name,
        model=model,
        config=config,
        antibody_tokenizer=antibody_tokenizer,
        antigen_tokenizer=antigen_tokenizer,
        checkpoint_path=checkpoint_path,
    )


def load_model(checkpoint_path: Path, config_path: Path | None = None) -> AffinityRanker:
    """Load a bare `AffinityRanker` for internal/developer use.

    This function intentionally returns only the tensor model. It does not
    attach tokenizers and is not the competition-facing entry point.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint = _torch_load_checkpoint(checkpoint_path, map_location="cpu")
    config = load_config(Path(config_path)) if config_path is not None else checkpoint["config"]

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
    batch = collate_rank_batch(
        examples,
        predictor.antibody_tokenizer,
        predictor.antigen_tokenizer,
    )
    batch = _move_rank_batch(batch, next(predictor.model.parameters()).device)

    predictor.model.eval()
    with torch.no_grad():
        scores = predictor.model(batch)

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
