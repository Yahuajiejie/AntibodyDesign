"""Experiment configuration loading (spec docs/programming_spec.md §5.1).

This module defines the dataclasses that fully describe one training run
(data paths, model architecture switches, training hyperparameters) and a
single entry point, :func:`load_config`, that turns a YAML file into a
:class:`Config`.

Ablation and control experiments are expressed purely as different YAML
files consumed by :func:`load_config`. Nothing in this module hardcodes a
dataset path, a model switch, or a default random seed -- every value comes
from the config file, and a missing value is an error rather than a silent
default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .record_filter import RecordFilterConfig, build_record_filter_config


@dataclass
class DataConfig:
    """Data loading and pairing parameters.

    Attributes:
        train_path: Path to the training `records.parquet`/`records.csv`
            (standard processed table, spec §3), or None when an automatic
            split will be built from `all_records_path`.
        valid_path: Path to the validation processed table, or None if no
            held-out validation set is configured.
        all_records_path: Path to the merged processed table used by automatic
            split strategies, or None in explicit split mode.
        test_path: Path to the test processed table, or None if no final test
            set is configured.
        split_strategy: One of "none", "debug_record_split", or
            "group_holdout_split". "none" means train/valid/test paths are
            supplied explicitly.
        split_dir: Directory where automatic splits are written, or None in
            explicit split mode.
        valid_fraction: Fraction of records/groups reserved for validation in
            automatic split mode.
        test_fraction: Fraction of records/groups reserved for test in
            automatic split mode.
        max_pairs_per_group: Maximum number of pairwise examples sampled
            per `group_id` (see `build_pairs`).
        pair_sample_strategy: Pair sampling policy passed to `build_pairs`.
            `"absolute_cap"` preserves the legacy behavior.
        pair_fraction: Fraction used by `"capped_proportional"` sampling.
        min_pairs_per_group: Lower target used by `"capped_proportional"`.
        large_group_threshold: Trainable-record threshold above which
            `build_pairs` uses memory-safe block sampling.
        pair_enumeration_limit: Candidate-pair threshold above which
            `build_pairs` refuses full pair enumeration.
        label_block_count: Number of rank-label quantile blocks used for
            large-group sampling.
        intra_block_pairs_per_large_group: Extra same-block fine-grained
            pairs sampled for each large group.
        discrete_label_unique_threshold: Unique-label threshold for treating
            a large group as discrete/repeated-label.
        discrete_label_ratio_threshold: Unique-label ratio threshold for
            treating a large group as discrete/repeated-label.
        seed: Random seed used for pair sampling and any other randomness
            in the data pipeline.
        record_filter: Optional selector applied to `all_records_path` before
            automatic splitting. Ignored when `split_strategy == "none"`.
    """

    train_path: Path | None
    valid_path: Path | None
    max_pairs_per_group: int
    seed: int
    pair_sample_strategy: str = "absolute_cap"
    pair_fraction: float | None = None
    min_pairs_per_group: int = 1
    large_group_threshold: int = 10_000
    pair_enumeration_limit: int = 100_000
    label_block_count: int = 5
    intra_block_pairs_per_large_group: int = 50
    discrete_label_unique_threshold: int = 32
    discrete_label_ratio_threshold: float = 0.05
    all_records_path: Path | None = None
    test_path: Path | None = None
    split_strategy: str = "none"
    split_dir: Path | None = None
    valid_fraction: float = 0.1
    test_fraction: float = 0.1
    record_filter: RecordFilterConfig = field(default_factory=RecordFilterConfig)


@dataclass(frozen=True)
class EncoderConfig:
    """One frozen/online foundation-model representation source."""

    name: str
    revision: str
    tokenizer_revision: str
    mode: str
    embedding_layer: int
    cache_dir: Path | None
    max_length: int | None
    long_sequence_strategy: str
    lora_rank: int | None = None
    lora_alpha: float | None = None
    lora_dropout: float | None = None


@dataclass(frozen=True)
class InteractionConfig:
    """Trainable fusion/projection/scoring architecture."""

    kind: str
    d_model: int
    num_layers: int
    num_heads: int
    ffn_multiplier: float
    dropout: float
    pooling: str
    bidirectional: bool


@dataclass(frozen=True)
class ObjectiveConfig:
    """Ranking objective selected by the Trainer."""

    name: str
    temperature: float
    sigma: float
    pointwise_loss: str


@dataclass(frozen=True)
class ModelConfig:
    """Embedding encoders, trainable interaction, and ranking objective."""

    antibody_encoder: EncoderConfig
    antigen_encoder: EncoderConfig | None
    interaction: InteractionConfig
    objective: ObjectiveConfig

    @property
    def d_model(self) -> int:
        """Compatibility view used by the legacy online model path."""
        return self.interaction.d_model

    @property
    def use_cross_attention(self) -> bool:
        """Compatibility view used by the legacy online model path."""
        return self.interaction.kind == "deep_cross_attention"


@dataclass
class TrainConfig:
    """Training loop hyperparameters.

    Attributes:
        batch_size: Number of pair examples per batch.
        lr: Optimizer learning rate.
        epochs: Number of training epochs.
        device: Torch device string, e.g. "cpu" or "cuda".
    """

    batch_size: int
    lr: float
    epochs: int
    device: str


@dataclass
class Config:
    """Top-level configuration for one training run.

    Attributes:
        data: Data loading and pairing parameters.
        model: Model architecture switches.
        train: Training loop hyperparameters.
    """

    data: DataConfig
    model: ModelConfig
    train: TrainConfig


_DATA_REQUIRED_KEYS = ("train_path", "valid_path", "max_pairs_per_group", "seed")
_VALID_SPLIT_STRATEGIES = {"none", "debug_record_split", "group_holdout_split"}
_TRAIN_REQUIRED_KEYS = ("batch_size", "lr", "epochs", "device")

_ENCODER_MODES = {"frozen_cached", "frozen_online", "lora_online"}
_LONG_SEQUENCE_STRATEGIES = {"error", "truncate", "chunk"}
_INTERACTION_KINDS = {"antibody_only", "concat", "deep_cross_attention"}
_POOLING_KINDS = {"masked_mean", "attention_pool"}
_OBJECTIVES = {"pointwise", "pairwise_ranknet", "listwise_listnet"}


def load_config(path: Path) -> Config:
    """Load and validate an experiment configuration from a YAML file.

    Args:
        path: Path to a YAML file with top-level `data`, `model`, and
            `train` sections matching `DataConfig`, `ModelConfig`, and
            `TrainConfig` respectively.

    Returns:
        A fully populated `Config`. Every field is taken verbatim from the
        file; no field is given a hidden default (including `data.seed`).

    Raises:
        FileNotFoundError: If `path` does not exist, or if `data.train_path`
            / `data.valid_path` (when not null) point to files that do not
            exist.
        ValueError: If `path` is not a YAML mapping, if a required section
            (`data`, `model`, `train`) is missing or not a mapping, or if a
            required field within a section is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a top-level mapping: {path}")

    data_cfg = _build_data_config(_require_section(raw, "data"))
    model_cfg = _build_model_config(_require_section(raw, "model"))
    train_cfg = _build_train_config(_require_section(raw, "train"))

    return Config(data=data_cfg, model=model_cfg, train=train_cfg)


def _require_section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    """Return `raw[name]` as a mapping, or raise `ValueError`.

    Args:
        raw: Parsed top-level config mapping.
        name: Name of the required section (`"data"`, `"model"`, `"train"`).

    Returns:
        The section as a dict.

    Raises:
        ValueError: If the section is missing or is not a mapping.
    """
    section = raw.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"Config is missing required section '{name}'")
    return section


def _require_keys(section: dict[str, Any], keys: tuple[str, ...], section_name: str) -> None:
    """Raise `ValueError` if any of `keys` is absent from `section`.

    Args:
        section: A config section mapping.
        keys: Field names required to be present (value may be `None`).
        section_name: Name of the section, used in the error message.

    Raises:
        ValueError: If one or more `keys` are missing from `section`.
    """
    missing = [key for key in keys if key not in section]
    if missing:
        raise ValueError(
            f"Config section '{section_name}' is missing required field(s): {missing}"
        )


def _require_existing_path(value: Any, field_name: str) -> Path:
    """Convert `value` to a `Path` and check that it exists.

    Args:
        value: Raw YAML value, expected to be a path string.
        field_name: Dotted field name, used in error messages.

    Returns:
        `value` as a `Path`.

    Raises:
        ValueError: If `value` is not a string or `Path`.
        FileNotFoundError: If the resulting path does not exist.
    """
    if not isinstance(value, (str, Path)):
        raise ValueError(f"Config field '{field_name}' must be a path string, got {value!r}")
    resolved = Path(value)
    if not resolved.exists():
        raise FileNotFoundError(f"Config field '{field_name}' points to a missing path: {resolved}")
    return resolved


def _optional_existing_path(value: Any, field_name: str) -> Path | None:
    """Convert `value` to an existing `Path`, preserving `None`.

    Args:
        value: Raw YAML value.
        field_name: Dotted field name, used in error messages.

    Returns:
        `None` if `value is None`, otherwise an existing `Path`.

    Raises:
        ValueError: If `value` is not None and not path-like.
        FileNotFoundError: If the resulting path does not exist.
    """
    if value is None:
        return None
    return _require_existing_path(value, field_name)


def _optional_path(value: Any, field_name: str) -> Path | None:
    """Convert `value` to `Path`, preserving `None` and not checking existence."""
    if value is None:
        return None
    if not isinstance(value, (str, Path)):
        raise ValueError(f"Config field '{field_name}' must be a path string or null, got {value!r}")
    return Path(value)


def _build_data_config(section: dict[str, Any]) -> DataConfig:
    """Build `DataConfig` from the `data` section of a config file.

    Args:
        section: The `data` mapping from the parsed YAML file.

    Returns:
        A `DataConfig` with `train_path` / `valid_path` resolved to `Path`.

    Raises:
        ValueError: If a required field is missing.
        FileNotFoundError: If `train_path`, or a non-null `valid_path`,
            does not exist on disk.
    """
    _require_keys(section, _DATA_REQUIRED_KEYS, "data")

    split_strategy = str(section.get("split_strategy", "none"))
    if split_strategy not in _VALID_SPLIT_STRATEGIES:
        raise ValueError(
            f"Config field 'data.split_strategy' must be one of "
            f"{sorted(_VALID_SPLIT_STRATEGIES)}, got {split_strategy!r}"
        )

    all_records_path = _optional_existing_path(
        section.get("all_records_path"), "data.all_records_path"
    )
    valid_path = _optional_existing_path(section["valid_path"], "data.valid_path")
    test_path = _optional_existing_path(section.get("test_path"), "data.test_path")
    split_dir = _optional_path(section.get("split_dir"), "data.split_dir")

    if split_strategy == "none":
        train_path = _require_existing_path(section["train_path"], "data.train_path")
    else:
        train_path = _optional_existing_path(section["train_path"], "data.train_path")
        if all_records_path is None:
            raise ValueError(
                "Config field 'data.all_records_path' is required when "
                f"data.split_strategy={split_strategy!r}"
            )
        if split_dir is None:
            raise ValueError(
                "Config field 'data.split_dir' is required when "
                f"data.split_strategy={split_strategy!r}"
            )

    valid_fraction = float(section.get("valid_fraction", 0.1))
    test_fraction = float(section.get("test_fraction", 0.1))
    if split_strategy != "none" and not (0.0 < valid_fraction + test_fraction < 1.0):
        raise ValueError(
            "data.valid_fraction + data.test_fraction must be greater than 0 "
            f"and less than 1 in automatic split mode, got {valid_fraction + test_fraction}"
        )

    return DataConfig(
        train_path=train_path,
        valid_path=valid_path,
        max_pairs_per_group=int(section["max_pairs_per_group"]),
        seed=int(section["seed"]),
        pair_sample_strategy=str(section.get("pair_sample_strategy", "absolute_cap")),
        pair_fraction=(
            None if section.get("pair_fraction") is None else float(section["pair_fraction"])
        ),
        min_pairs_per_group=int(section.get("min_pairs_per_group", 1)),
        large_group_threshold=int(section.get("large_group_threshold", 10_000)),
        pair_enumeration_limit=int(section.get("pair_enumeration_limit", 100_000)),
        label_block_count=int(section.get("label_block_count", 5)),
        intra_block_pairs_per_large_group=int(
            section.get("intra_block_pairs_per_large_group", 50)
        ),
        discrete_label_unique_threshold=int(section.get("discrete_label_unique_threshold", 32)),
        discrete_label_ratio_threshold=float(section.get("discrete_label_ratio_threshold", 0.05)),
        all_records_path=all_records_path,
        test_path=test_path,
        split_strategy=split_strategy,
        split_dir=split_dir,
        valid_fraction=valid_fraction,
        test_fraction=test_fraction,
        record_filter=build_record_filter_config(section.get("filter")),
    )


def _build_model_config(section: dict[str, Any]) -> ModelConfig:
    """Build the v0.65 nested model config, accepting legacy YAML temporarily."""
    antibody_raw = section.get("antibody_encoder")
    if isinstance(antibody_raw, dict):
        _require_keys(section, ("antibody_encoder", "antigen_encoder", "interaction", "objective"), "model")
        antibody_encoder = _build_encoder_config(antibody_raw, "model.antibody_encoder")
        antigen_raw = section["antigen_encoder"]
        if antigen_raw is not None and not isinstance(antigen_raw, dict):
            raise ValueError("Config field 'model.antigen_encoder' must be a mapping or null")
        antigen_encoder = (
            None
            if antigen_raw is None
            else _build_encoder_config(antigen_raw, "model.antigen_encoder")
        )
        interaction = _build_interaction_config(section["interaction"])
        objective = _build_objective_config(section["objective"])
    else:
        antibody_encoder, antigen_encoder, interaction, objective = _build_legacy_model_config(section)

    _validate_model_combination(antibody_encoder, antigen_encoder, interaction)
    return ModelConfig(
        antibody_encoder=antibody_encoder,
        antigen_encoder=antigen_encoder,
        interaction=interaction,
        objective=objective,
    )


def _build_encoder_config(section: dict[str, Any], field_name: str) -> EncoderConfig:
    required = (
        "name", "revision", "mode", "embedding_layer", "cache_dir",
        "max_length", "long_sequence_strategy",
    )
    _require_keys(section, required, field_name)
    name = str(section["name"]).strip()
    revision = str(section["revision"]).strip()
    tokenizer_revision = str(section.get("tokenizer_revision", revision)).strip()
    mode = str(section["mode"])
    strategy = str(section["long_sequence_strategy"])
    cache_dir = _optional_existing_path(section["cache_dir"], f"{field_name}.cache_dir")
    max_length = None if section["max_length"] is None else int(section["max_length"])
    if not name or not revision or not tokenizer_revision:
        raise ValueError(f"Config field '{field_name}' requires non-empty name/revisions")
    if mode not in _ENCODER_MODES:
        raise ValueError(f"Config field '{field_name}.mode' must be one of {sorted(_ENCODER_MODES)}")
    if strategy not in _LONG_SEQUENCE_STRATEGIES:
        raise ValueError(
            f"Config field '{field_name}.long_sequence_strategy' must be one of "
            f"{sorted(_LONG_SEQUENCE_STRATEGIES)}"
        )
    if max_length is not None and max_length < 1:
        raise ValueError(f"Config field '{field_name}.max_length' must be null or positive")
    if strategy in {"truncate", "chunk"} and max_length is None:
        raise ValueError(f"Config field '{field_name}.max_length' is required for strategy={strategy!r}")
    if mode == "frozen_cached":
        if cache_dir is None:
            raise ValueError(f"Config field '{field_name}.cache_dir' is required for frozen_cached")
        if not cache_dir.is_dir():
            raise ValueError(f"Config field '{field_name}.cache_dir' must be a directory")
        if revision.lower() in {"main", "master", "latest"}:
            raise ValueError(f"Config field '{field_name}.revision' must be immutable for frozen_cached")
        if tokenizer_revision.lower() in {"main", "master", "latest"}:
            raise ValueError(
                f"Config field '{field_name}.tokenizer_revision' must be immutable for frozen_cached"
            )
    lora_rank = _optional_int(section.get("lora_rank"))
    lora_alpha = _optional_float(section.get("lora_alpha"))
    lora_dropout = _optional_float(section.get("lora_dropout"))
    if mode == "lora_online":
        if lora_rank is None or lora_rank < 1 or lora_alpha is None or lora_alpha <= 0:
            raise ValueError(f"Config field '{field_name}' requires positive LoRA rank and alpha")
        if lora_dropout is None or not 0.0 <= lora_dropout < 1.0:
            raise ValueError(f"Config field '{field_name}.lora_dropout' must satisfy 0 <= value < 1")
        if cache_dir is not None:
            raise ValueError(f"Config field '{field_name}.cache_dir' must be null for lora_online")
    return EncoderConfig(
        name=name,
        revision=revision,
        tokenizer_revision=tokenizer_revision,
        mode=mode,
        embedding_layer=int(section["embedding_layer"]),
        cache_dir=cache_dir,
        max_length=max_length,
        long_sequence_strategy=strategy,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )


def _build_interaction_config(raw: Any) -> InteractionConfig:
    if not isinstance(raw, dict):
        raise ValueError("Config field 'model.interaction' must be a mapping")
    required = (
        "kind", "d_model", "num_layers", "num_heads", "ffn_multiplier",
        "dropout", "pooling", "bidirectional",
    )
    _require_keys(raw, required, "model.interaction")
    config = InteractionConfig(
        kind=str(raw["kind"]),
        d_model=int(raw["d_model"]),
        num_layers=int(raw["num_layers"]),
        num_heads=int(raw["num_heads"]),
        ffn_multiplier=float(raw["ffn_multiplier"]),
        dropout=float(raw["dropout"]),
        pooling=str(raw["pooling"]),
        bidirectional=_require_bool(raw["bidirectional"], "model.interaction.bidirectional"),
    )
    if config.kind not in _INTERACTION_KINDS:
        raise ValueError(f"model.interaction.kind must be one of {sorted(_INTERACTION_KINDS)}")
    if config.d_model < 1 or config.num_heads < 1 or config.d_model % config.num_heads != 0:
        raise ValueError("model.interaction requires positive d_model divisible by num_heads")
    if config.ffn_multiplier <= 0 or not 0.0 <= config.dropout < 1.0:
        raise ValueError("model.interaction requires positive ffn_multiplier and 0 <= dropout < 1")
    if config.pooling not in _POOLING_KINDS:
        raise ValueError(f"model.interaction.pooling must be one of {sorted(_POOLING_KINDS)}")
    if config.kind == "deep_cross_attention" and config.num_layers not in {4, 8, 16}:
        raise ValueError("deep_cross_attention requires num_layers in {4, 8, 16}")
    if config.kind != "deep_cross_attention" and config.num_layers != 0:
        raise ValueError(f"{config.kind} requires num_layers == 0")
    return config


def _build_objective_config(raw: Any) -> ObjectiveConfig:
    if not isinstance(raw, dict):
        raise ValueError("Config field 'model.objective' must be a mapping")
    _require_keys(raw, ("name", "temperature", "sigma", "pointwise_loss"), "model.objective")
    config = ObjectiveConfig(
        name=str(raw["name"]),
        temperature=float(raw["temperature"]),
        sigma=float(raw["sigma"]),
        pointwise_loss=str(raw["pointwise_loss"]),
    )
    if config.name not in _OBJECTIVES:
        raise ValueError(f"model.objective.name must be one of {sorted(_OBJECTIVES)}")
    if config.temperature <= 0 or config.sigma <= 0:
        raise ValueError("model.objective temperature and sigma must be positive")
    if config.pointwise_loss not in {"huber", "mse"}:
        raise ValueError("model.objective.pointwise_loss must be 'huber' or 'mse'")
    return config


def _build_legacy_model_config(
    section: dict[str, Any],
) -> tuple[EncoderConfig, EncoderConfig | None, InteractionConfig, ObjectiveConfig]:
    """Translate the pre-v0.65 flat YAML format to online-mode config objects."""
    required = ("antibody_encoder", "antigen_encoder", "d_model", "use_cross_attention")
    _require_keys(section, required, "model")
    antibody = _legacy_encoder(str(section["antibody_encoder"]))
    antigen = None if section["antigen_encoder"] is None else _legacy_encoder(str(section["antigen_encoder"]))
    use_cross_attention = _require_bool(section["use_cross_attention"], "model.use_cross_attention")
    kind = "antibody_only" if antigen is None else ("deep_cross_attention" if use_cross_attention else "concat")
    interaction = InteractionConfig(
        kind=kind,
        d_model=int(section["d_model"]),
        num_layers=1 if kind == "deep_cross_attention" else 0,
        num_heads=1,
        ffn_multiplier=4.0,
        dropout=0.1,
        pooling="masked_mean",
        bidirectional=False,
    )
    objective = ObjectiveConfig(
        name="pairwise_ranknet", temperature=1.0, sigma=1.0, pointwise_loss="huber"
    )
    return antibody, antigen, interaction, objective


def _legacy_encoder(name: str) -> EncoderConfig:
    return EncoderConfig(
        name=name,
        revision="main",
        tokenizer_revision="main",
        mode="frozen_online",
        embedding_layer=-1,
        cache_dir=None,
        max_length=None,
        long_sequence_strategy="error",
    )


def _validate_model_combination(
    antibody: EncoderConfig,
    antigen: EncoderConfig | None,
    interaction: InteractionConfig,
) -> None:
    if interaction.kind == "antibody_only" and antigen is not None:
        raise ValueError("antibody_only requires model.antigen_encoder=null")
    if interaction.kind != "antibody_only" and antigen is None:
        raise ValueError(f"{interaction.kind} requires model.antigen_encoder")
    if antigen is not None:
        cached_modes = {antibody.mode, antigen.mode}
        if "frozen_cached" in cached_modes and cached_modes != {"frozen_cached"}:
            raise ValueError(
                "cached antibody/antigen interaction requires both encoders in frozen_cached mode"
            )


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"Config field '{field_name}' must be boolean")
    return value


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _build_train_config(section: dict[str, Any]) -> TrainConfig:
    """Build `TrainConfig` from the `train` section of a config file.

    Args:
        section: The `train` mapping from the parsed YAML file.

    Returns:
        A `TrainConfig` populated from `section`.

    Raises:
        ValueError: If a required field is missing.
    """
    _require_keys(section, _TRAIN_REQUIRED_KEYS, "train")

    return TrainConfig(
        batch_size=int(section["batch_size"]),
        lr=float(section["lr"]),
        epochs=int(section["epochs"]),
        device=str(section["device"]),
    )
