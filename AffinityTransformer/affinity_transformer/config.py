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

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


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
        seed: Random seed used for pair sampling and any other randomness
            in the data pipeline.
    """

    train_path: Path | None
    valid_path: Path | None
    max_pairs_per_group: int
    seed: int
    all_records_path: Path | None = None
    test_path: Path | None = None
    split_strategy: str = "none"
    split_dir: Path | None = None
    valid_fraction: float = 0.1
    test_fraction: float = 0.1


@dataclass
class ModelConfig:
    """Model architecture switches.

    Attributes:
        antibody_encoder: Name/identifier of the antibody sequence encoder.
        antigen_encoder: Name/identifier of the antigen sequence encoder,
            or None if the model runs in antibody-only mode.
        d_model: Shared hidden dimension used across encoders and the
            scoring head.
        use_cross_attention: Whether the model applies antibody-antigen
            cross-attention (ignored when `antigen_encoder` is None).
    """

    antibody_encoder: str
    antigen_encoder: str | None
    d_model: int
    use_cross_attention: bool


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
_MODEL_REQUIRED_KEYS = ("antibody_encoder", "antigen_encoder", "d_model", "use_cross_attention")
_TRAIN_REQUIRED_KEYS = ("batch_size", "lr", "epochs", "device")


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
        all_records_path=all_records_path,
        test_path=test_path,
        split_strategy=split_strategy,
        split_dir=split_dir,
        valid_fraction=valid_fraction,
        test_fraction=test_fraction,
    )


def _build_model_config(section: dict[str, Any]) -> ModelConfig:
    """Build `ModelConfig` from the `model` section of a config file.

    Args:
        section: The `model` mapping from the parsed YAML file.

    Returns:
        A `ModelConfig` populated from `section`.

    Raises:
        ValueError: If a required field is missing.
    """
    _require_keys(section, _MODEL_REQUIRED_KEYS, "model")

    antigen_encoder_value = section["antigen_encoder"]
    antigen_encoder = None if antigen_encoder_value is None else str(antigen_encoder_value)

    return ModelConfig(
        antibody_encoder=str(section["antibody_encoder"]),
        antigen_encoder=antigen_encoder,
        d_model=int(section["d_model"]),
        use_cross_attention=bool(section["use_cross_attention"]),
    )


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
