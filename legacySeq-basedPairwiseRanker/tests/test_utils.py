"""Tests for affinity_transformer.utils (spec §5.9)."""

from __future__ import annotations

import logging
import random

import numpy as np
import torch

from affinity_transformer.utils import (
    ensure_dir,
    get_logger,
    hash_text,
    set_seed,
    validate_amino_acid_sequence,
)


def test_set_seed_makes_random_reproducible():
    set_seed(123)
    a = (random.random(), np.random.rand(), torch.rand(3).tolist())

    set_seed(123)
    b = (random.random(), np.random.rand(), torch.rand(3).tolist())

    assert a == b


def test_set_seed_different_seeds_differ():
    set_seed(1)
    a = torch.rand(3).tolist()

    set_seed(2)
    b = torch.rand(3).tolist()

    assert a != b


def test_hash_text_is_deterministic_and_distinguishes_inputs():
    assert hash_text("QVQLVQSGAEVKKPGASVKVSCKAS") == hash_text("QVQLVQSGAEVKKPGASVKVSCKAS")
    assert hash_text("AAA") != hash_text("AAB")
    # SHA-256 hex digest length.
    assert len(hash_text("AAA")) == 64


def test_validate_amino_acid_sequence_accepts_standard_residues():
    assert validate_amino_acid_sequence("QVQLVQSGAEVKKPGASVKVSCKAS") is True


def test_validate_amino_acid_sequence_rejects_empty_string():
    assert validate_amino_acid_sequence("") is False


def test_validate_amino_acid_sequence_rejects_invalid_characters():
    assert validate_amino_acid_sequence("QVQLVQSGAEVKKPGASVKVSCKASX") is False
    assert validate_amino_acid_sequence("qvqlvq") is False
    assert validate_amino_acid_sequence("AB|CD") is False


def test_ensure_dir_creates_nested_directory(tmp_path):
    target = tmp_path / "a" / "b" / "c"

    result = ensure_dir(target)

    assert result == target
    assert target.is_dir()


def test_ensure_dir_is_idempotent(tmp_path):
    target = tmp_path / "a"
    ensure_dir(target)

    # Should not raise when called again on an existing directory.
    ensure_dir(target)

    assert target.is_dir()


def test_get_logger_returns_logger_with_single_handler():
    logger1 = get_logger("affinity_transformer.test_utils_logger")
    logger2 = get_logger("affinity_transformer.test_utils_logger")

    assert logger1 is logger2
    assert isinstance(logger1, logging.Logger)
    assert len(logger1.handlers) == 1
