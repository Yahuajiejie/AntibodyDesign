"""Small generic helpers shared across `affinity_transformer` modules.

(spec docs/programming_spec.md §5.9)

Only genuinely dataset-agnostic, model-agnostic helpers belong here: seeding,
hashing, amino-acid alphabet checks, directory creation, and a thin logging
setup. Do not add raw-table column names, label-transform rules, or model
logic to this module (spec §5.9 "禁止" list) -- those belong to the data-prep
scripts, `dataset/`, and `model/`/`trainer.py` respectively.
"""

from __future__ import annotations

import hashlib
import logging
import random
from pathlib import Path

import numpy as np
import torch

# Standard 20 proteinogenic amino acids, upper-case one-letter codes. Sequences
# using extended/ambiguous codes (B, J, O, U, X, Z, lower-case, gaps, etc.) are
# not considered valid here; data-prep scripts that need to allow those should
# do so explicitly rather than relying on this default alphabet.
_VALID_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def set_seed(seed: int) -> None:
    """Seed every source of randomness this project uses.

    Args:
        seed: Random seed shared by Python's `random`, NumPy, and PyTorch
            (CPU and, if available, all CUDA devices).

    Returns:
        None. Has the side effect of reseeding global RNG state in
        `random`, `numpy`, and `torch`.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def hash_text(text: str) -> str:
    """Return a stable hex digest of `text`.

    Args:
        text: Arbitrary string to hash (e.g. a sequence, a config string).

    Returns:
        The SHA-256 hex digest of `text`, encoded as UTF-8. Deterministic
        across processes and platforms (unlike Python's built-in `hash`).
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_amino_acid_sequence(sequence: str) -> bool:
    """Check whether `sequence` is a non-empty string of standard amino acids.

    Args:
        sequence: Candidate `heavy_chain` / `light_chain` /
            `single_chain_sequence` / `antigen_sequence` value.

    Returns:
        `True` if `sequence` is non-empty and every character is one of the
        20 standard upper-case amino-acid one-letter codes (`ACDEFGHIKLMNPQ
        RSTVWY`, spec §4 rule 6). `False` for an empty string, for any
        character outside that alphabet (including lower-case letters,
        whitespace, `"|"`, `"X"`, gap characters, etc.), and -- since the
        signature takes `str` -- for `None` (which is not a `str`).

    Raises:
        Nothing. This is a pure predicate; callers decide whether an invalid
        sequence is a hard error, a dropped record, or something else.
    """
    if not sequence:
        return False
    return all(residue in _VALID_AMINO_ACIDS for residue in sequence)


def ensure_dir(path: Path) -> Path:
    """Create `path` (and any missing parents) if it does not already exist.

    Args:
        path: Directory to create.

    Returns:
        `path`, converted to `Path`, after ensuring it exists as a
        directory. Safe to call when `path` already exists.

    Raises:
        FileExistsError: If `path` already exists as a non-directory (e.g. a
            regular file), propagated from `Path.mkdir`.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured with a single stream handler.

    Args:
        name: Logger name, typically `__name__` of the calling module.

    Returns:
        A `logging.Logger` at `INFO` level with one `StreamHandler` (added
        only the first time this is called for a given `name`, so repeated
        calls do not create duplicate handlers or duplicated log lines).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
