"""Shared split infrastructure: validation, hashing, assignment, balancing.

Lowest-level module. Imports nothing from protocol, audit, or facade modules.
"""
from __future__ import annotations

import hashlib
import random

import pandas as pd


COLD_START_IDENTITY_COLUMNS = (
    "measurement_family_id",
    "antibody_sequence_key",
    "antibody_cluster_id",
    "antigen_sequence_key",
    "antigen_cluster_id",
    "interaction_key",
    "effective_antigen_input_hash",
)
# Protocol-specific required identity columns. Antibody cold-start (known
# antigen, unseen antibody -- entity_cold_start_protocols.md section 5.3) needs
# the antibody identity, the antigen sequence key (to require train-seen
# antigens) and the interaction key, but NOT antigen_cluster_id or
# effective_antigen_input_hash. Antigen cold-start keeps its existing full set
# unchanged.


def frame_hash(frame: pd.DataFrame) -> str:
    """Deterministic content hash of a DataFrame (column-order independent)."""
    canonical = frame.reindex(sorted(frame.columns), axis=1).astype(str)
    row_hashes = pd.util.hash_pandas_object(canonical, index=False)
    return hashlib.sha256(row_hashes.values.tobytes()).hexdigest()


def _assign_component_splits(
    records: pd.DataFrame,
    *,
    train_units: set[str],
    valid_units: set[str],
    test_units: set[str],
) -> pd.DataFrame:
    assigned = records.copy()
    split_by_component = {
        **{unit: "train" for unit in train_units},
        **{unit: "valid" for unit in valid_units},
        **{unit: "test" for unit in test_units},
    }
    assigned["_assigned_split"] = assigned["_component_id"].map(split_by_component)
    if assigned["_assigned_split"].isna().any():
        raise RuntimeError("some entity components were not assigned to a split")
    return assigned


def _assign_weighted_units_to_folds(
    weights: dict[str, int],
    n_splits: int,
    seed: int,
) -> list[set[str]]:
    if len(weights) < n_splits:
        raise ValueError(
            f"n_splits={n_splits} exceeds the number of components={len(weights)}"
        )
    units = sorted(weights)
    random.Random(seed).shuffle(units)
    tie_order = {unit: index for index, unit in enumerate(units)}
    units.sort(key=lambda unit: (-weights[unit], tie_order[unit]))
    fold_units: list[set[str]] = [set() for _ in range(n_splits)]
    fold_weights = [0] * n_splits
    for unit in units:
        fold_index = min(range(n_splits), key=lambda index: (fold_weights[index], index))
        fold_units[fold_index].add(unit)
        fold_weights[fold_index] += int(weights[unit])
    return fold_units


def _combine_excluded_records(
    *tables: pd.DataFrame,
    columns: pd.Index,
) -> pd.DataFrame:
    output_columns = list(columns) + ["_assigned_split", "protocol_exclusion_reason"]
    present = [table for table in tables if table is not None and not table.empty]
    if not present:
        return pd.DataFrame(columns=output_columns)
    combined = pd.concat(present, ignore_index=True, sort=False)
    for column in output_columns:
        if column not in combined.columns:
            combined[column] = None
    return combined[output_columns].sort_values(
        ["_assigned_split", "dataset_id", "record_id"], kind="stable"
    ).reset_index(drop=True)


def _drop_split_helpers(records: pd.DataFrame) -> pd.DataFrame:
    helper_columns = [
        column
        for column in ("_component_id", "_assigned_split", "_validation_fold")
        if column in records.columns
    ]
    return records.drop(columns=helper_columns).sort_values(
        ["dataset_id", "record_id"], kind="stable"
    ).reset_index(drop=True)


def _validate_fraction_and_eval_size(
    valid_fraction: float,
    test_fraction: float,
    min_eval_records: int,
) -> None:
    if valid_fraction < 0 or test_fraction < 0:
        raise ValueError("valid_fraction and test_fraction must be non-negative")
    if not (0.0 < valid_fraction + test_fraction < 1.0):
        raise ValueError("valid_fraction + test_fraction must be > 0 and < 1")
    if min_eval_records < 2:
        raise ValueError("min_eval_records must be at least 2")


def _derive_group_seed(seed: int, group_id: str) -> int:
    """Stable per-group seed derived from a base seed and `group_id`.

    Plain `hash()` is per-process-randomized (PYTHONHASHSEED), so it would
    make this split non-reproducible across runs/restarts. hashlib is not.
    """
    digest = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _concat_sorted(parts: list[pd.DataFrame], columns: pd.Index) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=columns)
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["dataset_id", "record_id"])
        .reset_index(drop=True)
    )


def _partition_units(
    units: list[str],
    valid_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[set[str], set[str], set[str]]:
    if len(units) < 3:
        raise ValueError(
            f"At least 3 split units are required to create train/valid/test, got {len(units)}"
        )

    shuffled = list(units)
    random.Random(seed).shuffle(shuffled)

    n_units = len(shuffled)
    n_valid = _fraction_count(n_units, valid_fraction)
    n_test = _fraction_count(n_units, test_fraction)
    while n_valid + n_test >= n_units:
        if n_test >= n_valid and n_test > 0:
            n_test -= 1
        elif n_valid > 0:
            n_valid -= 1
        else:
            break

    if n_valid == 0 and valid_fraction > 0:
        n_valid = 1
    if n_test == 0 and test_fraction > 0 and n_valid + n_test < n_units - 1:
        n_test = 1
    while n_valid + n_test >= n_units:
        if n_test >= n_valid and n_test > 0:
            n_test -= 1
        else:
            n_valid -= 1

    test_units = set(shuffled[:n_test])
    valid_units = set(shuffled[n_test:n_test + n_valid])
    train_units = set(shuffled[n_test + n_valid:])
    if not train_units or not valid_units or not test_units:
        raise ValueError(
            "Split fractions produced an empty train, valid, or test split; "
            f"n_units={n_units}, n_valid={n_valid}, n_test={n_test}"
        )
    return train_units, valid_units, test_units


def _fraction_count(n_units: int, fraction: float) -> int:
    if fraction <= 0:
        return 0
    return max(1, int(round(n_units * fraction)))


def _partition_weighted_units(
    weights: dict[str, int],
    valid_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[set[str], set[str], set[str]]:
    """Partition group ids while keeping over-target groups in train."""
    if len(weights) < 3:
        raise ValueError(
            f"At least 3 split units are required to create train/valid/test, got {len(weights)}"
        )
    if any(weight < 1 for weight in weights.values()):
        raise ValueError("split unit weights must be positive")

    total_weight = sum(weights.values())
    holdout_limit = _fraction_count(total_weight, max(valid_fraction, test_fraction))
    pinned_train = {unit for unit, weight in weights.items() if weight > holdout_limit}
    eligible = sorted(set(weights) - pinned_train)
    if len(eligible) < 2:
        return _partition_units(sorted(weights), valid_fraction, test_fraction, seed)

    n_valid = _fraction_count(len(weights), valid_fraction)
    n_test = _fraction_count(len(weights), test_fraction)
    reserve_for_train = 0 if pinned_train else 1
    while n_valid + n_test > len(eligible) - reserve_for_train:
        if n_test >= n_valid and n_test > 1:
            n_test -= 1
        elif n_valid > 1:
            n_valid -= 1
        else:
            break

    if n_valid < 1 or n_test < 1 or n_valid + n_test > len(eligible):
        return _partition_units(sorted(weights), valid_fraction, test_fraction, seed)

    units = list(eligible)
    rng = random.Random(seed)
    rng.shuffle(units)
    holdout_units = units[:n_test + n_valid]
    test_units, valid_units = _split_holdout_by_weight(
        holdout_units, weights, n_test=n_test, n_valid=n_valid
    )
    train_units = (set(units[n_test + n_valid:]) | pinned_train)

    if not train_units or not valid_units or not test_units:
        raise ValueError(
            "Split fractions produced an empty train, valid, or test split; "
            f"n_units={len(weights)}, n_valid={len(valid_units)}, n_test={len(test_units)}"
        )
    return train_units, valid_units, test_units


def _split_holdout_by_weight(
    holdout_units: list[str],
    weights: dict[str, int],
    n_test: int,
    n_valid: int,
) -> tuple[set[str], set[str]]:
    test_units: set[str] = set()
    valid_units: set[str] = set()
    test_weight = 0
    valid_weight = 0

    for unit in sorted(holdout_units, key=lambda item: weights[item], reverse=True):
        if len(test_units) >= n_test:
            valid_units.add(unit)
            valid_weight += weights[unit]
        elif len(valid_units) >= n_valid:
            test_units.add(unit)
            test_weight += weights[unit]
        elif test_weight <= valid_weight:
            test_units.add(unit)
            test_weight += weights[unit]
        else:
            valid_units.add(unit)
            valid_weight += weights[unit]

    return test_units, valid_units


def _rows_for_values(records: pd.DataFrame, column: str, values: set[str]) -> pd.DataFrame:
    mask = records[column].astype(str).isin(values)
    return records.loc[mask].sort_values(["dataset_id", "record_id"]).reset_index(drop=True)


def _trainable_records(records: pd.DataFrame) -> pd.DataFrame:
    keep = records["keep_for_training"].apply(_parse_bool)
    labels = pd.to_numeric(records["rank_label"], errors="coerce")
    finite = labels.apply(lambda value: pd.notna(value) and value not in (float("inf"), float("-inf")))
    return records.loc[keep & finite].copy()


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, float) and value in (0.0, 1.0):
        return bool(int(value))
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "t", "1", "yes", "y"}:
            return True
        if normalized in {"false", "f", "0", "no", "n"}:
            return False
    raise ValueError(f"Invalid boolean value for keep_for_training: {value!r}")
