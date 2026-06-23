"""build_splits strategy dispatcher and strategy registries."""
from __future__ import annotations

import pandas as pd

from .debug import _split_by_record
from .group import _split_by_group
from .antigen_cluster import _split_by_antigen_cluster
from .audits import _build_leakage_report, _build_summary
from .results import SplitResult


VALID_STRATEGIES = {"debug_record_split", "group_holdout_split", "antigen_cluster_holdout_split"}

# `within_antigen_split` is intentionally NOT in VALID_STRATEGIES / build_splits.
# It answers a different question (programming_spec_v1.0.md section 3.2:
# "known-antigen, new-antibody") and deliberately allows the same group_id
# to appear in more than one split -- mixing it into build_splits's dispatch
# would make it too easy to point a real training config at it by mistake
# and report the result as unseen-antigen generalization, which it is not.


AUXILIARY_STRATEGIES = {"within_antigen_split"}


def build_splits(
    records: pd.DataFrame,
    strategy: str,
    valid_fraction: float,
    test_fraction: float,
    seed: int,
    antigen_clusters: pd.DataFrame | None = None,
) -> SplitResult:
    """Build train/valid/test splits from one merged processed table.

    Args:
        records: Standard processed records.
        strategy: "debug_record_split", "group_holdout_split", or
            "antigen_cluster_holdout_split".
        valid_fraction: Fraction of units reserved for validation.
        test_fraction: Fraction of units reserved for test.
        seed: Random seed controlling unit shuffling.
        antigen_clusters: Required when `strategy ==
            "antigen_cluster_holdout_split"` -- the output of
            `antigen_clustering.compute_antigen_clusters` (a DataFrame
            mapping every `antigen_key` in `records` to an
            `antigen_cluster_id`). Ignored for other strategies.

    Returns:
        `SplitResult` containing train/valid/test records and two QC reports.

    Raises:
        ValueError: If required columns are missing, the strategy/fractions
            are invalid, too few units exist, `antigen_clusters` is missing
            or incomplete for `antigen_cluster_holdout_split`, or a leakage
            check fails.
    """
    _validate_inputs(records, strategy, valid_fraction, test_fraction)

    if strategy == "debug_record_split":
        train, valid, test = _split_by_record(records, valid_fraction, test_fraction, seed)
    elif strategy == "group_holdout_split":
        train, valid, test = _split_by_group(records, valid_fraction, test_fraction, seed)
    elif strategy == "antigen_cluster_holdout_split":
        if antigen_clusters is None:
            raise ValueError(
                "antigen_clusters is required for strategy='antigen_cluster_holdout_split' "
                "-- compute it with antigen_clustering.compute_antigen_clusters first"
            )
        train, valid, test = _split_by_antigen_cluster(
            records, valid_fraction, test_fraction, seed, antigen_clusters
        )
    else:  # guarded by _validate_inputs
        raise ValueError(f"Unsupported split strategy: {strategy!r}")

    summary = _build_summary(strategy, train=train, valid=valid, test=test)
    leakage_report = _build_leakage_report(strategy, train=train, valid=valid, test=test)
    failed = leakage_report[leakage_report["status"] != "PASS"]
    if not failed.empty:
        raise ValueError(f"Split leakage check failed: {failed.to_dict(orient='records')}")

    # _split_by_antigen_cluster keeps a `_antigen_cluster_id` helper column
    # so the leakage check above could verify it directly; strip it from
    # the user-facing output now that the check has passed.
    if "_antigen_cluster_id" in train.columns:
        train = train.drop(columns=["_antigen_cluster_id"])
        valid = valid.drop(columns=["_antigen_cluster_id"])
        test = test.drop(columns=["_antigen_cluster_id"])

    return SplitResult(train=train, valid=valid, test=test, summary=summary,
                       leakage_report=leakage_report)


def _validate_inputs(
    records: pd.DataFrame,
    strategy: str,
    valid_fraction: float,
    test_fraction: float,
) -> None:
    required = ("record_id", "group_id", "keep_for_training", "rank_label")
    if strategy == "antigen_cluster_holdout_split":
        required = required + ("antigen_key",)
    missing = [column for column in required if column not in records.columns]
    if missing:
        raise ValueError(f"records is missing required column(s): {missing}")
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"strategy must be one of {sorted(VALID_STRATEGIES)}, got {strategy!r}")
    if valid_fraction < 0 or test_fraction < 0:
        raise ValueError("valid_fraction and test_fraction must be non-negative")
    if not (0.0 < valid_fraction + test_fraction < 1.0):
        raise ValueError("valid_fraction + test_fraction must be > 0 and < 1")
    if records["record_id"].isna().any():
        raise ValueError("records contains null record_id values")
    if records["record_id"].astype(str).duplicated().any():
        duplicated = records.loc[records["record_id"].astype(str).duplicated(), "record_id"].tolist()
        raise ValueError(f"records contains duplicate record_id values: {duplicated[:10]}")
