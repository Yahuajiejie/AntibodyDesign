"""Tests for affinity_transformer.metrics (spec §5.6)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from affinity_transformer.metrics import compute_group_spearman, summarize_group_spearman


def _row(record_id, group_id, rank_label, score, label_kind="experimental", dataset_id="studyA/tableA"):
    return dict(
        record_id=record_id,
        group_id=group_id,
        rank_label=rank_label,
        score=score,
        label_kind=label_kind,
        dataset_id=dataset_id,
    )


def test_compute_group_spearman_perfect_positive_and_negative_correlation():
    predictions = pd.DataFrame([
        _row("a1", "groupA", 1.0, 0.1, dataset_id="studyA/tableA"),
        _row("a2", "groupA", 2.0, 0.2, dataset_id="studyA/tableA"),
        _row("a3", "groupA", 3.0, 0.3, dataset_id="studyA/tableA"),
        _row("b1", "groupB", 1.0, 0.3, dataset_id="studyB/tableB"),
        _row("b2", "groupB", 2.0, 0.2, dataset_id="studyB/tableB"),
        _row("b3", "groupB", 3.0, 0.1, dataset_id="studyB/tableB"),
    ])

    result = compute_group_spearman(predictions)

    by_group = result.set_index("group_id")
    assert by_group.loc["groupA", "spearman"] == pytest.approx(1.0)
    assert by_group.loc["groupB", "spearman"] == pytest.approx(-1.0)
    assert by_group.loc["groupA", "n_records"] == 3
    assert by_group.loc["groupA", "n_unique_labels"] == 3
    assert by_group.loc["groupA", "dataset_id"] == "studyA/tableA"
    assert by_group.loc["groupB", "dataset_id"] == "studyB/tableB"


def test_compute_group_spearman_skips_one_label_group():
    """spec §7.2: `compute_group_spearman skips one-label groups`."""
    predictions = pd.DataFrame([
        _row("c1", "groupC", 5.0, 0.1),
        _row("c2", "groupC", 5.0, 0.9),
    ])

    result = compute_group_spearman(predictions)

    row = result.set_index("group_id").loc["groupC"]
    assert row["n_unique_labels"] == 1
    assert row["n_records"] == 2
    assert math.isnan(row["spearman"])


def test_compute_group_spearman_requires_columns():
    predictions = pd.DataFrame([{"record_id": "a1", "group_id": "groupA"}])

    with pytest.raises(ValueError):
        compute_group_spearman(predictions)


def test_compute_group_spearman_empty_input_has_expected_columns():
    predictions = pd.DataFrame(columns=[
        "record_id", "group_id", "rank_label", "score", "label_kind", "dataset_id",
    ])

    result = compute_group_spearman(predictions)

    assert list(result.columns) == [
        "group_id", "dataset_id", "label_kind", "n_records", "n_unique_labels", "spearman",
    ]
    assert len(result) == 0


def test_summarize_group_spearman_macro_vs_weighted_average():
    """A large group and a small group with opposite correlations should
    pull the macro and weighted averages in different directions."""
    rows = []
    for i in range(10):
        rows.append(_row(f"e{i}", "groupE", float(i), float(i)))  # perfect positive, n=10
    rows.append(_row("f0", "groupF", 1.0, 2.0))
    rows.append(_row("f1", "groupF", 2.0, 1.0))  # perfect negative, n=2
    predictions = pd.DataFrame(rows)

    group_metrics = compute_group_spearman(predictions)
    summary = summarize_group_spearman(group_metrics)

    overall = summary["overall"]
    assert overall["n_groups"] == 2
    assert overall["n_valid_groups"] == 2
    assert overall["n_skipped_groups"] == 0
    assert overall["macro_spearman"] == pytest.approx(0.0)
    # weighted: (1.0*10 + (-1.0)*2) / 12
    assert overall["weighted_spearman"] == pytest.approx((10 - 2) / 12)


def test_summarize_group_spearman_counts_skipped_groups():
    predictions = pd.DataFrame([
        _row("a1", "groupA", 1.0, 0.1),
        _row("a2", "groupA", 2.0, 0.2),
        _row("c1", "groupC", 5.0, 0.1),
        _row("c2", "groupC", 5.0, 0.9),
    ])

    group_metrics = compute_group_spearman(predictions)
    summary = summarize_group_spearman(group_metrics)

    overall = summary["overall"]
    assert overall["n_groups"] == 2
    assert overall["n_valid_groups"] == 1
    assert overall["n_skipped_groups"] == 1
    assert overall["macro_spearman"] == pytest.approx(1.0)


def test_summarize_group_spearman_reports_binary_label_kind_separately():
    """spec §5.6 rule 4: binary-label Spearman must be reported separately."""
    predictions = pd.DataFrame([
        _row("a1", "groupA", 1.0, 0.1, label_kind="experimental"),
        _row("a2", "groupA", 2.0, 0.2, label_kind="experimental"),
        _row("d1", "groupD", 1.0, 0.9, label_kind="binary"),
        _row("d2", "groupD", 1.0, 0.8, label_kind="binary"),
        _row("d3", "groupD", 0.0, 0.2, label_kind="binary"),
        _row("d4", "groupD", 0.0, 0.1, label_kind="binary"),
    ])

    group_metrics = compute_group_spearman(predictions)
    summary = summarize_group_spearman(group_metrics)

    assert "experimental" in summary
    assert "binary" in summary
    assert summary["experimental"]["n_groups"] == 1
    assert summary["binary"]["n_groups"] == 1
    # overall combines both label kinds.
    assert summary["overall"]["n_groups"] == 2
