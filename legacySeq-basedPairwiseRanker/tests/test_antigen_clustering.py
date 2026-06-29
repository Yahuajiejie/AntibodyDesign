"""Tests for `antigen_clustering.compute_antigen_clusters`.

Background: `group_holdout_split` keeps each exact `group_id` on one side
of a split, but two different `antigen_key` values can be near-identical
sequences (e.g. a SARS-CoV-2 point-mutant naming scheme) and end up on
opposite sides -- this module clusters antigens by sequence similarity so
`antigen_cluster_holdout_split` can split by cluster instead.
"""

from __future__ import annotations

import pandas as pd
import pytest

from affinity_transformer.antigen_clustering import compute_antigen_clusters


def _mutate(seq: str, pos: int, new_char: str) -> str:
    return seq[:pos] + new_char + seq[pos + 1:]


def _family(prefix: str, base: str, n: int) -> list[dict]:
    """`n` point-mutant variants of `base`, all the same length."""
    return [
        {"antigen_key": f"{prefix}_{i}", "antigen_sequence": _mutate(base, i % len(base), "X")}
        for i in range(n)
    ]


_BASE_A = "MKTAYIAKQRQISFVKSHFSRQLEVRLGLIESQ" * 6
_BASE_B = "AAGGCCTTAAGGCCTTAAGGCCTTAAGGCCTTA" * 6
_BASE_C = "QWQWQWQWQWQWQWQWQWQWQWQWQWQWQWQWQ" * 6


def test_well_separated_families_form_separate_clusters():
    rows = _family("A", _BASE_A, 20) + _family("B", _BASE_B, 15) + _family("C", _BASE_C, 10)
    records = pd.DataFrame(rows)

    result = compute_antigen_clusters(records, similarity_threshold=0.9, linkage_method="average")

    assert result["antigen_cluster_id"].nunique() == 3
    by_cluster = result.groupby("antigen_cluster_id")["antigen_key"].apply(set)
    family_a_keys = {f"A_{i}" for i in range(20)}
    family_b_keys = {f"B_{i}" for i in range(15)}
    family_c_keys = {f"C_{i}" for i in range(10)}
    clusters_as_sets = set(frozenset(s) for s in by_cluster)
    assert frozenset(family_a_keys) in clusters_as_sets
    assert frozenset(family_b_keys) in clusters_as_sets
    assert frozenset(family_c_keys) in clusters_as_sets


def test_point_mutants_of_the_same_base_sequence_cluster_together():
    # Single point mutations out of a 198-residue sequence change <1% of
    # residues -- well within even a strict 95% identity threshold.
    rows = _family("A", _BASE_A, 30)
    records = pd.DataFrame(rows)

    result = compute_antigen_clusters(records, similarity_threshold=0.95, linkage_method="average")

    assert result["antigen_cluster_id"].nunique() == 1
    assert (result["cluster_size"] == 30).all()


def test_different_length_sequences_never_cluster_together():
    # Documented v1 limitation: only same-length sequences are compared.
    records = pd.DataFrame([
        {"antigen_key": "short", "antigen_sequence": "M" * 50},
        {"antigen_key": "long", "antigen_sequence": "M" * 51},
    ])
    result = compute_antigen_clusters(records, similarity_threshold=0.5, linkage_method="average")
    assert result["antigen_cluster_id"].nunique() == 2


def test_inconsistent_antigen_key_raises():
    records = pd.DataFrame([
        {"antigen_key": "MERS_CoV", "antigen_sequence": "M" * 100},
        {"antigen_key": "MERS_CoV", "antigen_sequence": "M" * 99 + "K"},  # different sequence, same key
        {"antigen_key": "Other", "antigen_sequence": "Q" * 100},
        {"antigen_key": "Other2", "antigen_sequence": "R" * 100},
    ])
    with pytest.raises(ValueError, match="more than one distinct antigen_sequence"):
        compute_antigen_clusters(records, similarity_threshold=0.9)


def test_rejects_single_linkage():
    records = pd.DataFrame([
        {"antigen_key": "A", "antigen_sequence": "M" * 100},
        {"antigen_key": "B", "antigen_sequence": "Q" * 100},
        {"antigen_key": "C", "antigen_sequence": "R" * 100},
    ])
    with pytest.raises(ValueError, match="linkage_method must be one of"):
        compute_antigen_clusters(records, similarity_threshold=0.9, linkage_method="single")


def test_missing_required_columns_raises():
    records = pd.DataFrame([{"antigen_key": "A"}])
    with pytest.raises(ValueError, match="missing required column"):
        compute_antigen_clusters(records)


def test_null_antigen_sequence_raises():
    records = pd.DataFrame([
        {"antigen_key": "A", "antigen_sequence": None},
        {"antigen_key": "B", "antigen_sequence": "Q" * 100},
        {"antigen_key": "C", "antigen_sequence": "R" * 100},
    ])
    with pytest.raises(ValueError, match="antigen_sequence contains null"):
        compute_antigen_clusters(records)


def test_invalid_similarity_threshold_raises():
    records = pd.DataFrame([{"antigen_key": "A", "antigen_sequence": "M" * 10}])
    with pytest.raises(ValueError, match="similarity_threshold must be"):
        compute_antigen_clusters(records, similarity_threshold=0.0)
    with pytest.raises(ValueError, match="similarity_threshold must be"):
        compute_antigen_clusters(records, similarity_threshold=1.5)


def test_deterministic_across_repeated_calls():
    rows = _family("A", _BASE_A, 25) + _family("B", _BASE_B, 12)
    records = pd.DataFrame(rows)
    a = compute_antigen_clusters(records, similarity_threshold=0.9, linkage_method="average")
    b = compute_antigen_clusters(records, similarity_threshold=0.9, linkage_method="average")
    pd.testing.assert_frame_equal(
        a.sort_values("antigen_key").reset_index(drop=True),
        b.sort_values("antigen_key").reset_index(drop=True),
    )


def test_lower_threshold_never_produces_more_clusters_than_higher_threshold():
    rows = _family("A", _BASE_A, 20) + _family("B", _BASE_B, 15) + _family("C", _BASE_C, 10)
    records = pd.DataFrame(rows)
    loose = compute_antigen_clusters(records, similarity_threshold=0.5, linkage_method="average")
    strict = compute_antigen_clusters(records, similarity_threshold=0.99, linkage_method="average")
    assert loose["antigen_cluster_id"].nunique() <= strict["antigen_cluster_id"].nunique()
