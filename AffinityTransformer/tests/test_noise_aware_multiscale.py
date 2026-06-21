"""Tests for `noise_aware_multiscale.py` (spec §13).

This strategy is not wired into `build_pairs`/`pairs.py` yet, so these tests
call `_noise_aware_multiscale_pairs` directly, the same way
`noise_floor_analysis.py` drove the (now-superseded) `noise_floor_tree.py`
prototype before it was ever wired in either.
"""

from __future__ import annotations

import collections

import pandas as pd
import pytest

from affinity_transformer.dataset.pair_sampling.noise_aware_multiscale import (
    _build_tau_separated_anchors,
    _noise_aware_multiscale_pairs,
)
from affinity_transformer.dataset.pair_sampling.tau_registry import resolve_tau_for_group

GROUP_ID = "studyX/tableX/agX/neg_log10_kd_M/experimental"


def _records_from_labels(labels: list[float], prefix: str = "r") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_id": [f"{prefix}/{i:06d}" for i in range(len(labels))],
            "rank_label": labels,
        }
    )


def _degree_and_canonical_pairs(rows: list[dict[str, object]]):
    degree: dict[str, int] = collections.defaultdict(int)
    canonical_pairs: set[tuple[str, str]] = set()
    for row in rows:
        a, b = str(row["record_id_i"]), str(row["record_id_j"])
        degree[a] += 1
        degree[b] += 1
        canonical_pairs.add(tuple(sorted((a, b))))
    return degree, canonical_pairs


# 1. Dense long chain must not collapse into one cluster / produce no pairs.
def test_dense_chain_does_not_collapse():
    n = 101
    labels = [round(i * 0.1, 2) for i in range(n)]  # 0.0, 0.1, ..., 10.0
    tau = 0.3

    anchors = _build_tau_separated_anchors(labels, tau)
    assert len(anchors) > 1  # must not collapse to a single anchor

    records = _records_from_labels(labels)
    rows = _noise_aware_multiscale_pairs(GROUP_ID, records, seed=0, tau=tau)
    assert len(rows) > 0  # must not report "no_resolvable_pairs"
    for row in rows:
        assert abs(row["label_i"] - row["label_j"]) >= tau - 1e-9


# 2. Whole group narrower than tau: no resolvable pairs anywhere, and no
#    fallback to literal adjacency (which would just be tau violations).
def test_whole_group_unresolvable_returns_empty():
    labels = [0.00, 0.01, 0.02, 0.03]
    tau = 0.1
    records = _records_from_labels(labels)

    rows = _noise_aware_multiscale_pairs(GROUP_ID, records, seed=0, tau=tau)
    assert rows == []


# 3. Pair count must stay linear in n, not blow up toward O(n^2).
@pytest.mark.parametrize("n", [50, 500, 5000])
def test_pair_count_is_linear(n):
    labels = [i * 0.01 for i in range(n)]  # span n*0.01, dense relative to tau
    tau = 0.3
    records = _records_from_labels(labels)
    extra_edges_per_record = 2

    rows = _noise_aware_multiscale_pairs(
        GROUP_ID, records, seed=0, tau=tau, extra_edges_per_record=extra_edges_per_record
    )
    # backbone (m-1) + coverage (<= n-m) + enrichment (<= extra*n): <= n-1+extra*n
    assert len(rows) <= n - 1 + extra_edges_per_record * n


# 4. Every record that has at least one >=tau-away partner anywhere in the
#    group must appear in at least one emitted pair.
def test_resolvable_records_get_full_coverage():
    n = 101
    labels = [round(i * 0.1, 2) for i in range(n)]
    tau = 0.3
    records = _records_from_labels(labels)

    rows = _noise_aware_multiscale_pairs(GROUP_ID, records, seed=0, tau=tau)
    covered = {str(row["record_id_i"]) for row in rows} | {str(row["record_id_j"]) for row in rows}
    all_ids = set(records["record_id"])
    # With span 10.0 >> tau=0.3, every record has plenty of far-band partners.
    assert covered == all_ids


# 5. Every emitted hard pair must satisfy the noise-floor invariant.
def test_all_hard_pairs_respect_noise_floor():
    n = 300
    labels = [i * 0.05 for i in range(n)]
    tau = 0.4
    records = _records_from_labels(labels)

    rows = _noise_aware_multiscale_pairs(GROUP_ID, records, seed=0, tau=tau)
    assert len(rows) > 0
    for row in rows:
        assert abs(row["label_i"] - row["label_j"]) >= tau - 1e-9


# 6. Reproducibility: same seed+group_id+input -> identical pairs;
#    different seed -> at least some enrichment edges differ.
def test_reproducible_for_fixed_seed_varies_across_seeds():
    n = 300
    labels = [i * 0.05 for i in range(n)]
    tau = 0.4
    records = _records_from_labels(labels)

    rows_a = _noise_aware_multiscale_pairs(GROUP_ID, records, seed=0, tau=tau)
    rows_b = _noise_aware_multiscale_pairs(GROUP_ID, records, seed=0, tau=tau)
    assert rows_a == rows_b

    rows_other_seed = _noise_aware_multiscale_pairs(GROUP_ID, records, seed=1, tau=tau)
    _, pairs_a = _degree_and_canonical_pairs(rows_a)
    _, pairs_other = _degree_and_canonical_pairs(rows_other_seed)
    assert pairs_a != pairs_other


# 7. No duplicate pairs (direction-independent).
def test_no_duplicate_pairs():
    n = 300
    labels = [i * 0.05 for i in range(n)]
    tau = 0.4
    records = _records_from_labels(labels)

    rows = _noise_aware_multiscale_pairs(GROUP_ID, records, seed=0, tau=tau)
    seen_keys = [tuple(sorted((str(row["record_id_i"]), str(row["record_id_j"])))) for row in rows]
    assert len(seen_keys) == len(set(seen_keys))


# 8. Degree must not concentrate onto a handful of nodes (the exact failure
#    the old noise_floor_tree.py representative-attachment scheme had).
def test_degree_does_not_concentrate():
    n = 5000
    labels = [i * 1e-4 for i in range(n)]  # very dense: span 0.5, tau >> per-step gap
    tau = 0.05
    records = _records_from_labels(labels)
    max_degree = 12

    rows = _noise_aware_multiscale_pairs(
        GROUP_ID, records, seed=0, tau=tau, max_degree=max_degree
    )
    degree, _ = _degree_and_canonical_pairs(rows)
    max_observed = max(degree.values())
    # Old prototype: a single representative in a comparably dense group
    # could absorb O(n / m_clusters) edges -- thousands. Here it must stay
    # within a small constant of max_degree, not scale with n.
    assert max_observed <= max_degree + 5, f"degree hub detected: max degree {max_observed}"


# 9. Exact ties never form hard pairs; tied records still get coverage via
#    any distinguishable record elsewhere in the group.
def test_exact_ties_excluded_but_still_covered():
    tied_labels = [5.0] * 40
    spread_labels = [0.0, 1.0, 9.0, 10.0]
    labels = tied_labels + spread_labels
    tau = 0.3
    records = _records_from_labels(labels)

    rows = _noise_aware_multiscale_pairs(GROUP_ID, records, seed=0, tau=tau)
    for row in rows:
        assert row["label_i"] != row["label_j"]

    tied_ids = set(records["record_id"][:40])
    covered = {str(row["record_id_i"]) for row in rows} | {str(row["record_id_j"]) for row in rows}
    assert tied_ids.issubset(covered)


# --- tau_registry.py -------------------------------------------------------

def _records_with_antigen(antigen_key: str, n: int = 5) -> pd.DataFrame:
    df = _records_from_labels([float(i) for i in range(n)])
    df["antigen_key"] = antigen_key
    return df


@pytest.mark.parametrize(
    "antigen_key,expected_tau,expected_label",
    [
        ("SARS_CoV_2", 0.3, "alphaseq_sars_cov_2"),
        ("SARS_CoV_2_E483W", 0.15, "bloom_titeseq_sarbecovirus"),
        ("MERS_CoV", 0.15, "bloom_titeseq_sarbecovirus"),
        ("HIV_YU2", 0.5, "catnap_hiv_neutralization"),
        ("AgSKEMPI_3bdy_V_WT", 0.35, "skempi_ddg"),
        ("HER2", 0.2, "default_unresearched"),  # unmatched -> default fallback
        ("OVA", 0.2, "default_unresearched"),
    ],
)
def test_resolve_tau_for_group_matches_expected_rule(antigen_key, expected_tau, expected_label):
    records = _records_with_antigen(antigen_key)
    tau, label, basis = resolve_tau_for_group(records, default_tau=0.2)
    assert tau == expected_tau
    assert label == expected_label
    assert basis  # always non-empty, citation/justification text


def test_resolve_tau_for_group_rejects_mixed_antigen_keys():
    records = pd.concat(
        [_records_with_antigen("SARS_CoV_2", n=3), _records_with_antigen("HIV_YU2", n=3)],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="antigen_key"):
        resolve_tau_for_group(records, default_tau=0.2)


def test_resolve_tau_for_group_honors_custom_default_tau():
    records = _records_with_antigen("totally_unknown_source")
    tau, label, _basis = resolve_tau_for_group(records, default_tau=0.42)
    assert tau == 0.42
    assert label == "default_unresearched"
