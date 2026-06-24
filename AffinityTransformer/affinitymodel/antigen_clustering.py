"""Antigen sequence clustering for `splits.antigen_cluster_holdout_split`.

`group_holdout_split` keeps each exact `group_id` (antigen_key + metric +
source) on one side of train/valid/test, but two DIFFERENT antigen_keys can
be near-identical sequences (e.g. `SARS_CoV_2` and `SARS_CoV_2_R356V`, a
single point mutant of it) and land on opposite sides -- so "unseen antigen"
isn't actually guaranteed (programming_spec_v1.0.md section 3.3). This
module clusters antigens by sequence similarity so that splitting can be
done by `antigen_cluster_id` instead, keeping near-duplicate antigens
together.

Design notes (read before changing the threshold or the algorithm):

- Linkage method is deliberately "average" or "complete", never "single".
  This codebase has direct prior experience with single-linkage clustering
  chaining an entire dense set of near-duplicate items into one giant
  cluster instead of many tight ones -- see
  `dataset/pair_sampling/noise_floor_tree.py`'s module docstring for the
  pair-sampling version of the exact same failure mode (a SARS_CoV_2 mega
  group collapsed to a single cluster under single linkage). average/
  complete linkage both require most/all pairwise distances within a
  cluster to be small, not just a chain of pairwise-close neighbors, so
  they don't have this problem. The published AbRank benchmark
  (arXiv:2506.17857) uses single linkage for the equivalent antibody/
  antigen clustering step in their own data curation -- we are
  deliberately deviating from that choice here, for the reason above.

- Real antigen sequences in this project's data are long (median ~870
  residues in the AbRank subset alone) and there are thousands of distinct
  `antigen_key` values, so a naive full O(k^2) pairwise edit-distance
  computation is not cheap (k=5,613 in just the AbRank subset already
  implies ~1.6e7 pairs). This module avoids that by bucketing sequences by
  EXACT length first -- comparing only same-length sequences with fast
  vectorized Hamming distance (mismatched-residue fraction), then running
  hierarchical clustering only within each length bucket.

  KNOWN LIMITATION: antigens differing by even one inserted/deleted residue
  are never compared (different bucket) and will never cluster together,
  even if otherwise highly similar. This catches the dominant real case in
  this project's data -- deep-mutational-scanning point mutants, which by
  construction never change sequence length -- but misses indel-based near
  duplicates. Extending this to cross-length comparison (e.g. edit distance
  for sequences within a small length window) is future work, not
  implemented here.

- There is no single correct `similarity_threshold`. AbRank's own paper
  settled on 0.75 specifically because of HIV Env's extreme diversity
  (60-70% pairwise identity across known strains) -- 0.9 was found to
  over-fragment that panel. A SARS-CoV-2-dominated panel may not need to go
  that low. Compute clusters at a few thresholds (e.g. 0.5, 0.75, 0.9) and
  inspect the resulting cluster-size distribution before committing to one
  for a real split -- don't default to a number without checking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

ANTIGEN_CLUSTER_COLUMNS = ("antigen_key", "antigen_cluster_id", "cluster_size")

_VALID_LINKAGE_METHODS = {"average", "complete"}


def compute_antigen_clusters(
    records: pd.DataFrame,
    similarity_threshold: float = 0.75,
    linkage_method: str = "average",
) -> pd.DataFrame:
    """Cluster distinct `antigen_key` values by antigen-sequence similarity.

    Args:
        records: Standard processed records; must include `antigen_key` and
            `antigen_sequence`. Each `antigen_key` must map to exactly one
            distinct sequence (same assumption
            `tau_registry.resolve_tau_for_group` makes for groups) --
            raises if violated.
        similarity_threshold: Minimum fraction of identical residues
            (same-length sequences only, see module docstring) for two
            antigens to be placed in the same cluster. No universal
            default -- see module docstring.
        linkage_method: "average" or "complete". NOT "single" -- see module
            docstring for why.

    Returns:
        DataFrame with one row per distinct `antigen_key`, columns
        `ANTIGEN_CLUSTER_COLUMNS`. `antigen_cluster_id` is deterministic
        given the same input, threshold, and linkage method.

    Raises:
        ValueError: If required columns are missing, `antigen_key` maps to
            more than one distinct sequence, `antigen_sequence` has nulls,
            `similarity_threshold` is out of `(0.0, 1.0]`, or
            `linkage_method` is invalid.
    """
    required = ("antigen_key", "antigen_sequence")
    missing = [column for column in required if column not in records.columns]
    if missing:
        raise ValueError(f"records is missing required column(s): {missing}")
    if linkage_method not in _VALID_LINKAGE_METHODS:
        raise ValueError(
            f"linkage_method must be one of {sorted(_VALID_LINKAGE_METHODS)}, "
            f"got {linkage_method!r}"
        )
    if not (0.0 < similarity_threshold <= 1.0):
        raise ValueError("similarity_threshold must be in (0.0, 1.0]")

    unique = records[["antigen_key", "antigen_sequence"]].drop_duplicates()
    if unique["antigen_sequence"].isna().any():
        raise ValueError("antigen_sequence contains null values")
    per_key_sequence_counts = unique.groupby("antigen_key")["antigen_sequence"].nunique()
    bad_keys = per_key_sequence_counts[per_key_sequence_counts > 1]
    if not bad_keys.empty:
        raise ValueError(
            f"antigen_key maps to more than one distinct antigen_sequence: "
            f"{bad_keys.index.tolist()[:10]}"
        )

    unique = unique.sort_values("antigen_key").reset_index(drop=True)
    distance_threshold = 1.0 - similarity_threshold

    rows: list[dict[str, object]] = []
    for length, bucket in unique.groupby(unique["antigen_sequence"].str.len(), sort=True):
        bucket = bucket.reset_index(drop=True)
        sequences = bucket["antigen_sequence"].tolist()
        if len(sequences) == 1:
            labels = np.array([1])
        else:
            labels = _cluster_equal_length_bucket(sequences, distance_threshold, linkage_method)
        sizes = pd.Series(labels).value_counts()
        for antigen_key, label in zip(bucket["antigen_key"], labels):
            rows.append({
                "antigen_key": antigen_key,
                "antigen_cluster_id": f"len{length}_c{label}",
                "cluster_size": int(sizes[label]),
            })

    return (
        pd.DataFrame(rows, columns=ANTIGEN_CLUSTER_COLUMNS)
        .sort_values("antigen_key")
        .reset_index(drop=True)
    )


def _cluster_equal_length_bucket(
    sequences: list[str],
    distance_threshold: float,
    linkage_method: str,
) -> np.ndarray:
    """Cluster same-length sequences by normalized Hamming distance.

    `pdist(..., metric="hamming")` already returns the mismatched-fraction
    (i.e. edit distance normalized by length, since these are equal-length
    substitution-only comparisons -- no indels possible within one bucket).
    """
    encoded = np.array([np.frombuffer(seq.encode("ascii"), dtype=np.uint8) for seq in sequences])
    condensed = pdist(encoded, metric="hamming")
    merged = linkage(condensed, method=linkage_method)
    return fcluster(merged, t=distance_threshold, criterion="distance")
