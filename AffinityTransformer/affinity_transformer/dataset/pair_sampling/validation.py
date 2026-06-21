"""Parameter validation for pair sampling."""

from __future__ import annotations


_VALID_PAIR_SAMPLE_STRATEGIES = {
    "absolute_cap",
    "capped_proportional",
    "balanced_tree",
    "randomized_bst",
    "noise_aware_multiscale",
}

_VALID_UNRESOLVED_POLICIES = {"skip"}


def _validate_pair_sampling(
    max_pairs_per_group: int,
    pair_sample_strategy: str,
    pair_fraction: float | None,
    min_pairs_per_group: int,
    large_group_threshold: int,
    pair_enumeration_limit: int,
    label_block_count: int,
    intra_block_pairs_per_large_group: int,
    discrete_label_unique_threshold: int,
    discrete_label_ratio_threshold: float,
    tree_extra_random_pairs_per_group: int = 0,
    noise_aware_extra_edges_per_record: int = 2,
    noise_aware_max_degree: int = 12,
    noise_aware_candidate_probe_count: int = 8,
    noise_aware_default_tau: float = 0.2,
    noise_aware_unresolved_policy: str = "skip",
) -> None:
    if max_pairs_per_group < 1:
        raise ValueError(f"max_pairs_per_group must be >= 1, got {max_pairs_per_group}")
    if min_pairs_per_group < 1:
        raise ValueError(f"min_pairs_per_group must be >= 1, got {min_pairs_per_group}")
    if large_group_threshold < 2:
        raise ValueError(f"large_group_threshold must be >= 2, got {large_group_threshold}")
    if pair_enumeration_limit < 1:
        raise ValueError(f"pair_enumeration_limit must be >= 1, got {pair_enumeration_limit}")
    if label_block_count < 2:
        raise ValueError(f"label_block_count must be >= 2, got {label_block_count}")
    if intra_block_pairs_per_large_group < 0:
        raise ValueError(
            "intra_block_pairs_per_large_group must be >= 0, "
            f"got {intra_block_pairs_per_large_group}"
        )
    if discrete_label_unique_threshold < 2:
        raise ValueError(
            "discrete_label_unique_threshold must be >= 2, "
            f"got {discrete_label_unique_threshold}"
        )
    if not (0.0 < discrete_label_ratio_threshold <= 1.0):
        raise ValueError(
            "discrete_label_ratio_threshold must be in (0, 1], "
            f"got {discrete_label_ratio_threshold}"
        )
    if pair_sample_strategy not in _VALID_PAIR_SAMPLE_STRATEGIES:
        raise ValueError(
            f"pair_sample_strategy must be one of {sorted(_VALID_PAIR_SAMPLE_STRATEGIES)}, "
            f"got {pair_sample_strategy!r}"
        )
    if pair_sample_strategy == "capped_proportional":
        if pair_fraction is None or not (0.0 < pair_fraction <= 1.0):
            raise ValueError(
                "pair_fraction must be in (0, 1] when pair_sample_strategy='capped_proportional'"
            )
    if tree_extra_random_pairs_per_group < 0:
        raise ValueError(
            "tree_extra_random_pairs_per_group must be >= 0, "
            f"got {tree_extra_random_pairs_per_group}"
        )
    if noise_aware_extra_edges_per_record < 0:
        raise ValueError(
            "noise_aware_extra_edges_per_record must be >= 0, "
            f"got {noise_aware_extra_edges_per_record}"
        )
    if noise_aware_max_degree < 1:
        raise ValueError(f"noise_aware_max_degree must be >= 1, got {noise_aware_max_degree}")
    if noise_aware_candidate_probe_count < 1:
        raise ValueError(
            "noise_aware_candidate_probe_count must be >= 1, "
            f"got {noise_aware_candidate_probe_count}"
        )
    if noise_aware_default_tau < 0:
        raise ValueError(f"noise_aware_default_tau must be >= 0, got {noise_aware_default_tau}")
    if noise_aware_unresolved_policy not in _VALID_UNRESOLVED_POLICIES:
        raise ValueError(
            "noise_aware_unresolved_policy must be one of "
            f"{sorted(_VALID_UNRESOLVED_POLICIES)}, got {noise_aware_unresolved_policy!r}"
        )
