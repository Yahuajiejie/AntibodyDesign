"""Balanced-tree pair construction: O(n) comparisons, O(log n) graph diameter.

`absolute_cap`/`capped_proportional` (see `common.py`) target a *fraction of
candidate pairs*, which grows quadratically with group size -- so for any
group above a few hundred records, the fraction saturates the flat
`max_pairs_per_group` ceiling regardless of true size (see
`compute_group_pair_weights` in `training/loaders.py`, which exists to patch
that mismatch after the fact via loss reweighting).

This module sidesteps the mismatch at the source: build exactly
`n_records - 1` pairs per group by recursively splitting the group's
rank-sorted records at the midpoint (always comparing a subrange's median to
the medians of its two halves). A pair count linear in `n_records` means a
group's natural, *unweighted* contribution to pooled training loss already
tracks its true size -- no separate reweighting required.

A pure nearest-neighbor chain (compare every record only to its immediate
rank-neighbor) would also be linear in `n_records`, but its comparison graph
is a path: the distance between two far-apart records is O(n) hops, so any
ranking signal between them must propagate through every record in between,
one weak transitive step at a time. The balanced-tree structure keeps the
same O(n) edge budget but reduces that worst-case distance to O(log n): any
two records are connected via at most ~2*log2(n) hops through their lowest
common ancestor, instead of needing every record in between.

About half of the tree's edges still connect literal rank-neighbors (every
leaf's only edge is to its parent, and at the finest recursion level that
parent is its nearest neighbor) -- that's intentional, not a gap: those are
exactly the hard, locally-discriminating comparisons a chain would also
provide. The other half, concentrated near the root, span exponentially
larger rank gaps (the root's two edges each span roughly a quarter of the
group), and it's those edges that collapse the graph's diameter from O(n) to
O(log n).

The tree itself is fully deterministic (always the exact median), which
trades away one form of robustness: a single mislabeled record at a
structurally privileged split point has no redundant path to be
outvoted by, and the same exact comparisons recur every epoch with no
resampling diversity. `extra_random_pairs_per_group` (default 0, opt-in)
draws that many additional uniformly-random pairs on top of the
deterministic backbone -- cheap (still O(n) for any reasonable budget) and
restores some of the randomness/redundancy a pure tree gives up, without
sacrificing the guaranteed O(log n) diameter the backbone provides.

Splitting the group evenly between low- and high-affinity halves means
exactly half the tree's leaves fall in each half -- no systematic bias
toward either end -- but it does not, on its own, make every record
well-covered, and it does not make every *scale* of comparison equally
robust. Two gaps, addressed by `_add_redundancy_edges`:

1. Roughly half of all `n` records are leaves with only one edge (to their
   parent) -- under-covered regardless of which half they're in.
2. The few edges near the root -- the ones that actually collapse the
   diameter from O(n) to O(log n) -- have *zero* redundancy: each one is
   the sole bridge between two large branches. A single wrong comparison
   there has far more reach than a wrong comparison near a leaf, yet (an
   earlier version of this module's grandparent-only patch shows) naive
   redundancy additions tend to concentrate near the leaves, where they're
   needed least, and leave the root under-protected.

The fix gives every node (not just leaves) one ancestor-jump edge to the
ancestor at roughly half its own depth -- skipping straight to depth/2
rather than always "grandparent" (depth-2) keeps the jump distance
proportional to how deep a node already is, instead of concentrated at
whichever depth happens to hold the most nodes (the leaves). It also gives
every node one random edge to another node *at the same depth*, which
ancestor-jumps alone cannot provide: two nodes in sibling branches can
otherwise only reach each other by routing through the root, an explicit
single point of failure for the bridge between the two halves. With only
two depth-1 nodes (the root's own children), that random same-depth draw
deterministically links them directly -- bypassing the root entirely.
Both additions are one edge per node, so total edges stay at roughly `3*n`
(still O(n)).
"""

from __future__ import annotations

import random

import pandas as pd

from .common import _emit_pair


def _balanced_tree_pairs(
    group_id: object,
    group: pd.DataFrame,
    seed: int,
    extra_random_pairs_per_group: int = 0,
) -> list[dict[str, object]]:
    """Build the balanced-tree comparison set for one rankable group.

    Args:
        group_id: This group's `group_id` (carried onto every emitted row).
        group: One group's trainable records; must have `record_id` and
            `rank_label` columns.
        seed: Random seed for `extra_random_pairs_per_group`'s draws.
            Unused (no randomness) when `extra_random_pairs_per_group == 0`.
        extra_random_pairs_per_group: Additional uniformly-random pairs to
            draw on top of the deterministic tree backbone, deduplicated
            against it and against each other. `0` (default) yields the
            pure deterministic tree.

    Returns:
        A list of `_pair_row(...)` dicts, ready to append into the same
        rows list `build_pairs` accumulates from the other strategies.
        Ties (`label_i == label_j`) are skipped throughout, matching
        `_candidate_pairs`'s convention -- a tied comparison carries no
        ranking signal.
    """
    items = sorted(
        zip(group["record_id"].astype(str), group["rank_label"].astype(float)),
        key=lambda item: (item[1], item[0]),
    )
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    parent_of: dict[int, int] = {}
    _build_tree_edges(items, 0, len(items), group_id, rows, seen, parent_of)
    _add_redundancy_edges(items, parent_of, group_id, rows, seen, seed=f"{seed}:{group_id}:redundancy")

    if extra_random_pairs_per_group > 0 and len(items) >= 2:
        _add_random_pairs(
            items,
            target=extra_random_pairs_per_group,
            seed=f"{seed}:{group_id}:tree_extra",
            group_id=group_id,
            rows=rows,
            seen=seen,
        )
    return rows


def _build_tree_edges(
    items: list[tuple[str, float]],
    lo: int,
    hi: int,
    group_id: object,
    rows: list[dict[str, object]],
    seen: set[tuple[str, str]],
    parent_of: dict[int, int],
) -> int | None:
    """Recursively link `items[lo:hi]` into a balanced tree.

    Returns this subrange's root index into `items` (the midpoint), or
    `None` for an empty subrange, so the caller can link it to its parent.
    Operates on index bounds into the single shared `items` list -- never
    slices a copy -- so this is O(n) total work and O(log n) recursion
    depth for an n-item group, not O(n log n).

    `parent_of` is populated as a side effect (`parent_of[child] = mid` for
    every parent-child edge emitted), so `_add_redundancy_edges` can later
    walk it to recover every node's depth and ancestor chain without
    re-deriving the tree shape.
    """
    if hi <= lo:
        return None
    mid = (lo + hi) // 2
    left_root = _build_tree_edges(items, lo, mid, group_id, rows, seen, parent_of)
    right_root = _build_tree_edges(items, mid + 1, hi, group_id, rows, seen, parent_of)

    record_id_mid, label_mid = items[mid]
    for child_index in (left_root, right_root):
        if child_index is None:
            continue
        record_id_child, label_child = items[child_index]
        _emit_pair(record_id_mid, label_mid, record_id_child, label_child, group_id, rows, seen)
        parent_of[child_index] = mid
    return mid


def _add_redundancy_edges(
    items: list[tuple[str, float]],
    parent_of: dict[int, int],
    group_id: object,
    rows: list[dict[str, object]],
    seen: set[tuple[str, str]],
    seed: str,
) -> None:
    """Give every node one ancestor-jump edge and one same-depth random edge.

    Two passes, both O(n) total:

    1. Ancestor jump: each node at depth `d` links to its ancestor at depth
       `d // 2`, but only when that's at least 2 levels up (otherwise it's
       just the node's existing parent edge again). This puts the jump
       distance roughly halfway to the root regardless of how deep the
       node already is -- shallow nodes get short or no jumps (they're
       already close to everything), deep nodes get longer ones -- instead
       of every redundant edge landing at the same fixed distance (e.g.
       always "grandparent"), which would concentrate purely at whichever
       depth holds the most nodes.
    2. Same-depth random edge: each node also links to one other,
       uniformly-random node at its *own* depth. Ancestor jumps alone can
       never connect two nodes in different branches except by routing
       through a shared ancestor -- this is the only mechanism here that
       gives sibling branches a direct bridge that doesn't depend on any
       single edge near the root.

    Needs each node's depth, which `parent_of` alone doesn't carry; this
    derives it with one BFS from the root (found as the one index that
    never appears as a key in `parent_of`).
    """
    n = len(items)
    if n < 2:
        return

    children_of: dict[int, list[int]] = {}
    for child, parent in parent_of.items():
        children_of.setdefault(parent, []).append(child)
    root = next(iter(set(range(n)) - set(parent_of.keys())))

    depth_of: dict[int, int] = {root: 0}
    stack = [root]
    while stack:
        node = stack.pop()
        for child in children_of.get(node, ()):
            depth_of[child] = depth_of[node] + 1
            stack.append(child)

    nodes_by_depth: dict[int, list[int]] = {}
    for index, depth in depth_of.items():
        nodes_by_depth.setdefault(depth, []).append(index)

    rng = random.Random(seed)
    for index in range(n):
        depth = depth_of[index]
        record_id_x, label_x = items[index]

        target_depth = depth // 2
        jump_distance = depth - target_depth
        if jump_distance >= 2:
            ancestor = index
            for _ in range(jump_distance):
                ancestor = parent_of[ancestor]
            record_id_a, label_a = items[ancestor]
            _emit_pair(record_id_x, label_x, record_id_a, label_a, group_id, rows, seen)

        peers = nodes_by_depth[depth]
        if len(peers) >= 2:
            candidates = [p for p in peers if p != index]
            peer = rng.choice(candidates)
            record_id_p, label_p = items[peer]
            _emit_pair(record_id_x, label_x, record_id_p, label_p, group_id, rows, seen)


def _add_random_pairs(
    items: list[tuple[str, float]],
    target: int,
    seed: str,
    group_id: object,
    rows: list[dict[str, object]],
    seen: set[tuple[str, str]],
) -> None:
    """Draw up to `target` extra uniformly-random pairs, deduped against `seen`."""
    rng = random.Random(seed)
    n = len(items)
    max_attempts = max(1000, target * 100)
    start_count = len(rows)
    attempts = 0
    while len(rows) - start_count < target and attempts < max_attempts:
        attempts += 1
        i, j = rng.randrange(n), rng.randrange(n)
        if i == j:
            continue
        record_id_a, label_a = items[i]
        record_id_b, label_b = items[j]
        _emit_pair(record_id_a, label_a, record_id_b, label_b, group_id, rows, seen)
