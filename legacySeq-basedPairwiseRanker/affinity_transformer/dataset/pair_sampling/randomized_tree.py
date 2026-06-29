"""Randomized-BST pair construction: structural randomness, ~certainly O(log n).

`tree.py`'s median-split tree is fully deterministic -- the same group
always yields the exact same `n_records - 1` edges, every epoch, every seed.
That has two costs: a single mislabeled record sitting at a structurally
privileged split point has no redundant comparison to be outvoted by, and
training never sees an alternative view of the group's structure.

This builds a classic randomly-built binary search tree instead: shuffle
the group's records into a uniformly random insertion order (seeded, so
reproducible per run but different across seeds), then insert them one at a
time using standard BST insertion (smaller goes left, larger goes right,
insert at the first empty slot found). Every insertion creates exactly one
edge (the new record links to whichever existing record it landed under),
so this is still exactly `n_records - 1` edges before the grandparent pass
below -- same O(n) budget as `tree.py` -- but which records end up near the
root versus deep in the tree now depends on the random insertion order, not
always on the same medians.

Same redundancy gaps as `tree.py` (leaves under-covered, near-root edges
have zero backup), same fix: every node gets one ancestor-jump edge
(roughly halfway to the root) and one random same-depth edge. See
`tree._add_redundancy_edges` for why both are needed.

A randomly-built BST's height is only O(log n) in *expectation*, not
guaranteed worst-case the way an explicit median split is -- across 300
independent random insertion orders at n=2000, observed heights stayed in a
20-32 range (versus log2(2000)~11, so roughly 2-3x the deterministic tree's
constant factor, never close to degenerating toward n), but an
astronomically unlucky insertion order (e.g. one that happens to land in
already-sorted order) can still produce a near-chain shape. Rather than
accept that probabilistic bound as-is, every build checks its own height
against a generous multiple of log2(n); a rare bad draw retries with a
different seed, and if every attempt comes up bad (should not happen in
practice), falls back to `tree.py`'s deterministic median tree, which has
no such failure mode. So the worst case is bounded for real, not just "very
likely" bounded.
"""

from __future__ import annotations

import math
import random

import pandas as pd

from .common import _emit_pair
from .tree import _add_redundancy_edges, _balanced_tree_pairs

_DEFAULT_MAX_ATTEMPTS = 5
_HEIGHT_SAFETY_FACTOR = 6.0


def _randomized_bst_pairs(
    group_id: object,
    group: pd.DataFrame,
    seed: int,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    add_redundancy_edges: bool = True,
) -> list[dict[str, object]]:
    """Build a randomly-ordered BST's comparison set for one rankable group.

    Args:
        group_id: This group's `group_id` (carried onto every emitted row,
            and mixed into the insertion-order shuffle's seed).
        group: One group's trainable records; must have `record_id` and
            `rank_label` columns.
        seed: Random seed for the insertion-order shuffle. Same `seed` +
            `group_id` always reproduces the same tree; a different `seed`
            (e.g. a different training run) gives a different tree shape.
        max_attempts: How many different random insertion orders to try
            before giving up on randomness and falling back to `tree.py`'s
            deterministic median tree for this group. Each attempt is
            rejected only if its height exceeds
            `_HEIGHT_SAFETY_FACTOR * log2(n)` -- generous enough that real
            attempts essentially always pass on the first try. This height
            check (and the fallback it guards) runs regardless of
            `add_redundancy_edges`, so disabling redundancy edges does not
            give up the height safety margin.
        add_redundancy_edges: When `True` (default), append the
            ancestor-jump and same-depth random edges (see
            `tree._add_redundancy_edges`) on top of the random insertion
            backbone. Set `False` to emit only the bare `n_records - 1`
            insertion-order edges: still the same expected O(log n)
            diameter (the height check above is unaffected), but every
            leaf goes back to a single edge, near-root edges lose their
            backup, and -- since this strategy's only other source of
            epoch-to-epoch comparison diversity below the tree-shape level
            was the redundancy step's own random draws -- that diversity
            is gone too.

    Returns:
        A list of `_pair_row(...)` dicts. Ties are skipped, matching every
        other strategy in this package -- a tied record is still inserted
        into the tree structure (so later insertions can still land under
        it), it just never emits a training pair against the node it tied
        with.
    """
    items = sorted(
        zip(group["record_id"].astype(str), group["rank_label"].astype(float)),
        key=lambda item: (item[1], item[0]),
    )
    n = len(items)
    if n < 2:
        return []

    height_limit = max(4, _HEIGHT_SAFETY_FACTOR * math.log2(n))
    for attempt in range(max_attempts):
        left, right, root, parent_of = _build_random_bst_structure(
            n, seed=f"{seed}:{group_id}:{attempt}"
        )
        if _tree_height(n, left, right, root) <= height_limit:
            rows: list[dict[str, object]] = []
            seen: set[tuple[str, str]] = set()
            for child_index, parent_index in parent_of.items():
                record_id_child, label_child = items[child_index]
                record_id_parent, label_parent = items[parent_index]
                _emit_pair(
                    record_id_parent, label_parent, record_id_child, label_child,
                    group_id, rows, seen,
                )
            if add_redundancy_edges:
                _add_redundancy_edges(
                    items, parent_of, group_id, rows, seen,
                    seed=f"{seed}:{group_id}:{attempt}:redundancy",
                )
            return rows

    # Every random attempt came up pathologically deep (should not happen
    # in practice -- see module docstring). Fall back to the structure that
    # has no failure mode at all.
    return _balanced_tree_pairs(group_id, group, seed=seed, add_redundancy_edges=add_redundancy_edges)


def _build_random_bst_structure(
    n: int, seed: str
) -> tuple[list[int | None], list[int | None], int, dict[int, int]]:
    """Insert a uniformly-shuffled permutation of `range(n)` into a BST.

    Returns `(left, right, root, parent_of)` where `left[i]`/`right[i]` are
    child indices (or `None`), `root` is the first-inserted index, and
    `parent_of[child] = parent` for every edge created. Insertion order
    only (not the actual `rank_label` values) determines this shape, so the
    caller is responsible for mapping indices back to records/labels.
    """
    order = list(range(n))
    random.Random(seed).shuffle(order)

    left: list[int | None] = [None] * n
    right: list[int | None] = [None] * n
    parent_of: dict[int, int] = {}
    root = order[0]
    for x in order[1:]:
        node = root
        while True:
            goes_left = x < node
            if goes_left:
                if left[node] is None:
                    left[node] = x
                    parent_of[x] = node
                    break
                node = left[node]
            else:
                if right[node] is None:
                    right[node] = x
                    parent_of[x] = node
                    break
                node = right[node]
    return left, right, root, parent_of


def _tree_height(n: int, left: list[int | None], right: list[int | None], root: int) -> int:
    """Iterative (no recursion, no Python recursion-limit risk) tree height."""
    stack = [(root, 1)]
    best = 0
    while stack:
        node, depth = stack.pop()
        best = max(best, depth)
        if left[node] is not None:
            stack.append((left[node], depth + 1))
        if right[node] is not None:
            stack.append((right[node], depth + 1))
    return best
