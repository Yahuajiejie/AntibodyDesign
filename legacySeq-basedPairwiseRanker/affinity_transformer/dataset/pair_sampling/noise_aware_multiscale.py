"""Noise-aware multi-scale pair graph: O(n) edges, no degree hubs, every
edge resolvable.

Replaces the earlier `noise_floor_tree.py` prototype. That prototype had two
problems, both found by analysis before it was ever wired into training:

1. Its clustering rule compared each record only to the *previous* sorted
   record ("single-linkage"), so a long run of small steps could chain an
   entire dense group into one cluster -- the opposite of what clustering by
   noise floor was supposed to do. This module's `_build_tau_separated_anchors`
   instead compares each record to its *anchor* (the start of the current
   run), capping every run's total span at `tau` and making chaining
   impossible by construction.
2. Every non-representative record had to attach to one of its cluster's
   *representative* nodes -- in a dense group with few clusters, that meant
   thousands of records all funnelling onto the same one or two
   representatives, a degree hub the model would overfit to. This module
   never restricts attachment to representatives: every record (anchor or
   not) can be any other record's comparison partner, and partner choice is
   explicitly degree-balanced (`_choose_degree_balanced_partner`) so no
   handful of nodes absorbs a disproportionate share of edges.

Three stages, all O(n) or O(n log n), never O(n^2):

1. **Tau-separated anchors (one O(n) pass).** Sort by `rank_label`, then
   walk left to right; start a new anchor whenever the gap to the *current*
   anchor (not the previous point) reaches `tau`. Any two anchors are
   therefore at least `tau` apart, so a tree built only over anchors has
   every edge automatically resolvable -- no chaining is possible because no
   single small step can ever extend a run past `tau` without registering a
   new anchor immediately.
2. **Backbone over anchors (O(m log m)), reusing `tree.py` verbatim.** Build
   `tree._build_tree_edges` over the `m` anchors (the same recursive
   median-split code `tree.py`/`randomized_tree.py`/the old `noise_floor_tree.py`
   all use), optionally followed by `tree._add_redundancy_edges` on the
   anchors only (`add_anchor_redundancy`, default `False` -- left off so the
   later multi-scale enrichment edges aren't redundant with it, keeping the
   first round of ablations easy to read).
3. **Coverage, then enrichment, both via degree-balanced multi-scale probing
   (O(n) total).** For every non-anchor record, find one partner with a
   resolvable gap by probing increasingly wide label-distance bands (near:
   `[tau, 2*tau)`, medium: `[2*tau, 4*tau)`, far: `[4*tau, +inf)`, then an
   unscaled global fallback covering all of `gap >= tau`) via `bisect` on
   the sorted labels -- never materializing a candidate list larger than a
   few dozen entries, regardless of group size. Then every record (anchor or
   not) gets up to `extra_edges_per_record` more such edges, biased toward
   different scales, so the comparison graph carries near/medium/far
   evidence rather than only the single scale coverage happened to find.
   Partner choice always prefers whichever probed candidate currently has
   the lowest degree, so load spreads across the group instead of piling
   onto a few nodes.

What this module refuses to do, on purpose:

- **Never discards a record that has a resolvable partner.** Every record
  with at least one other record `>= tau` away anywhere in the group gets at
  least one edge to it (`unresolved_policy="skip"`'s "skip" only applies to
  records with provably *no* such partner anywhere in the group -- see
  below -- not to records the prototype simply failed to reach).
- **Never fabricates a definite ranking for genuinely indistinguishable
  records.** If two records are within `tau` of each other, they are never
  paired against each other as a hard 0/1 label -- that comparison carries
  no signal, only noise to overfit (`_emit_pair` already drops exact ties on
  this principle; the noise floor extends the same refusal to *near*-ties).
- **Never amplifies or smooths any label.** Every emitted pair carries the
  two records' own original `rank_label` values, unchanged.

Whole-group degenerate case: if `labels[-1] - labels[0] < tau`, every record
in the group is within `tau` of every other record -- there is no
resolvable pair anywhere in this group at this `tau`, for anyone. This is
the one case where the function returns an empty list (this is `0`-pair
groups acting as `0` training-loss contribution from this group, *not*
`0` records remaining in the dataset). Falling back to the literal-adjacency
tree here, as `tree.py`/the old `noise_floor_tree.py` prototype effectively
do, would silently reintroduce the exact noise-floor-violating comparisons
this module exists to remove, just renamed as a "fallback" -- this module
treats that as the data telling us this group has no reliable ranking
supervision to give, not as a sampling failure to paper over.

The analogous *partial* case -- a record with no resolvable partner even
though other records in the group do have one -- is checked in O(1) per
record (a record's farthest possible partner is always one of the group's
two label extremes, since records are sorted) and also skipped under
`unresolved_policy="skip"`, the only supported policy in this version. A
`"soft"` policy (emit a low-confidence, weighted pair instead of skipping)
is deliberately deferred: it requires a trainer that understands
`target_probability`/`pair_weight`, which `ranknet_loss`'s current hard
0/1-label, unweighted-by-default contract does not, and bolting on soft
labels before there is evidence the hard-label version even helps would
make any later ablation impossible to read cleanly.
"""

from __future__ import annotations

import bisect
import hashlib
import random

import pandas as pd

from .common import _emit_pair
from .tree import _add_redundancy_edges, _build_tree_edges

_SCALE_BANDS: tuple[str, ...] = ("near", "medium", "far")
_BAND_MULTIPLIERS: dict[str, tuple[float, float | None]] = {
    "near": (1.0, 2.0),
    "medium": (2.0, 4.0),
    "far": (4.0, None),
    "global": (1.0, None),
}
_SUPPORTED_UNRESOLVED_POLICIES = ("skip",)


def _noise_aware_multiscale_pairs(
    group_id: object,
    group: pd.DataFrame,
    seed: int,
    tau: float,
    *,
    extra_edges_per_record: int = 2,
    max_degree: int = 12,
    candidate_probe_count: int = 8,
    add_anchor_redundancy: bool = False,
    unresolved_policy: str = "skip",
) -> list[dict[str, object]]:
    """Build the noise-aware multi-scale comparison set for one group.

    Args:
        group_id: This group's `group_id` (carried onto every emitted row).
        group: One group's trainable records; must have `record_id` and
            `rank_label` columns.
        seed: Random seed for every randomized step below (anchor
            redundancy, coverage probing, enrichment probing, the
            near/medium/far second-edge choice). Deterministic for a fixed
            `(seed, group_id)` and fixed input -- see module docstring.
        tau: This group's noise floor: the smallest `rank_label` gap
            considered distinguishable from measurement noise. Must be
            `>= 0`. Callers are expected to look this up per data source
            (assay/metric), not pass one project-wide constant -- see
            module docstring's discussion in the surrounding design doc.
        extra_edges_per_record: How many additional multi-scale edges to
            try to add per record on top of the anchor backbone / coverage
            edge. `0` disables enrichment entirely (useful as an ablation
            isolating "did replacing the coverage backbone alone help").
        max_degree: Soft-in-coverage, hard-in-enrichment cap on a node's
            degree. Coverage always succeeds even if every candidate is
            already at this degree (a record having *some* resolvable
            comparison matters more than perfectly even load). Enrichment
            never exceeds it: an enrichment edge that would push the
            partner over `max_degree` is skipped rather than forced.
        candidate_probe_count: How many candidate indices to sample
            (without materializing the full interval) from each label-band
            search when picking a partner.
        add_anchor_redundancy: When `True`, apply `tree._add_redundancy_edges`
            to the anchor-level backbone. Default `False`: left off so a
            first round of ablations isn't confounded by two different
            redundancy mechanisms (this one and multi-scale enrichment)
            stacked on top of each other.
        unresolved_policy: Only `"skip"` is implemented in this version --
            see module docstring. Anything else raises.

    Returns:
        A list of `_pair_row(...)` dicts. Ties and near-ties (`gap < tau`)
        are never emitted. Every record with at least one `>= tau`-away
        partner anywhere in the group appears in at least one returned
        pair, using its own original, unmodified `rank_label`.

    Raises:
        ValueError: If `tau < 0` or `unresolved_policy` is not `"skip"`.
    """
    if tau < 0:
        raise ValueError(f"tau must be >= 0, got {tau}")
    if unresolved_policy not in _SUPPORTED_UNRESOLVED_POLICIES:
        raise ValueError(
            f"unresolved_policy={unresolved_policy!r} is not supported in "
            f"this version; only {_SUPPORTED_UNRESOLVED_POLICIES!r} is "
            "implemented (see module docstring's discussion of soft labels)."
        )

    items = sorted(
        zip(group["record_id"].astype(str), group["rank_label"].astype(float)),
        key=lambda item: (item[1], item[0]),
    )
    n = len(items)
    if n < 2:
        return []

    record_ids = [item[0] for item in items]
    labels = [item[1] for item in items]

    if labels[-1] - labels[0] < tau:
        # The whole group's label span is narrower than the noise floor: no
        # record here has any resolvable partner. See module docstring --
        # this is deliberately not a fallback to literal adjacency.
        return []

    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    degree = [0] * n
    index_of_record_id = {record_id: index for index, record_id in enumerate(record_ids)}

    anchor_indices = _build_tau_separated_anchors(labels, tau)
    m = len(anchor_indices)
    # Guaranteed m >= 2: if every point stayed within tau of the first
    # anchor, labels[-1] - labels[0] < tau, contradicting the check above.
    assert m >= 2

    anchor_items = [items[i] for i in anchor_indices]
    parent_of_anchor: dict[int, int] = {}
    _build_tree_edges(anchor_items, 0, m, group_id, rows, seen, parent_of_anchor)
    if add_anchor_redundancy:
        _add_redundancy_edges(
            anchor_items, parent_of_anchor, group_id, rows, seen,
            seed=f"{seed}:{group_id}:anchor_redundancy",
        )
    _bump_degree_for_rows(rows, 0, index_of_record_id, degree)
    n_backbone_pairs = len(rows)

    is_anchor = [False] * n
    for anchor_index in anchor_indices:
        is_anchor[anchor_index] = True

    coverage_rng = random.Random(f"{seed}:{group_id}:coverage")
    n_unresolved_records = 0
    for index in range(n):
        if is_anchor[index]:
            continue
        if not _has_any_resolvable_partner(labels, index, tau):
            n_unresolved_records += 1
            continue
        partner = _find_coverage_partner(
            index,
            labels=labels,
            tau=tau,
            degree=degree,
            record_ids=record_ids,
            seen=seen,
            rng=coverage_rng,
            probe_count=candidate_probe_count,
        )
        if partner is None:
            # A resolvable partner provably exists (checked above) but
            # bounded probing did not land on it this run -- rare, and
            # counted rather than chased with unbounded extra work.
            n_unresolved_records += 1
            continue
        _emit_and_track(
            record_ids[index], labels[index], index,
            record_ids[partner], labels[partner], partner,
            group_id, rows, seen, degree,
        )
    n_coverage_pairs = len(rows) - n_backbone_pairs

    if extra_edges_per_record > 0:
        enrichment_rng = random.Random(f"{seed}:{group_id}:enrichment")
        _add_enrichment_edges(
            n,
            labels=labels,
            record_ids=record_ids,
            tau=tau,
            seed=seed,
            group_id=group_id,
            degree=degree,
            seen=seen,
            rows=rows,
            rng=enrichment_rng,
            extra_edges_per_record=extra_edges_per_record,
            max_degree=max_degree,
            probe_count=candidate_probe_count,
        )

    return rows


def _build_tau_separated_anchors(labels: list[float], tau: float) -> list[int]:
    """Partition sorted `labels` into anchors at least `tau` apart.

    Single O(n) left-to-right pass. Unlike single-linkage clustering
    ("gap to the *previous* point exceeds tau"), this compares each point to
    the *current anchor* -- the most recent point that itself started a new
    run -- so a run's total span can never exceed `tau` no matter how many
    small steps it took to get there. That is what prevents the chaining
    failure mode the earlier `noise_floor_tree.py` prototype had: with the
    previous-point rule, a long sequence of steps each just under `tau`
    could chain an entire group into one giant run even though its first and
    last members are far apart; anchoring to the run's start makes that
    impossible by construction.
    """
    anchors = [0]
    last_anchor = 0
    for i in range(1, len(labels)):
        if labels[i] - labels[last_anchor] >= tau:
            anchors.append(i)
            last_anchor = i
    return anchors


def _has_any_resolvable_partner(labels: list[float], index: int, tau: float) -> bool:
    """O(1) check for whether `index` has any `>= tau`-away partner at all.

    Since `labels` is sorted, the farthest possible partner from any point
    is always one of the group's two extremes -- no need to search to find
    out whether searching is even worth attempting.
    """
    y = labels[index]
    return max(y - labels[0], labels[-1] - y) >= tau


def _candidate_intervals(
    labels: list[float], index: int, tau: float, band: str
) -> list[tuple[int, int]]:
    """Index ranges `[start, end)` into `labels` whose gap from `labels[index]`
    falls in the requested `band`, on whichever side(s) are non-empty.

    `band` is one of `"near"` (`[tau, 2*tau)`), `"medium"` (`[2*tau, 4*tau)`),
    `"far"` (`[4*tau, +inf)`), or `"global"` (`[tau, +inf)` -- the union of
    the other three, used as a last-resort fallback that doesn't care which
    scale the partner ends up at). Uses `bisect` against the already-sorted
    `labels`, so this never materializes or scans the candidates themselves
    -- O(log n) regardless of how many indices the returned ranges cover.
    """
    if band not in _BAND_MULTIPLIERS:
        raise ValueError(f"unknown band {band!r}")
    low_mult, high_mult = _BAND_MULTIPLIERS[band]
    y = labels[index]
    n = len(labels)
    intervals: list[tuple[int, int]] = []

    right_lo = bisect.bisect_left(labels, y + low_mult * tau)
    right_hi = n if high_mult is None else bisect.bisect_left(labels, y + high_mult * tau)
    if right_lo < right_hi:
        intervals.append((right_lo, right_hi))

    left_hi = bisect.bisect_right(labels, y - low_mult * tau)
    left_lo = 0 if high_mult is None else bisect.bisect_right(labels, y - high_mult * tau)
    if left_lo < left_hi:
        intervals.append((left_lo, left_hi))

    return intervals


def _probe_interval(
    start: int, end: int, rng: random.Random, count: int
) -> list[int]:
    """Sample up to `count` distinct indices from `[start, end)`.

    Returns every index in the range when it has `count` or fewer entries;
    otherwise draws `count` distinct indices at random. Never builds
    `list(range(start, end))` for a large range -- that is the O(n) blow-up
    this module exists to avoid.
    """
    size = end - start
    if size <= 0:
        return []
    if size <= count:
        return list(range(start, end))
    chosen: set[int] = set()
    while len(chosen) < count:
        chosen.add(rng.randrange(start, end))
    return list(chosen)


def _choose_degree_balanced_partner(
    index: int,
    intervals: list[tuple[int, int]],
    *,
    labels: list[float],
    record_ids: list[str],
    degree: list[int],
    seen: set[tuple[str, str]],
    rng: random.Random,
    max_degree: int | None,
    probe_count: int,
    exclude: set[int] | None = None,
) -> int | None:
    """Probe `intervals` and return whichever valid candidate has the
    lowest current degree (ties broken by `rng`), or `None` if none qualify.

    A candidate is valid when it is not `index` itself, not an exact label
    tie with `index` (that pair would be dropped by `_emit_pair` anyway --
    rejecting it here avoids wasting a probe slot's worth of selection
    pressure on a guaranteed no-op), not already in `seen`/`exclude`, and
    (only when `max_degree` is not `None`) currently under `max_degree`.
    `max_degree=None` makes this a pure degree-*preference* with no hard
    cutoff -- coverage's contract (always succeed if a partner exists at
    all) needs that; enrichment's contract (never exceed `max_degree`)
    passes an actual limit.
    """
    candidates: list[int] = []
    for start, end in intervals:
        candidates.extend(_probe_interval(start, end, rng, probe_count))

    label_x = labels[index]
    record_id_x = record_ids[index]
    best: int | None = None
    best_score: tuple[int, float] | None = None
    for j in candidates:
        if j == index:
            continue
        if exclude is not None and j in exclude:
            continue
        if labels[j] == label_x:
            continue
        if max_degree is not None and degree[j] >= max_degree:
            continue
        key = tuple(sorted((record_id_x, record_ids[j])))
        if key in seen:
            continue
        score = (degree[j], rng.random())
        if best_score is None or score < best_score:
            best_score = score
            best = j
    return best


def _find_coverage_partner(
    index: int,
    *,
    labels: list[float],
    tau: float,
    degree: list[int],
    record_ids: list[str],
    seen: set[tuple[str, str]],
    rng: random.Random,
    probe_count: int,
    max_retries_per_band: int = 3,
) -> int | None:
    """Find one `>= tau`-away partner for `index`, trying narrower bands
    first. `max_degree=None` throughout -- coverage never refuses a
    resolvable partner over degree, only prefers a lower-degree one when
    there's a choice. Retries each band a few times before moving on: a
    single unlucky probe (every sampled candidate already `seen`, or an
    exact tie) should not make an available band look empty.
    """
    for band in (*_SCALE_BANDS, "global"):
        intervals = _candidate_intervals(labels, index, tau, band)
        if not intervals:
            continue
        for _attempt in range(max_retries_per_band):
            partner = _choose_degree_balanced_partner(
                index,
                intervals,
                labels=labels,
                record_ids=record_ids,
                degree=degree,
                seen=seen,
                rng=rng,
                max_degree=None,
                probe_count=probe_count,
            )
            if partner is not None:
                return partner
    return None


def _stable_second_band(seed: int, group_id: object, record_id: str) -> str:
    """Deterministically (across processes) choose `"medium"` or `"far"`
    for a record's second enrichment edge, biased toward `"medium"`.

    Uses a stable hash (`hashlib`, not Python's built-in `hash()`) because
    `hash()` of strings is randomized per-process by default
    (`PYTHONHASHSEED`) -- using it here would silently break the
    same-seed-same-result reproducibility this module otherwise guarantees.
    """
    digest = hashlib.sha256(f"{seed}:{group_id}:{record_id}:second_band".encode()).hexdigest()
    fraction = int(digest[:8], 16) / 0xFFFFFFFF
    return "medium" if fraction < 0.6 else "far"


def _bump_degree_for_rows(
    rows: list[dict[str, object]],
    start: int,
    index_of_record_id: dict[str, int],
    degree: list[int],
) -> None:
    """Increment `degree` for every row added since position `start`.

    Used after bulk-emitting via `tree._build_tree_edges`/
    `_add_redundancy_edges`, which call `_emit_pair` directly and so don't
    go through this module's own degree bookkeeping.
    """
    for row in rows[start:]:
        degree[index_of_record_id[str(row["record_id_i"])]] += 1
        degree[index_of_record_id[str(row["record_id_j"])]] += 1


def _emit_and_track(
    record_id_a: str,
    label_a: float,
    index_a: int,
    record_id_b: str,
    label_b: float,
    index_b: int,
    group_id: object,
    rows: list[dict[str, object]],
    seen: set[tuple[str, str]],
    degree: list[int],
) -> bool:
    """`_emit_pair`, then bump `degree` for both endpoints iff a row was
    actually appended (it may silently no-op on an exact tie or duplicate).

    Centralizing this avoids the easy mistake of bumping degree
    unconditionally and drifting out of sync with what `_emit_pair` actually
    decided to keep.
    """
    before = len(rows)
    _emit_pair(record_id_a, label_a, record_id_b, label_b, group_id, rows, seen)
    if len(rows) > before:
        degree[index_a] += 1
        degree[index_b] += 1
        return True
    return False


def _add_enrichment_edges(
    n: int,
    *,
    labels: list[float],
    record_ids: list[str],
    tau: float,
    seed: int,
    group_id: object,
    degree: list[int],
    seen: set[tuple[str, str]],
    rows: list[dict[str, object]],
    rng: random.Random,
    extra_edges_per_record: int,
    max_degree: int,
    probe_count: int,
) -> None:
    """Give every record up to `extra_edges_per_record` additional
    multi-scale edges (on top of its backbone/coverage edge), spread across
    near/medium/far scales rather than concentrated at one.

    The first extra edge prefers `near`, falling back to `medium` then
    `far`. Each subsequent edge prefers `medium` or `far` (a deterministic,
    per-record choice via `_stable_second_band`, biased toward `medium`),
    excluding partners this record already used in an earlier enrichment
    edge this call. `max_degree` is a hard cap here (unlike coverage): an
    edge that would push a candidate over it is simply skipped rather than
    forced, so enrichment never recreates the degree-hub problem it exists
    to avoid introducing in the first place.
    """
    for index in range(n):
        used_partners: set[int] = set()
        record_id_x, label_x = record_ids[index], labels[index]

        for band in _SCALE_BANDS:
            intervals = _candidate_intervals(labels, index, tau, band)
            if not intervals:
                continue
            partner = _choose_degree_balanced_partner(
                index, intervals,
                labels=labels, record_ids=record_ids, degree=degree, seen=seen,
                rng=rng, max_degree=max_degree, probe_count=probe_count,
                exclude=used_partners,
            )
            if partner is not None:
                if _emit_and_track(
                    record_id_x, label_x, index,
                    record_ids[partner], labels[partner], partner,
                    group_id, rows, seen, degree,
                ):
                    used_partners.add(partner)
                break

        for _extra in range(1, extra_edges_per_record):
            preferred = _stable_second_band(seed, group_id, record_id_x)
            band_order = [preferred] + [b for b in ("medium", "far") if b != preferred]
            placed = False
            for band in band_order:
                intervals = _candidate_intervals(labels, index, tau, band)
                if not intervals:
                    continue
                partner = _choose_degree_balanced_partner(
                    index, intervals,
                    labels=labels, record_ids=record_ids, degree=degree, seen=seen,
                    rng=rng, max_degree=max_degree, probe_count=probe_count,
                    exclude=used_partners,
                )
                if partner is not None:
                    if _emit_and_track(
                        record_id_x, label_x, index,
                        record_ids[partner], labels[partner], partner,
                        group_id, rows, seen, degree,
                    ):
                        used_partners.add(partner)
                    placed = True
                    break
            if not placed:
                # No candidate available under max_degree at any scale for
                # this extra edge -- skip it rather than force one; degree
                # balance takes priority over hitting the exact edge count.
                continue


def _audit_noise_aware_multiscale(
    group_id: object,
    group: pd.DataFrame,
    tau: float,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    """Compute the §11 audit statistics for one group's already-built `rows`.

    Not on the hot path of `_noise_aware_multiscale_pairs` itself (every
    other strategy in this package returns only rows; keeping that contract
    here too keeps `build_pairs` uniform) -- call this separately from
    analysis scripts or tests when the breakdown is needed.
    """
    record_ids = group["record_id"].astype(str).tolist()
    labels_by_id = dict(zip(record_ids, group["rank_label"].astype(float)))
    n_records = len(record_ids)

    pairs = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["record_id_i", "record_id_j", "label_i", "label_j"]
    )
    gaps = (pairs["label_i"] - pairs["label_j"]).abs() if not pairs.empty else pd.Series([], dtype=float)

    degree: dict[str, int] = {record_id: 0 for record_id in record_ids}
    for row in rows:
        degree[str(row["record_id_i"])] += 1
        degree[str(row["record_id_j"])] += 1
    degree_values = pd.Series(list(degree.values()), dtype=float)

    covered = set(pairs["record_id_i"]) | set(pairs["record_id_j"]) if not pairs.empty else set()
    sorted_labels = sorted(labels_by_id.values())
    n_intrinsically_unresolved = sum(
        1 for label in labels_by_id.values()
        if max(label - sorted_labels[0], sorted_labels[-1] - label) < tau
    ) if sorted_labels and (sorted_labels[-1] - sorted_labels[0]) >= tau else n_records

    hard_resolvable_ids = {
        record_id for record_id, label in labels_by_id.items()
        if sorted_labels and max(label - sorted_labels[0], sorted_labels[-1] - label) >= tau
    }

    def _gap_fraction(low: float, high: float | None) -> float:
        if gaps.empty:
            return 0.0
        mask = gaps >= low
        if high is not None:
            mask &= gaps < high
        return float(mask.mean())

    return {
        "n_records": n_records,
        "n_pairs": len(rows),
        "n_unresolved_records": n_intrinsically_unresolved,
        "record_coverage_all": len(covered) / n_records if n_records else 0.0,
        "record_coverage_hard_resolvable": (
            len(covered & hard_resolvable_ids) / len(hard_resolvable_ids)
            if hard_resolvable_ids else 1.0
        ),
        "degree_min": float(degree_values.min()) if not degree_values.empty else 0.0,
        "degree_median": float(degree_values.median()) if not degree_values.empty else 0.0,
        "degree_p90": float(degree_values.quantile(0.9)) if not degree_values.empty else 0.0,
        "degree_max": float(degree_values.max()) if not degree_values.empty else 0.0,
        "gap_min": float(gaps.min()) if not gaps.empty else None,
        "gap_median": float(gaps.median()) if not gaps.empty else None,
        "gap_max": float(gaps.max()) if not gaps.empty else None,
        "near_pair_fraction": _gap_fraction(tau, 2 * tau),
        "medium_pair_fraction": _gap_fraction(2 * tau, 4 * tau),
        "far_pair_fraction": _gap_fraction(4 * tau, None),
    }
