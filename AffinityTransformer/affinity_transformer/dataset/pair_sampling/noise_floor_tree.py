"""Superseded by `noise_aware_multiscale.py`.

This module used to hold a `_noise_floor_tree_pairs` prototype with a real
bug: its clustering step compared each sorted record only to the
*previous* record ("single-linkage"), which lets a long run of small steps
chain an entire dense group into one cluster -- see
`docs/experiments/noise_floor_tree_analysis.md` for the empirical writeup
(SARS_CoV_2 mega group: tau=0.3 collapsed to 1 cluster under the old rule,
triggering a fallback that lost 23% of the group's records).

`noise_aware_multiscale.py` replaces it outright: its anchor construction
(`_build_tau_separated_anchors`) compares each record to its *anchor*
instead of the previous point, which rules out chaining by construction,
and it also fixes a second issue the prototype had -- non-representative
records were forced to attach to a cluster representative, concentrating
degree onto a handful of nodes in dense groups. The replacement never
restricts attachment to representatives and explicitly balances degree.

This file is intentionally left empty of any pair-sampling logic. It is not
imported by `__init__.py`/`pairs.py`/`config.py` (neither this nor the
replacement is wired into the pair-sampling dispatch yet).
"""

from __future__ import annotations
