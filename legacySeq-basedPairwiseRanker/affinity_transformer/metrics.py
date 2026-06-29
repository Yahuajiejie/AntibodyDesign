"""Group-level evaluation metrics.

(spec docs/programming_spec.md §5.6)

This module only computes metrics from a `predictions` table that already
has one row per `(record_id, group_id)` with a model `score` and the
dataset's `rank_label`. It does not read raw CSVs, does not build
pairs/groups (that is `dataset.py`, spec §5.2), and does not run the model
(that is `model/` / `trainer.py`).

Spearman correlation is computed without `scipy` (unavailable in this
sandbox): Spearman's rho is the Pearson correlation of the ranks, so
`x.rank().corr(y.rank())` is used instead of `scipy.stats.spearmanr`. This
gives identical results to `scipy.stats.spearmanr` (no tie-correction
differences for the purposes here, since `pandas.Series.rank()` already
applies the standard average-rank tie-breaking).
"""

from __future__ import annotations

import pandas as pd

#: Columns required on the `predictions` argument of `compute_group_spearman`
#: (spec §5.6 "输入字段").
PREDICTIONS_COLUMNS = (
    "record_id",
    "group_id",
    "rank_label",
    "score",
    "label_kind",
    "dataset_id",
)

#: Columns of the `pd.DataFrame` returned by `compute_group_spearman` (spec
#: §5.6 "输出字段").
GROUP_SPEARMAN_COLUMNS = (
    "group_id",
    "dataset_id",
    "label_kind",
    "n_records",
    "n_unique_labels",
    "spearman",
)


def compute_group_spearman(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute per-group Spearman correlation between `rank_label` and `score`.

    Args:
        predictions: One row per scored record, with at least the columns
            `record_id, group_id, rank_label, score, label_kind, dataset_id`
            (spec §5.6 "输入字段"). `dataset_id` and `label_kind` are assumed
            constant within a `group_id` (true by construction of
            `dataset.build_groups`, spec §5.2); the first value seen in each
            group is used.

    Returns:
        One row per distinct `group_id`, with columns `group_id, dataset_id,
        label_kind, n_records, n_unique_labels, spearman` (spec §5.6 "输出
        字段"):
            - `n_records`: number of rows for this group.
            - `n_unique_labels`: number of distinct `rank_label` values in
              this group.
            - `spearman`: Spearman correlation between `rank_label` and
              `score` within the group, computed as the Pearson correlation
              of their ranks (no `scipy` dependency). `NaN` if
              `n_unique_labels < 2` (spec §5.6 rule 1: such groups are not
              scored) or if `score` is constant within the group (Spearman
              undefined). Groups with `n_unique_labels < 2` still appear as
              a row here (with `spearman = NaN`) so callers can count
              skipped groups; `summarize_group_spearman` does this.
        Row order follows the order `group_id` values first appear in
        `predictions`. Empty input yields an empty `DataFrame` with the
        columns above.

    Raises:
        ValueError: If `predictions` is missing any of the required columns
            listed above.
    """
    missing = [c for c in PREDICTIONS_COLUMNS if c not in predictions.columns]
    if missing:
        raise ValueError(
            f"predictions is missing required column(s): {missing}. "
            f"Required columns: {list(PREDICTIONS_COLUMNS)}."
        )

    rows: list[dict[str, object]] = []
    for group_id, group in predictions.groupby("group_id", sort=False):
        n_unique_labels = int(group["rank_label"].nunique())
        if n_unique_labels < 2:
            spearman = float("nan")
        else:
            spearman = group["rank_label"].rank().corr(group["score"].rank())

        rows.append({
            "group_id": group_id,
            "dataset_id": group["dataset_id"].iloc[0],
            "label_kind": group["label_kind"].iloc[0],
            "n_records": len(group),
            "n_unique_labels": n_unique_labels,
            "spearman": spearman,
        })

    return pd.DataFrame(rows, columns=list(GROUP_SPEARMAN_COLUMNS))


def summarize_group_spearman(group_metrics: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    """Aggregate per-group Spearman correlations into macro/weighted averages.

    Implements spec §5.6 rules 2-4:
        - rule 2: reports both a macro average (each group weighted equally,
          regardless of size -- avoids large groups dominating the metric)
          and a `n_records`-weighted average (each record weighted equally).
        - rule 3: never reports a single overall number on its own -- always
          alongside the per-`label_kind` breakdown.
        - rule 4: `binary`-labelled groups are aggregated separately from
          other `label_kind` values (e.g. `experimental`), via their own
          entry in the returned dict.

    Args:
        group_metrics: Output of `compute_group_spearman`, i.e. a
            `DataFrame` with at least the columns `label_kind`, `n_records`,
            and `spearman` (one row per group; `spearman` may be `NaN` for
            skipped groups).

    Returns:
        A dict with one entry per distinct `label_kind` value present in
        `group_metrics`, plus an `"overall"` entry combining all groups
        regardless of `label_kind`. Each entry is a dict with:
            - `n_groups`: total number of groups in this subset.
            - `n_valid_groups`: number of groups with non-`NaN` `spearman`
              (spec §5.6 rule 1 groups, i.e. `n_unique_labels >= 2` and a
              non-degenerate `score`).
            - `n_skipped_groups`: `n_groups - n_valid_groups`.
            - `macro_spearman`: unweighted mean `spearman` over valid groups,
              `NaN` if `n_valid_groups == 0`.
            - `weighted_spearman`: `n_records`-weighted mean `spearman` over
              valid groups, `NaN` if `n_valid_groups == 0`.
        If `group_metrics` is empty, only the `"overall"` entry is present,
        with all counts `0` and both averages `NaN`.
    """
    summary: dict[str, dict[str, float | int]] = {"overall": _summarize_subset(group_metrics)}
    for label_kind, subset in group_metrics.groupby("label_kind"):
        summary[str(label_kind)] = _summarize_subset(subset)
    return summary


def _summarize_subset(subset: pd.DataFrame) -> dict[str, float | int]:
    """Compute the macro/weighted-average summary for one subset of groups.

    Args:
        subset: Rows of a `compute_group_spearman` result (any subset, e.g.
            all rows, or just those with `label_kind == "binary"`).

    Returns:
        Dict with keys `n_groups, n_valid_groups, n_skipped_groups,
        macro_spearman, weighted_spearman` (see `summarize_group_spearman`).
    """
    n_groups = len(subset)
    valid = subset.dropna(subset=["spearman"])
    n_valid_groups = len(valid)
    n_skipped_groups = n_groups - n_valid_groups

    if n_valid_groups == 0:
        macro_spearman = float("nan")
        weighted_spearman = float("nan")
    else:
        macro_spearman = float(valid["spearman"].mean())
        weights = valid["n_records"]
        weighted_spearman = float((valid["spearman"] * weights).sum() / weights.sum())

    return {
        "n_groups": n_groups,
        "n_valid_groups": n_valid_groups,
        "n_skipped_groups": n_skipped_groups,
        "macro_spearman": macro_spearman,
        "weighted_spearman": weighted_spearman,
    }
