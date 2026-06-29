"""Tests for `splits.build_within_antigen_split` -- the auxiliary
"known-antigen, new-antibody" protocol from programming_spec_v1.0.md
section 3.2.

Design ("做法一" / leniency decision, see conversation around 2026-06-22):
each `group_id` is split independently by antibody-sequence identity. The
same antibody sequence is explicitly ALLOWED to land in train via one group
and valid/test via a different, unrelated group -- that is not leakage,
because `dataset.pairs.build_pairs` only ever constructs comparison pairs
*within* one group_id, so the specific relationship being evaluated (this
antibody vs THIS antigen's other candidates) was never trained on either
way. What must never cross a split is a `record_id`, and within any single
group, an antibody assigned to that group's train must not also appear in
that same group's valid/test.
"""

from __future__ import annotations

import pandas as pd
import pytest

from affinity_transformer.splits import build_within_antigen_split

_BASE_ROW = {
    "dataset_id": "studyA/tableA",
    "keep_for_training": True,
    "antigen_key": "agA",
    "single_chain_sequence": None,
}


def _row(record_id: str, group_id: str, heavy: str, light: str, label: float) -> dict:
    return {
        **_BASE_ROW,
        "record_id": record_id,
        "group_id": group_id,
        "heavy_chain": heavy,
        "light_chain": light,
        "rank_label": label,
    }


def _antibody_group(group_id: str, n_antibodies: int, offset: int = 0) -> list[dict]:
    """One group with `n_antibodies` distinct antibody-sequence units, one
    record per unit. `offset` controls the antibody-name range, so two
    groups can be made to share (or not share) antibody identities.
    """
    rows = []
    for i in range(offset, offset + n_antibodies):
        rows.append(_row(
            record_id=f"{group_id}/{i}",
            group_id=group_id,
            heavy=f"HEAVY{i:03d}QVQLVQSGAEVKKPGASVKVSCKAS",
            light=f"LIGHT{i:03d}DIQMTQSPSSLSASVGDRVTITC",
            label=float(i),
        ))
    return rows


def _tiny_group(group_id: str = "g/tiny") -> list[dict]:
    """Only 2 distinct antibody units (one duplicated) -- must get pinned."""
    return [
        _row(f"{group_id}/0", group_id, "HEAVY_A", "LIGHT_A", 1.0),
        _row(f"{group_id}/1", group_id, "HEAVY_A", "LIGHT_A", 1.1),  # same antibody, dup row
        _row(f"{group_id}/2", group_id, "HEAVY_B", "LIGHT_B", 2.0),
    ]


def _records(*groups: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for group in groups:
        rows.extend(group)
    return pd.DataFrame(rows)


def _identities(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(frame["heavy_chain"], frame["light_chain"]))


def test_antibody_never_crosses_a_split_within_its_own_group():
    records = _records(_antibody_group("g/big", n_antibodies=40))
    result = build_within_antigen_split(records, valid_fraction=0.2, test_fraction=0.2, seed=0, min_eval_records=3)

    train_ids, valid_ids, test_ids = _identities(result.train), _identities(result.valid), _identities(result.test)
    assert not (train_ids & valid_ids)
    assert not (train_ids & test_ids)
    assert not (valid_ids & test_ids)
    assert valid_ids and test_ids


def test_group_id_legitimately_overlaps_train_and_holdout():
    records = _records(_antibody_group("g/big", n_antibodies=40))
    result = build_within_antigen_split(records, valid_fraction=0.2, test_fraction=0.2, seed=0, min_eval_records=3)
    assert "g/big" in set(result.train["group_id"])
    assert "g/big" in set(result.valid["group_id"])
    assert "g/big" in set(result.test["group_id"])


def test_same_antibody_across_two_groups_is_allowed_to_cross_splits():
    """The core 做法一 behavior: g/A and g/B share the SAME antibody
    identities (offset=0 for both, simulating one antibody tested against
    two different antigens/metrics). It is fine -- not leakage -- for an
    antibody to land in g/A's train AND g/B's valid, because the two
    groups' comparisons are independent (build_pairs never pairs across
    groups). Confirm this actually happens with this fixture (i.e. that the
    implementation doesn't silently impose a global constraint), and that
    the leakage report still reports PASS.
    """
    records = _records(
        _antibody_group("g/A", n_antibodies=40, offset=0),
        _antibody_group("g/B", n_antibodies=40, offset=0),
    )
    result = build_within_antigen_split(records, valid_fraction=0.2, test_fraction=0.2, seed=0, min_eval_records=3)

    train_ids = _identities(result.train)
    valid_ids = _identities(result.valid)
    test_ids = _identities(result.test)
    # Some antibody identity must appear in train (via one group) AND in
    # valid or test (via the other group) -- if this fails, the
    # implementation has regressed to the over-strict global-partition
    # design that was deliberately rejected.
    assert (train_ids & valid_ids) or (train_ids & test_ids)

    failed = result.leakage_report.loc[result.leakage_report["status"] != "PASS"]
    assert failed.empty


def test_within_one_group_antibody_still_cannot_cross_that_groups_own_splits():
    """The thing 做法一 still requires: WITHIN a single group, an antibody
    in that group's train must not also be in that SAME group's valid/test
    (otherwise build_pairs would literally repeat a comparison)."""
    records = _records(_antibody_group("g/big", n_antibodies=40))
    result = build_within_antigen_split(records, valid_fraction=0.2, test_fraction=0.2, seed=0, min_eval_records=3)

    def group_identities(frame, group_id):
        rows = frame.loc[frame["group_id"] == group_id]
        return _identities(rows)

    train_g = group_identities(result.train, "g/big")
    valid_g = group_identities(result.valid, "g/big")
    test_g = group_identities(result.test, "g/big")
    assert not (train_g & valid_g)
    assert not (train_g & test_g)
    assert not (valid_g & test_g)


def test_tiny_group_is_pinned_entirely_to_train():
    records = _records(_antibody_group("g/big", n_antibodies=40), _tiny_group())
    result = build_within_antigen_split(records, valid_fraction=0.2, test_fraction=0.2, seed=0, min_eval_records=3)

    assert "g/tiny" not in set(result.valid["group_id"])
    assert "g/tiny" not in set(result.test["group_id"])
    assert set(records.loc[records["group_id"] == "g/tiny", "record_id"]) <= set(result.train["record_id"])

    pinned = result.pinned_groups
    tiny_row = pinned.loc[pinned["group_id"] == "g/tiny"].iloc[0]
    assert tiny_row["n_antibody_units"] == 2
    assert "fewer than 3" in tiny_row["reason"]


def test_min_eval_records_pins_groups_too_small_to_holdout_reliably():
    # g/small9 has 9 distinct antibodies, 1 record each -> with
    # valid/test_fraction=0.1 each, projected valid/test would be ~1 record,
    # below a strict min_eval_records threshold. g/big is included so the
    # overall split still has something in valid/test.
    records = _records(
        _antibody_group("g/small9", n_antibodies=9),
        _antibody_group("g/big", n_antibodies=60),
    )
    result = build_within_antigen_split(records, valid_fraction=0.1, test_fraction=0.1, seed=0, min_eval_records=5)

    assert "g/small9" not in set(result.valid["group_id"])
    assert "g/small9" not in set(result.test["group_id"])
    pinned = result.pinned_groups
    row = pinned.loc[pinned["group_id"] == "g/small9"].iloc[0]
    assert "min_eval_records" in row["reason"]


def test_reproducible_for_same_seed_and_varies_with_different_seed():
    records = _records(_antibody_group("g/big", n_antibodies=40))
    a = build_within_antigen_split(records, 0.2, 0.2, seed=0, min_eval_records=3)
    b = build_within_antigen_split(records, 0.2, 0.2, seed=0, min_eval_records=3)
    assert sorted(a.valid["record_id"]) == sorted(b.valid["record_id"])
    assert sorted(a.test["record_id"]) == sorted(b.test["record_id"])

    c = build_within_antigen_split(records, 0.2, 0.2, seed=1, min_eval_records=3)
    assert sorted(a.valid["record_id"]) != sorted(c.valid["record_id"]) or \
        sorted(a.test["record_id"]) != sorted(c.test["record_id"])


def test_no_record_dropped_or_duplicated():
    records = _records(_antibody_group("g/big", n_antibodies=40))
    result = build_within_antigen_split(records, 0.2, 0.2, seed=0, min_eval_records=3)
    total = len(result.train) + len(result.valid) + len(result.test)
    assert total == len(records)
    all_record_ids = (
        list(result.train["record_id"]) + list(result.valid["record_id"]) + list(result.test["record_id"])
    )
    assert len(all_record_ids) == len(set(all_record_ids))


def test_raises_when_no_group_is_splittable():
    records = _records(_tiny_group())
    with pytest.raises(ValueError, match="No group had enough antibody-sequence units"):
        build_within_antigen_split(records, 0.2, 0.2, seed=0, min_eval_records=3)


def test_requires_antibody_sequence_columns():
    records = _records(_antibody_group("g/big", n_antibodies=10)).drop(columns=["heavy_chain"])
    with pytest.raises(ValueError, match="missing required column"):
        build_within_antigen_split(records, 0.2, 0.2, seed=0)


def test_leakage_report_only_checks_record_id_overlap():
    records = _records(_antibody_group("g/big", n_antibodies=40))
    result = build_within_antigen_split(records, 0.2, 0.2, seed=0, min_eval_records=3)
    assert list(result.leakage_report["check_name"]) == ["record_id_overlap"]
