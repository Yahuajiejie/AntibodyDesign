"""Tests for the antigen-context-local within-antigen antibody split.

The intended protocol is:

records
  -> pool similar/exact antigens into an antigen context
  -> within each antigen context
  -> hold out antibody units/components into train/valid/test

So ``group_id`` is not the protocol boundary.  Different groups that share the
same ``antigen_cluster_id`` or ``antigen_sequence_key`` must be pooled before
antibody holdout happens.
"""

from __future__ import annotations

import pandas as pd
import pytest

from affinity_transformer.splits import build_within_antigen_split


_BASE_ROW = {
    "dataset_id": "studyA/tableA",
    "keep_for_training": True,
    "antigen_key": "legacy/agA",
    "single_chain_sequence": None,
}


def _row(
    record_id: str,
    group_id: str,
    antibody_index: int,
    *,
    antigen_sequence_key: str = "ag-seq/A",
    antigen_cluster_id: str = "ag-cluster/A",
    antibody_cluster_id: str | None = None,
    measurement_family_id: str | None = None,
    interaction_key: str | None = None,
    label: float | None = None,
) -> dict:
    antibody_cluster_id = antibody_cluster_id or f"ab-cluster/{antibody_index}"
    antibody_sequence_key = antibody_cluster_id.replace("ab-cluster/", "ab-seq/")
    return {
        **_BASE_ROW,
        "record_id": record_id,
        "group_id": group_id,
        "heavy_chain": f"HEAVY{antibody_index:03d}QVQLVQSGAEVKKPGASVKVSCKAS",
        "light_chain": f"LIGHT{antibody_index:03d}DIQMTQSPSSLSASVGDRVTITC",
        "rank_label": float(antibody_index) if label is None else label,
        "antigen_sequence_key": antigen_sequence_key,
        "antigen_cluster_id": antigen_cluster_id,
        "antibody_sequence_key": antibody_sequence_key,
        "antibody_cluster_id": antibody_cluster_id,
        "measurement_family_id": measurement_family_id or f"mf/{record_id}",
        "interaction_key": interaction_key or f"int/{antigen_cluster_id}/{antibody_cluster_id}",
    }


def _antibody_group(
    group_id: str,
    n_antibodies: int,
    *,
    offset: int = 0,
    antigen_sequence_key: str = "ag-seq/A",
    antigen_cluster_id: str = "ag-cluster/A",
) -> list[dict]:
    rows = []
    for i in range(offset, offset + n_antibodies):
        rows.append(_row(
            record_id=f"{group_id}/{i}",
            group_id=group_id,
            antibody_index=i,
            antigen_sequence_key=antigen_sequence_key,
            antigen_cluster_id=antigen_cluster_id,
        ))
    return rows


def _tiny_context(group_id: str = "g/tiny") -> list[dict]:
    return [
        _row(f"{group_id}/0", group_id, 0, antigen_cluster_id="ag-cluster/tiny"),
        _row(f"{group_id}/1", group_id, 0, antigen_cluster_id="ag-cluster/tiny"),
        _row(f"{group_id}/2", group_id, 1, antigen_cluster_id="ag-cluster/tiny"),
    ]


def _records(*groups: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for group in groups:
        rows.extend(group)
    return pd.DataFrame(rows)


def _context_antibody_units(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(frame["antigen_cluster_id"], frame["antibody_cluster_id"]))


def _record_split(result, record_id: str) -> str:
    for split_name in ("train", "valid", "test"):
        if record_id in set(getattr(result, split_name)["record_id"]):
            return split_name
    raise AssertionError(f"record_id not found in any split: {record_id}")


def test_antibody_units_never_cross_within_antigen_context():
    records = _records(_antibody_group("g/big", n_antibodies=40))
    result = build_within_antigen_split(
        records, valid_fraction=0.2, test_fraction=0.2, seed=0, min_eval_records=3
    )

    train_units = _context_antibody_units(result.train)
    valid_units = _context_antibody_units(result.valid)
    test_units = _context_antibody_units(result.test)
    assert not (train_units & valid_units)
    assert not (train_units & test_units)
    assert not (valid_units & test_units)
    assert valid_units and test_units


def test_group_id_can_overlap_because_same_context_is_split_inside_group():
    records = _records(_antibody_group("g/big", n_antibodies=40))
    result = build_within_antigen_split(
        records, valid_fraction=0.2, test_fraction=0.2, seed=0, min_eval_records=3
    )

    assert "g/big" in set(result.train["group_id"])
    assert "g/big" in set(result.valid["group_id"])
    assert "g/big" in set(result.test["group_id"])


def test_same_antigen_cluster_across_groups_is_pooled_before_antibody_holdout():
    records = _records(
        _antibody_group(
            "g/A", n_antibodies=40, offset=0,
            antigen_sequence_key="ag-seq/A-construct-1",
            antigen_cluster_id="ag-cluster/shared",
        ),
        _antibody_group(
            "g/B", n_antibodies=40, offset=0,
            antigen_sequence_key="ag-seq/A-construct-2",
            antigen_cluster_id="ag-cluster/shared",
        ),
    )
    result = build_within_antigen_split(
        records, valid_fraction=0.2, test_fraction=0.2, seed=0, min_eval_records=3
    )

    for antibody_cluster_id in sorted(records["antibody_cluster_id"].unique()):
        placements = {
            split_name
            for split_name in ("train", "valid", "test")
            if antibody_cluster_id in set(getattr(result, split_name)["antibody_cluster_id"])
        }
        assert len(placements) == 1


def test_antigen_sequence_key_is_used_when_cluster_id_is_absent():
    records = _records(
        _antibody_group("g/A", n_antibodies=30, offset=0, antigen_sequence_key="ag-seq/shared"),
        _antibody_group("g/B", n_antibodies=30, offset=0, antigen_sequence_key="ag-seq/shared"),
    ).drop(columns=["antigen_cluster_id"])
    result = build_within_antigen_split(
        records, valid_fraction=0.2, test_fraction=0.2, seed=0, min_eval_records=3
    )

    for antibody_cluster_id in sorted(records["antibody_cluster_id"].unique()):
        placements = {
            split_name
            for split_name in ("train", "valid", "test")
            if antibody_cluster_id in set(getattr(result, split_name)["antibody_cluster_id"])
        }
        assert len(placements) == 1


def test_measurement_family_links_records_inside_antigen_context():
    rows = _antibody_group("g/big", n_antibodies=40)
    rows[0]["measurement_family_id"] = "mf/shared"
    rows[1]["measurement_family_id"] = "mf/shared"
    records = _records(rows)

    result = build_within_antigen_split(
        records, valid_fraction=0.2, test_fraction=0.2, seed=0, min_eval_records=3
    )

    assert _record_split(result, "g/big/0") == _record_split(result, "g/big/1")


def test_interaction_key_links_records_inside_antigen_context():
    rows = _antibody_group("g/big", n_antibodies=40)
    rows[2]["interaction_key"] = "int/shared"
    rows[3]["interaction_key"] = "int/shared"
    records = _records(rows)

    result = build_within_antigen_split(
        records, valid_fraction=0.2, test_fraction=0.2, seed=0, min_eval_records=3
    )

    assert _record_split(result, "g/big/2") == _record_split(result, "g/big/3")


def test_tiny_antigen_context_is_pinned_entirely_to_train():
    records = _records(_antibody_group("g/big", n_antibodies=40), _tiny_context())
    result = build_within_antigen_split(
        records, valid_fraction=0.2, test_fraction=0.2, seed=0, min_eval_records=3
    )

    assert "ag-cluster/tiny" not in set(result.valid["antigen_cluster_id"])
    assert "ag-cluster/tiny" not in set(result.test["antigen_cluster_id"])
    assert set(records.loc[records["antigen_cluster_id"] == "ag-cluster/tiny", "record_id"]) <= set(
        result.train["record_id"]
    )

    pinned = result.pinned_groups
    tiny_row = pinned.loc[pinned["group_id"] == "g/tiny"].iloc[0]
    assert tiny_row["n_antibody_units"] == 2
    assert "fewer than 3" in tiny_row["reason"]


def test_min_eval_records_pins_context_too_small_to_holdout_reliably():
    records = _records(
        _antibody_group("g/small9", n_antibodies=9, antigen_cluster_id="ag-cluster/small"),
        _antibody_group("g/big", n_antibodies=60, antigen_cluster_id="ag-cluster/big"),
    )
    result = build_within_antigen_split(
        records, valid_fraction=0.1, test_fraction=0.1, seed=0, min_eval_records=5
    )

    assert "ag-cluster/small" not in set(result.valid["antigen_cluster_id"])
    assert "ag-cluster/small" not in set(result.test["antigen_cluster_id"])
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


def test_raises_when_no_antigen_context_is_splittable():
    records = _records(_tiny_context())
    with pytest.raises(ValueError, match="No antigen context had enough antibody components"):
        build_within_antigen_split(records, 0.2, 0.2, seed=0, min_eval_records=3)


def test_requires_antigen_context_column():
    records = _records(_antibody_group("g/big", n_antibodies=10)).drop(
        columns=["antigen_cluster_id", "antigen_sequence_key"]
    )
    with pytest.raises(ValueError, match="antigen context column"):
        build_within_antigen_split(records, 0.2, 0.2, seed=0)


def test_requires_antibody_sequence_columns_when_identity_columns_are_absent():
    records = _records(_antibody_group("g/big", n_antibodies=10)).drop(
        columns=["antibody_cluster_id", "antibody_sequence_key", "heavy_chain"]
    )
    with pytest.raises(ValueError, match="antibody sequence fallback"):
        build_within_antigen_split(records, 0.2, 0.2, seed=0)


def test_leakage_report_checks_context_scoped_antibody_and_duplicate_links():
    records = _records(_antibody_group("g/big", n_antibodies=40))
    result = build_within_antigen_split(records, 0.2, 0.2, seed=0, min_eval_records=3)

    assert list(result.leakage_report["check_name"]) == [
        "record_id_overlap",
        "within_antigen_antibody_unit_overlap",
        "within_antigen_component_overlap",
        "valid_antigen_context_seen_in_train",
        "test_antigen_context_seen_in_train",
        "within_antigen_measurement_family_id_overlap",
        "within_antigen_interaction_key_overlap",
    ]
    assert (result.leakage_report["status"] == "PASS").all()
