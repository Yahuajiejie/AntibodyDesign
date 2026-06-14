import pandas as pd
import pytest

from affinity_transformer.record_filter import (
    AntigenAntibodyPair,
    RecordFilterConfig,
    antibody_sequence_hashes,
    build_record_filter_config,
    filter_records,
)


def test_filter_records_includes_dataset_and_antigen(toy_records):
    config = RecordFilterConfig(
        include_dataset_ids=("studyA/tableA", "studyB/tableB"),
        include_antigen_keys=("agB",),
    )

    result = filter_records(toy_records, config)

    assert result["dataset_id"].unique().tolist() == ["studyB/tableB"]
    assert result["antigen_key"].unique().tolist() == ["agB"]


def test_filter_records_requires_antigen_sequence(toy_records):
    result = filter_records(
        toy_records,
        RecordFilterConfig(include_dataset_ids=("studyC/tableC",), require_antigen_sequence=True),
    )

    assert result.empty


def test_filter_records_applies_group_label_threshold(toy_records):
    result = filter_records(
        toy_records,
        RecordFilterConfig(
            min_trainable_records_per_group=2,
            min_unique_labels_per_group=2,
        ),
    )

    assert "studyE/tableE/agE/neg_log10_kd_M/experimental" not in set(result["group_id"])
    assert "studyA/tableA/agA/neg_log10_kd_M/experimental" in set(result["group_id"])


def test_filter_records_selects_antigen_antibody_pair(toy_records):
    records = toy_records.copy()
    records.loc[records["record_id"].eq("studyA/tableA/2"), "antibody_id"] = "ab1"
    records.loc[records["record_id"].eq("studyA/tableA/3"), "antibody_id"] = "ab2"

    result = filter_records(
        records,
        RecordFilterConfig(
            include_antigen_antibody_pairs=(
                AntigenAntibodyPair(antigen_key="agA", antibody_id="ab2"),
            )
        ),
    )

    assert result["record_id"].tolist() == ["studyA/tableA/3"]


def test_filter_records_selects_antibody_sequence_hash(toy_records):
    hashes = antibody_sequence_hashes(toy_records)
    target_hash = hashes.loc[toy_records["record_id"].eq("studyA/tableA/2")].iloc[0]

    result = filter_records(
        toy_records,
        RecordFilterConfig(include_antibody_sequence_hashes=(target_hash,)),
    )

    assert "studyA/tableA/2" in set(result["record_id"])
    assert set(antibody_sequence_hashes(result)) == {target_hash}


def test_build_record_filter_config_rejects_unknown_field():
    with pytest.raises(ValueError, match="unknown"):
        build_record_filter_config({"include_dataset_ids": ["x"], "surprise": ["y"]})


def test_build_record_filter_config_accepts_pair_mapping():
    config = build_record_filter_config({
        "include_antigen_antibody_pairs": [
            {"antigen_key": "agA", "antibody_id": "ab1"},
        ],
    })

    assert config.include_antigen_antibody_pairs == (
        AntigenAntibodyPair(antigen_key="agA", antibody_id="ab1"),
    )
