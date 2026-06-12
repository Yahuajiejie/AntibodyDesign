"""Shared pytest fixtures for affinity_transformer tests."""

from __future__ import annotations

import math

import pandas as pd
import pytest


def _row(**overrides: object) -> dict[str, object]:
    """Build one standard-table row (spec §3) with sensible defaults.

    Args:
        **overrides: Fields to override on top of the defaults.

    Returns:
        A dict covering every column in `affinity_transformer.dataset.REQUIRED_COLUMNS`.
    """
    base: dict[str, object] = dict(
        record_id="",
        dataset_id="study/table",
        study_id="study",
        table_id="table",
        source_file="data/binding/study_table.csv",
        source_row=2,
        antibody_id=None,
        antibody_type="Fv",
        heavy_chain="QVQLVQSGAEVKKPGASVKVSCKAS",
        light_chain="DIQMTQSPSSLSASVGDRVTITC",
        single_chain_sequence=None,
        antigen_key="agA",
        antigen_name="Antigen A",
        antigen_sequence="MKTAYIAKQRQISFVKSHFSRQLE",
        antigen_source="provided",
        assay_name="SPR",
        assay_type="binding",
        metric_name="neg_log10_kd_M",
        metric_value_raw="8.0",
        metric_value_numeric=8.0,
        metric_unit="-log10(KD/M)",
        metric_direction="higher_is_better",
        transform_rule="rank_label = neg_log10_kd_M",
        rank_label=1.0,
        label_kind="experimental",
        group_id="study/table/agA/neg_log10_kd_M/experimental",
        keep_for_training=True,
        drop_reason=None,
    )
    base.update(overrides)
    return base


@pytest.fixture()
def toy_records() -> pd.DataFrame:
    """Toy processed table covering the spec §7.3 group types.

    Groups:
        - Fv group (3 records, distinct labels -> 3 candidate pairs).
        - VHH group (2 records, distinct labels -> 1 candidate pair).
        - Missing-antigen group (2 records, antigen_sequence=None -> 1
          candidate pair).
        - Binary group (4 records: 2 positive + 2 negative -> 4 cross-class
          candidate pairs, 0 same-class pairs).
        - Single-label group (2 records sharing one rank_label -> 0
          candidate pairs).

    Plus one `keep_for_training=False` record and one record with a
    non-finite `rank_label`, both of which `filter_trainable_records` must
    remove.
    """
    rows: list[dict[str, object]] = []

    fv_group = "studyA/tableA/agA/neg_log10_kd_M/experimental"
    for i, label in enumerate((1.0, 2.0, 3.0), start=2):
        rows.append(_row(
            record_id=f"studyA/tableA/{i}",
            dataset_id="studyA/tableA", study_id="studyA", table_id="tableA",
            source_row=i, antibody_type="Fv", group_id=fv_group, rank_label=label,
        ))

    vhh_group = "studyB/tableB/agB/neg_log10_kd_M/experimental"
    for i, label in enumerate((0.5, 1.5), start=2):
        rows.append(_row(
            record_id=f"studyB/tableB/{i}",
            dataset_id="studyB/tableB", study_id="studyB", table_id="tableB",
            source_row=i, antibody_type="VHH",
            heavy_chain="QVKLEESGGGLVQAGGSLRLSCAAS", light_chain=None,
            antigen_key="agB", antigen_name="Antigen B",
            antigen_sequence="MSTNPKPQRKTKRNTNRRPQDVKFPGGGQ",
            group_id=vhh_group, rank_label=label,
        ))

    missing_ag_group = "studyC/tableC/unknown_antigen/neg_log10_kd_M/experimental"
    for i, label in enumerate((0.1, 0.9), start=2):
        rows.append(_row(
            record_id=f"studyC/tableC/{i}",
            dataset_id="studyC/tableC", study_id="studyC", table_id="tableC",
            source_row=i, antibody_type="Fv",
            antigen_key="unknown_antigen", antigen_name=None,
            antigen_sequence=None, antigen_source="missing",
            group_id=missing_ag_group, rank_label=label,
        ))

    binary_group = "studyD/tableD/agD/bind/binary"
    for i, label in enumerate((1.0, 1.0, 0.0, 0.0), start=2):
        rows.append(_row(
            record_id=f"studyD/tableD/{i}",
            dataset_id="studyD/tableD", study_id="studyD", table_id="tableD",
            source_row=i, antibody_type="Fv",
            antigen_key="agD", antigen_name="Antigen D",
            assay_name="ELISA", assay_type="binding",
            metric_name="bind", metric_value_raw=str(int(label)),
            metric_value_numeric=label, metric_unit=None,
            metric_direction="higher_is_better",
            transform_rule="rank_label = bind (0/1)",
            rank_label=label, label_kind="binary",
            group_id=binary_group,
        ))

    single_group = "studyE/tableE/agE/neg_log10_kd_M/experimental"
    for i in (2, 3):
        rows.append(_row(
            record_id=f"studyE/tableE/{i}",
            dataset_id="studyE/tableE", study_id="studyE", table_id="tableE",
            source_row=i, antibody_type="Fv",
            antigen_key="agE", antigen_name="Antigen E",
            group_id=single_group, rank_label=5.0,
        ))

    rows.append(_row(
        record_id="studyF/tableF/2",
        dataset_id="studyF/tableF", study_id="studyF", table_id="tableF",
        source_row=2, group_id="studyF/tableF/agF/neg_log10_kd_M/experimental",
        rank_label=2.0, keep_for_training=False, drop_reason="missing_or_invalid_heavy_chain",
    ))
    rows.append(_row(
        record_id="studyG/tableG/2",
        dataset_id="studyG/tableG", study_id="studyG", table_id="tableG",
        source_row=2, group_id="studyG/tableG/agG/neg_log10_kd_M/experimental",
        rank_label=math.nan, keep_for_training=True,
    ))

    return pd.DataFrame(rows)


"""
|  行 | record_id       | dataset_id    | antibody_type | antigen_key     | antigen_name | antigen_sequence              | metric_name    | label_kind   | rank_label | group_id                                                  | keep_for_training | drop_reason                    |
| -: | --------------- | ------------- | ------------- | --------------- | ------------ | ----------------------------- | -------------- | ------------ | ---------: | --------------------------------------------------------- | ----------------- | ------------------------------ |
|  1 | studyA/tableA/2 | studyA/tableA | Fv            | agA             | Antigen A    | MKTAYIAKQRQISFVKSHFSRQLE      | neg_log10_kd_M | experimental |        1.0 | studyA/tableA/agA/neg_log10_kd_M/experimental             | True              | None                           |
|  2 | studyA/tableA/3 | studyA/tableA | Fv            | agA             | Antigen A    | MKTAYIAKQRQISFVKSHFSRQLE      | neg_log10_kd_M | experimental |        2.0 | studyA/tableA/agA/neg_log10_kd_M/experimental             | True              | None                           |
|  3 | studyA/tableA/4 | studyA/tableA | Fv            | agA             | Antigen A    | MKTAYIAKQRQISFVKSHFSRQLE      | neg_log10_kd_M | experimental |        3.0 | studyA/tableA/agA/neg_log10_kd_M/experimental             | True              | None                           |
|  4 | studyB/tableB/2 | studyB/tableB | VHH           | agB             | Antigen B    | MSTNPKPQRKTKRNTNRRPQDVKFPGGGQ | neg_log10_kd_M | experimental |        0.5 | studyB/tableB/agB/neg_log10_kd_M/experimental             | True              | None                           |
|  5 | studyB/tableB/3 | studyB/tableB | VHH           | agB             | Antigen B    | MSTNPKPQRKTKRNTNRRPQDVKFPGGGQ | neg_log10_kd_M | experimental |        1.5 | studyB/tableB/agB/neg_log10_kd_M/experimental             | True              | None                           |
|  6 | studyC/tableC/2 | studyC/tableC | Fv            | unknown_antigen | None         | None                          | neg_log10_kd_M | experimental |        0.1 | studyC/tableC/unknown_antigen/neg_log10_kd_M/experimental | True              | None                           |
|  7 | studyC/tableC/3 | studyC/tableC | Fv            | unknown_antigen | None         | None                          | neg_log10_kd_M | experimental |        0.9 | studyC/tableC/unknown_antigen/neg_log10_kd_M/experimental | True              | None                           |
|  8 | studyD/tableD/2 | studyD/tableD | Fv            | agD             | Antigen D    | MKTAYIAKQRQISFVKSHFSRQLE      | bind           | binary       |        1.0 | studyD/tableD/agD/bind/binary                             | True              | None                           |
|  9 | studyD/tableD/3 | studyD/tableD | Fv            | agD             | Antigen D    | MKTAYIAKQRQISFVKSHFSRQLE      | bind           | binary       |        1.0 | studyD/tableD/agD/bind/binary                             | True              | None                           |
| 10 | studyD/tableD/4 | studyD/tableD | Fv            | agD             | Antigen D    | MKTAYIAKQRQISFVKSHFSRQLE      | bind           | binary       |        0.0 | studyD/tableD/agD/bind/binary                             | True              | None                           |
| 11 | studyD/tableD/5 | studyD/tableD | Fv            | agD             | Antigen D    | MKTAYIAKQRQISFVKSHFSRQLE      | bind           | binary       |        0.0 | studyD/tableD/agD/bind/binary                             | True              | None                           |
| 12 | studyE/tableE/2 | studyE/tableE | Fv            | agE             | Antigen E    | MKTAYIAKQRQISFVKSHFSRQLE      | neg_log10_kd_M | experimental |        5.0 | studyE/tableE/agE/neg_log10_kd_M/experimental             | True              | None                           |
| 13 | studyE/tableE/3 | studyE/tableE | Fv            | agE             | Antigen E    | MKTAYIAKQRQISFVKSHFSRQLE      | neg_log10_kd_M | experimental |        5.0 | studyE/tableE/agE/neg_log10_kd_M/experimental             | True              | None                           |
| 14 | studyF/tableF/2 | studyF/tableF | Fv            | agA             | Antigen A    | MKTAYIAKQRQISFVKSHFSRQLE      | neg_log10_kd_M | experimental |        2.0 | studyF/tableF/agF/neg_log10_kd_M/experimental             | False             | missing_or_invalid_heavy_chain |
| 15 | studyG/tableG/2 | studyG/tableG | Fv            | agA             | Antigen A    | MKTAYIAKQRQISFVKSHFSRQLE      | neg_log10_kd_M | experimental |        NaN | studyG/tableG/agG/neg_log10_kd_M/experimental             | True              | None                           |

"""