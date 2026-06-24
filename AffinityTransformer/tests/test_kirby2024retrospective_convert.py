"""Regression tests for kirby2024retrospective converters (batch 2C).

Contract: scripts/prepare/binding/kirby2024retrospective/conversion_contract.yaml
Human anti-SARS-CoV-2 (Wuhan RBD) antibodies. The 'Kd [M]' raw header is a
mislabel -- values are nM; the converter converts nM->M before -log10. Binary
table is 0/1 and must stay a separate group from the continuous Kd table.
"""
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WUHAN_RBD_LEN = 223
REQUIRED_COLUMNS = [
    "record_id", "dataset_id", "study_id", "table_id", "source_file", "source_row",
    "antibody_id", "antibody_type", "heavy_chain", "light_chain", "single_chain_sequence",
    "antigen_key", "antigen_name", "antigen_sequence", "antigen_source",
    "assay_name", "assay_type", "metric_name", "metric_value_raw", "metric_value_numeric",
    "metric_unit", "metric_direction", "transform_rule", "rank_label", "label_kind",
    "group_id", "keep_for_training", "drop_reason",
]


def _run(table_id: str, out_dir: Path) -> pd.DataFrame:
    conv = ROOT / "scripts/prepare/binding/kirby2024retrospective" / table_id / "convert.py"
    src = ROOT / "data/binding" / f"kirby2024retrospective_{table_id}.csv"
    subprocess.run([sys.executable, str(conv), str(src), str(out_dir)],
                   check=True, cwd=ROOT, capture_output=True)
    return pd.read_parquet(out_dir / "records.parquet")


def test_kd_treated_as_nM(tmp_path):
    df = _run("ab-SARSCoV2_kd", tmp_path)
    assert list(df.columns) == REQUIRED_COLUMNS
    assert len(df) == 869
    assert df["record_id"].is_unique
    r0 = df.iloc[0]
    assert r0["source_row"] == 2
    # raw 582.33 is nM -> rank_label = -log10(582.33e-9) = 6.2348
    assert abs(r0["rank_label"] - (-math.log10(582.33e-9))) < 1e-6
    assert r0["metric_value_raw"] == "582.33"
    assert r0["metric_name"] == "neg_log10_kd_M"
    assert r0["metric_unit"] == "-log10(KD/M)"
    assert r0["metric_direction"] == "higher_is_better"
    assert r0["label_kind"] == "experimental"
    assert r0["group_id"] == "kirby2024retrospective/ab-SARSCoV2_kd/CoV2_Wuhan_RBD/neg_log10_kd_M/experimental"
    assert r0["antigen_source"] == "retrieved"
    assert len(r0["antigen_sequence"]) == WUHAN_RBD_LEN


def test_binary_semantics(tmp_path):
    df = _run("ab-SARSCoV2_binary_kd", tmp_path)
    assert len(df) == 1407
    r0 = df.iloc[0]
    assert r0["label_kind"] == "binary"
    assert r0["metric_name"] == "binary"
    assert r0["group_id"] == "kirby2024retrospective/ab-SARSCoV2_binary_kd/CoV2_Wuhan_RBD/binary/binary"
    vals = set(pd.to_numeric(df.loc[df["keep_for_training"], "rank_label"]).unique())
    assert vals <= {0.0, 1.0}


def test_binary_and_continuous_distinct(tmp_path):
    kd = _run("ab-SARSCoV2_kd", tmp_path / "kd")
    bn = _run("ab-SARSCoV2_binary_kd", tmp_path / "bn")
    assert set(kd["group_id"]).isdisjoint(set(bn["group_id"]))
    assert set(kd["label_kind"]) == {"experimental"}
    assert set(bn["label_kind"]) == {"binary"}


def test_deterministic(tmp_path):
    a = _run("ab-SARSCoV2_kd", tmp_path / "a")
    b = _run("ab-SARSCoV2_kd", tmp_path / "b")
    assert a.fillna("NA").equals(b.fillna("NA"))
