"""Regression tests for adams2017measuring converters (batch 2E).

Contract: scripts/prepare/binding/adams2017measuring/conversion_contract.yaml
4-4-20 anti-fluorescein scFv vs fluorescein (hapten). rank_label = -log10(K_D[M]),
higher_is_better, experimental. Source: eLife 23156.
"""
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = [
    "record_id", "dataset_id", "study_id", "table_id", "source_file", "source_row",
    "antibody_id", "antibody_type", "heavy_chain", "light_chain", "single_chain_sequence",
    "antigen_key", "antigen_name", "antigen_sequence", "antigen_source",
    "assay_name", "assay_type", "metric_name", "metric_value_raw", "metric_value_numeric",
    "metric_unit", "metric_direction", "transform_rule", "rank_label", "label_kind",
    "group_id", "keep_for_training", "drop_reason",
]


def _run(table_id: str, raw_name: str, out_dir: Path) -> pd.DataFrame:
    conv = ROOT / "scripts/prepare/binding/adams2017measuring" / table_id / "convert.py"
    src = ROOT / "data/binding" / raw_name
    subprocess.run([sys.executable, str(conv), str(src), str(out_dir)],
                   check=True, cwd=ROOT, capture_output=True)
    return pd.read_parquet(out_dir / "records.parquet")


def test_flow_schema_counts_and_label(tmp_path):
    df = _run("4420-fluorescein_kd-flow",
              "adams2017measuring_4420-fluorescein_kd-flow.csv", tmp_path)
    assert list(df.columns) == REQUIRED_COLUMNS
    assert len(df) == 15
    assert df["record_id"].is_unique
    # row 0 (source_row 2): fitness=2.818e-09 -> -log10 = 8.5501
    r0 = df.iloc[0]
    assert r0["source_row"] == 2
    assert abs(r0["rank_label"] - (-math.log10(2.818e-09))) < 1e-9
    assert r0["metric_name"] == "neg_log10_kd_M"
    assert r0["metric_unit"] == "-log10(KD/M)"
    assert r0["metric_direction"] == "higher_is_better"
    assert r0["label_kind"] == "experimental"
    assert r0["group_id"] == "adams2017measuring/4420-fluorescein_kd-flow/Fluorescein/neg_log10_kd_M/experimental"
    # 4-4-20 chain orientation: heavy = VH (EVKL...), antigen is a hapten (no seq)
    assert r0["heavy_chain"].startswith("EVKL")
    assert r0["light_chain"].startswith("DVVMTQ")
    assert r0["antigen_source"] == "missing"
    assert r0["antigen_sequence"] is None or (isinstance(r0["antigen_sequence"], float) and math.isnan(r0["antigen_sequence"]))


def test_titeseq_counts_and_transform(tmp_path):
    df = _run("4420-fluorescein_kd-titeseq",
              "adams2017measuring_4420-fluorescein_kd-titeseq.csv", tmp_path)
    assert list(df.columns) == REQUIRED_COLUMNS
    assert len(df) == 11052
    assert df["record_id"].is_unique
    r0 = df.iloc[0]
    assert abs(r0["rank_label"] - (-math.log10(2.399e-09))) < 1e-9  # fitness row0
    assert r0["antigen_source"] == "missing"
    assert (df["metric_direction"] == "higher_is_better").all()


def test_flow_deterministic(tmp_path):
    a = _run("4420-fluorescein_kd-flow",
             "adams2017measuring_4420-fluorescein_kd-flow.csv", tmp_path / "a")
    b = _run("4420-fluorescein_kd-flow",
             "adams2017measuring_4420-fluorescein_kd-flow.csv", tmp_path / "b")
    assert a.equals(b)
