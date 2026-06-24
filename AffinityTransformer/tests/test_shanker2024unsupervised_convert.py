"""Regression tests for shanker2024unsupervised converters (batch 2C).

Contract: scripts/prepare/binding/shanker2024unsupervised/conversion_contract.yaml
LY-CoV1404 / SA58 vs SARS-CoV-2 variants. IC50 (ng/µL) and Kd (M), both -log10,
higher_is_better, experimental. Approved sentinel policy: non-physical Kd values
(SA58-BQ.1.1_Kd: 0; SA58-XBB.1.5_Kd: 7.64e19) are preserved but untrainable with
drop_reason="nonphysical_kd_sentinel" and no -log10 applied.
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
    conv = ROOT / "scripts/prepare/binding/shanker2024unsupervised" / table_id / "convert.py"
    src = ROOT / "data/binding" / f"shanker2024unsupervised_{table_id}.csv"
    subprocess.run([sys.executable, str(conv), str(src), str(out_dir)],
                   check=True, cwd=ROOT, capture_output=True)
    return pd.read_parquet(out_dir / "records.parquet")


def test_ic50_wuhan_schema_transform_antigen(tmp_path):
    df = _run("Ly1404_Wuhan_IC50", tmp_path)
    assert list(df.columns) == REQUIRED_COLUMNS
    assert len(df) == 32
    assert df["record_id"].is_unique
    r0 = df.iloc[0]
    assert r0["source_row"] == 2
    # WT IC50 = 2.7 ng/µL -> -log10(2.7)
    assert abs(r0["rank_label"] - (-math.log10(2.7))) < 1e-9
    assert r0["metric_name"] == "neg_log10_ic50"
    assert r0["metric_unit"] == "-log10(IC50 ng/µL)"
    assert r0["metric_direction"] == "higher_is_better"
    assert r0["label_kind"] == "experimental"
    assert r0["group_id"] == "shanker2024unsupervised/Ly1404_Wuhan_IC50/CoV2_Wuhan_RBD/neg_log10_ic50/experimental"
    # Wuhan RBD retrieved
    assert r0["antigen_source"] == "retrieved"
    assert len(r0["antigen_sequence"]) == WUHAN_RBD_LEN


def test_ic50_variant_antigen_kept_missing(tmp_path):
    df = _run("Ly1404-BQ.1.1_IC50", tmp_path)
    assert len(df) == 50
    r0 = df.iloc[0]
    assert r0["antigen_source"] == "missing"
    assert r0["antigen_sequence"] is None or pd.isna(r0["antigen_sequence"])
    assert abs(r0["rank_label"] - (-math.log10(110.0))) < 1e-9  # WT 110 ng/µL


def _check_sentinel(df, sentinel_row, raw_str):
    s = df[df["source_row"] == sentinel_row]
    assert len(s) == 1, "sentinel record must be preserved (not deleted)"
    s = s.iloc[0]
    assert s["keep_for_training"] == False           # noqa: E712
    assert s["drop_reason"] == "nonphysical_kd_sentinel"
    assert not math.isfinite(s["rank_label"])         # no finite transformed label
    assert s["metric_value_raw"] == raw_str           # original raw value preserved
    assert float(s["metric_value_numeric"]) == float(raw_str)


def test_sa58_bq11_kd_sentinel_zero(tmp_path):
    df = _run("SA58-BQ.1.1_Kd", tmp_path)
    assert len(df) == 7                                # row preserved, not dropped
    _check_sentinel(df, sentinel_row=8, raw_str="0.0")
    # a valid row in the SAME table remains trainable with a finite label
    valid = df[df["source_row"] == 2].iloc[0]          # T55N, 4.59e-9
    assert valid["keep_for_training"] == True          # noqa: E712
    assert abs(valid["rank_label"] - (-math.log10(4.59e-09))) < 1e-9
    assert valid["metric_name"] == "neg_log10_kd_M"
    assert valid["metric_unit"] == "-log10(KD/M)"


def test_sa58_xbb15_kd_sentinel_large(tmp_path):
    df = _run("SA58-XBB.1.5_Kd", tmp_path)
    assert len(df) == 30
    _check_sentinel(df, sentinel_row=22, raw_str="7.64e+19")
    valid = df[df["source_row"] == 2].iloc[0]          # WT, 2.38e-6
    assert valid["keep_for_training"] == True          # noqa: E712
    assert abs(valid["rank_label"] - (-math.log10(2.38e-06))) < 1e-9


def test_valid_kd_table_has_no_sentinels(tmp_path):
    df = _run("Ly1404-BQ.1.1_Kd", tmp_path)
    assert len(df) == 36
    assert "nonphysical_kd_sentinel" not in set(df["drop_reason"].dropna())
    r0 = df.iloc[0]
    assert abs(r0["rank_label"] - (-math.log10(1.7e-06))) < 1e-9  # WT 1.7e-6


def test_deterministic(tmp_path):
    a = _run("SA58-XBB.1.5_Kd", tmp_path / "a")
    b = _run("SA58-XBB.1.5_Kd", tmp_path / "b")
    # NaN-safe equality
    assert a.fillna("NA").equals(b.fillna("NA"))
