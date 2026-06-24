"""Regression tests for shanehsazzadeh2024igdesign converters (batch 2A).

Contract: scripts/prepare/binding/shanehsazzadeh2024igdesign/conversion_contract.yaml
Therapeutic mAb + IgDesign variants vs therapeutic antigens. Raw fitness = K_D in
nM; rank_label = -log10(K_D_nM * 1e-9) -> neg_log10_kd_M, higher_is_better.
Only IL17A was construct-verified; the other antigens must not be silently
replaced.
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
EXPECTED = {
    "Afasevikumab-IL17A_kd": ("IL17A", 13),
    "Bimagrumab-ACVR2B_kd": ("ACVR2B", 24),
    "Eculizumab-C5_kd": ("C5", 34),
    "Osocimab-FXI_kd": ("FXI", 47),
    "Spesolimab-IL36R_kd": ("IL36R", 40),
    "Tezepelumab-TSLP_kd": ("TSLP", 127),
    "Utomilumab-TNFRSF9_kd": ("TNFRSF9", 36),
}


def _run(table_id: str, out_dir: Path) -> pd.DataFrame:
    conv = ROOT / "scripts/prepare/binding/shanehsazzadeh2024igdesign" / table_id / "convert.py"
    src = ROOT / "data/binding" / f"shanehsazzadeh2024igdesign_{table_id}.csv"
    subprocess.run([sys.executable, str(conv), str(src), str(out_dir)],
                   check=True, cwd=ROOT, capture_output=True)
    return pd.read_parquet(out_dir / "records.parquet")


def test_all_tables_schema_counts_and_antigen_keys(tmp_path):
    for table_id, (antigen_key, count) in EXPECTED.items():
        df = _run(table_id, tmp_path / table_id)
        assert list(df.columns) == REQUIRED_COLUMNS, table_id
        assert len(df) == count, table_id
        assert df["record_id"].is_unique, table_id
        r0 = df.iloc[0]
        assert r0["antigen_key"] == antigen_key
        assert r0["antigen_source"] == "retrieved"
        assert r0["metric_name"] == "neg_log10_kd_M"
        assert r0["metric_direction"] == "higher_is_better"
        assert r0["label_kind"] == "experimental"
        assert r0["antibody_type"] == "IgG"
        assert r0["source_file"].endswith(f"shanehsazzadeh2024igdesign_{table_id}.csv")


def test_nm_to_M_conversion(tmp_path):
    df = _run("Afasevikumab-IL17A_kd", tmp_path)
    r0 = df.iloc[0]  # Positive Control, KD = 1.2625 nM
    assert r0["source_row"] == 2
    assert abs(r0["rank_label"] - (-math.log10(1.2625e-9))) < 1e-9  # ~8.8988
    assert r0["group_id"] == "shanehsazzadeh2024igdesign/Afasevikumab-IL17A_kd/IL17A/neg_log10_kd_M/experimental"
    # IL17A construct verified exact (Q16552 mature aa24-155 = 132 aa)
    assert len(r0["antigen_sequence"]) == 132
    assert r0["antigen_sequence"].startswith("GITIPRNP")


def test_unverified_antigen_constructs_not_replaced(tmp_path):
    # The IL36R accession was corrected to IL1RL2 (NOT GITR); guard it is kept,
    # and that constructs are non-empty retrieved sequences (not blanked / not
    # swapped to a guessed full-length).
    il36r = _run("Spesolimab-IL36R_kd", tmp_path / "il36r").iloc[0]
    assert il36r["antigen_key"] == "IL36R"
    assert il36r["antigen_source"] == "retrieved"
    assert isinstance(il36r["antigen_sequence"], str) and len(il36r["antigen_sequence"]) > 50
    c5 = _run("Eculizumab-C5_kd", tmp_path / "c5").iloc[0]
    assert c5["antigen_key"] == "C5"
    assert len(c5["antigen_sequence"]) == 1676  # full-length C5 retained as-is


def test_deterministic(tmp_path):
    a = _run("Bimagrumab-ACVR2B_kd", tmp_path / "a")
    b = _run("Bimagrumab-ACVR2B_kd", tmp_path / "b")
    assert a.equals(b)
