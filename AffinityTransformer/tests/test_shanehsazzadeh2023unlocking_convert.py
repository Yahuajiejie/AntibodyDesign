"""Regression tests for shanehsazzadeh2023unlocking converters (batch 2A).

Contract: scripts/prepare/binding/shanehsazzadeh2023unlocking/conversion_contract.yaml
trastuzumab vs HER2. Kd tables use a PRE-COMPUTED -log10(K_D[M]) column (must NOT
be transformed again). ADCC table: EC50 in pM -> -log10(EC50*1e-12) -> M.
assay_type for ADCC is left as 'binding' per the approved contract (documented
limitation).
"""
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"adcc_ec50": 13, "kd_hher2_fab": 13, "kd_hher2_mab": 13, "zerokd_trastuzumab": 422}


def _run(table_id: str, out_dir: Path) -> pd.DataFrame:
    conv = ROOT / "scripts/prepare/binding/shanehsazzadeh2023unlocking" / table_id / "convert.py"
    src = ROOT / "data/binding" / f"shanehsazzadeh2023unlocking_{table_id}.csv"
    subprocess.run([sys.executable, str(conv), str(src), str(out_dir)],
                   check=True, cwd=ROOT, capture_output=True)
    return pd.read_parquet(out_dir / "records.parquet")


def test_counts_and_antigen(tmp_path):
    for table_id, count in EXPECTED.items():
        df = _run(table_id, tmp_path / table_id)
        assert len(df) == count, table_id
        assert df["record_id"].is_unique, table_id
        assert df.iloc[0]["antigen_key"] == "hHER2"
        assert df.iloc[0]["antigen_source"] == "retrieved"


def test_precomputed_kd_not_transformed_twice(tmp_path):
    df = _run("kd_hher2_fab", tmp_path)
    r0 = df.iloc[0]  # fitness column already = -log10(K_D[M]) = 8.860120914
    assert abs(r0["rank_label"] - 8.860120914) < 1e-9
    assert float(r0["metric_value_raw"]) == 8.860120914   # raw preserved, used directly
    assert r0["metric_name"] == "neg_log10_kd_M"
    assert r0["metric_unit"] == "-log10(KD/M)"
    assert r0["antibody_type"] == "Fab"
    assert r0["metric_direction"] == "higher_is_better"


def test_adcc_pM_to_M_conversion(tmp_path):
    df = _run("adcc_ec50", tmp_path)
    r0 = df.iloc[0]  # ADCC EC50 = 56.97 pM
    assert abs(r0["rank_label"] - (-math.log10(56.97e-12))) < 1e-9  # ~10.2444
    assert r0["metric_name"] == "neg_log10_ec50_M"
    assert r0["metric_unit"] == "-log10(EC50/M)"
    assert r0["label_kind"] == "experimental"
    # contract leaves assay_type unchanged ('binding') -- documented limitation
    assert r0["assay_type"] == "binding"


def test_deterministic(tmp_path):
    a = _run("zerokd_trastuzumab", tmp_path / "a")
    b = _run("zerokd_trastuzumab", tmp_path / "b")
    assert a.equals(b)
