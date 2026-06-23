"""Regression tests for zimmerman2020antibody converter (batch 2E).

Contract: scripts/prepare/binding/zimmerman2020antibody/conversion_contract.yaml
4-4-20 anti-fluorescein variants. VERIFIED CORRECTION: the raw 'heavy' column
holds the VL and the raw 'light' column holds the VH, so the converter must swap
them. rank_label = fitness = -log10(K_D[M]), higher_is_better, experimental.
"""
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/binding/zimmerman2020antibody_4420_kd.csv"


def _run(out_dir: Path) -> pd.DataFrame:
    conv = ROOT / "scripts/prepare/binding/zimmerman2020antibody/4420_kd/convert.py"
    subprocess.run([sys.executable, str(conv), str(RAW), str(out_dir)],
                   check=True, cwd=ROOT, capture_output=True)
    return pd.read_parquet(out_dir / "records.parquet")


def test_heavy_light_swap_is_corrected(tmp_path):
    df = _run(tmp_path)
    raw = pd.read_csv(RAW, dtype=str)
    # Raw mislabels chains: 'heavy' col = VL (DVVMTQ...), 'light' col = VH (EVKL...).
    assert raw["heavy"].iloc[0].startswith("DVVMTQ")
    assert raw["light"].iloc[0].startswith("EVKL")
    r0 = df.iloc[0]
    # Output must carry the TRUE chain types (swap applied):
    assert r0["heavy_chain"].startswith("EVKL"), "heavy_chain must be the VH"
    assert r0["light_chain"].startswith("DVVMTQ"), "light_chain must be the VL"
    # explicit: output heavy_chain == raw 'light' column; light_chain == raw 'heavy'
    assert r0["heavy_chain"] == raw["light"].iloc[0]
    assert r0["light_chain"] == raw["heavy"].iloc[0]


def test_schema_counts_label_and_antigen(tmp_path):
    df = _run(tmp_path)
    assert len(df) == 21
    assert df["record_id"].is_unique
    r0 = df.iloc[0]
    assert r0["source_row"] == 2
    # rank_label = fitness column directly (already -log10 K_D); row0 = 3.823908741
    assert abs(r0["rank_label"] - 3.823908741) < 1e-6
    assert r0["metric_name"] == "neg_log10_kd_M"
    assert r0["metric_direction"] == "higher_is_better"
    assert r0["label_kind"] == "experimental"
    assert r0["group_id"] == "zimmerman2020antibody/4420_kd/Fluorescein/neg_log10_kd_M/experimental"
    # fluorescein hapten: no antigen sequence
    assert r0["antigen_source"] == "missing"
    assert r0["antigen_sequence"] is None or (isinstance(r0["antigen_sequence"], float) and math.isnan(r0["antigen_sequence"]))


def test_deterministic(tmp_path):
    a = _run(tmp_path / "a")
    b = _run(tmp_path / "b")
    assert a.equals(b)
