"""Regression tests for garbinski2023 converter (batch 2E).

Contract: scripts/prepare/binding/garbinski2023/conversion_contract.yaml
Internal GSK dataset, no publication. Antigen identity is proprietary/undisclosed
and MUST stay explicit (unknown_antigen / missing), not invented.
rank_label = fitness = -log10(K_D[M]), higher_is_better, experimental.
"""
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/binding/garbinski2023_kd.csv"


def _run(out_dir: Path) -> pd.DataFrame:
    conv = ROOT / "scripts/prepare/binding/garbinski2023/kd/convert.py"
    subprocess.run([sys.executable, str(conv), str(RAW), str(out_dir)],
                   check=True, cwd=ROOT, capture_output=True)
    return pd.read_parquet(out_dir / "records.parquet")


def test_counts_label_and_unknown_antigen_kept_explicit(tmp_path):
    df = _run(tmp_path)
    assert len(df) == 81  # contract: exact (corrects earlier 80)
    assert df["record_id"].is_unique
    r0 = df.iloc[0]
    assert r0["source_row"] == 2
    # rank_label = fitness directly; row0 = 10.4698003
    assert abs(r0["rank_label"] - 10.4698003) < 1e-6
    assert r0["metric_name"] == "neg_log10_kd_M"
    assert r0["metric_direction"] == "higher_is_better"
    assert r0["label_kind"] == "experimental"
    assert r0["group_id"] == "garbinski2023/kd/unknown_antigen/neg_log10_kd_M/experimental"
    assert r0["heavy_chain"].startswith("VQLVESGGG")  # VH
    # proprietary antigen kept explicit, NOT invented
    assert r0["antigen_key"] == "unknown_antigen"
    assert r0["antigen_source"] == "missing"
    assert r0["antigen_sequence"] is None or (isinstance(r0["antigen_sequence"], float) and math.isnan(r0["antigen_sequence"]))


def test_deterministic(tmp_path):
    a = _run(tmp_path / "a")
    b = _run(tmp_path / "b")
    assert a.equals(b)
