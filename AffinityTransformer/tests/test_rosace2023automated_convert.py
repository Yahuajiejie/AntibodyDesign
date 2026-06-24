"""Regression tests for rosace2023automated converters (batch 2A).

Contract: scripts/prepare/binding/rosace2023automated/conversion_contract.yaml
adalimumab / golimumab vs TNF-alpha. fitness is the PRE-COMPUTED -log10(K_D[nM])
column (must NOT be transformed again). metric = neg_log10_kd_nM (nM-based).
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _run(table_id: str, out_dir: Path) -> pd.DataFrame:
    conv = ROOT / "scripts/prepare/binding/rosace2023automated" / table_id / "convert.py"
    src = ROOT / "data/binding" / f"rosace2023automated_{table_id}.csv"
    subprocess.run([sys.executable, str(conv), str(src), str(out_dir)],
                   check=True, cwd=ROOT, capture_output=True)
    return pd.read_parquet(out_dir / "records.parquet")


def test_adalimumab_precomputed_nM_not_retransformed(tmp_path):
    df = _run("kd_adalimumab", tmp_path)
    assert len(df) == 14
    assert df["record_id"].is_unique
    r0 = df.iloc[0]  # WT, fitness = -log10(Kd[nM]) = 0.8860566...
    assert r0["source_row"] == 2
    assert abs(r0["rank_label"] - 0.8860566476931633) < 1e-9
    assert float(r0["metric_value_raw"]) == 0.8860566476931633
    assert r0["metric_name"] == "neg_log10_kd_nM"
    assert r0["metric_unit"] == "-log10(KD/nM)"
    assert r0["metric_direction"] == "higher_is_better"
    assert r0["label_kind"] == "experimental"
    assert r0["antigen_key"] == "TNFa"
    assert r0["antibody_type"] == "Fv"
    assert r0["group_id"] == "rosace2023automated/kd_adalimumab/TNFa/neg_log10_kd_nM/experimental"


def test_golimumab_count(tmp_path):
    df = _run("kd_golimumab", tmp_path)
    assert len(df) == 5
    assert df.iloc[0]["antigen_key"] == "TNFa"


def test_deterministic(tmp_path):
    a = _run("kd_adalimumab", tmp_path / "a")
    b = _run("kd_adalimumab", tmp_path / "b")
    assert a.equals(b)
