"""Regression tests for peterson2024integrated converters (batch 2D).

Contract: scripts/prepare/binding/peterson2024integrated/conversion_contract.yaml
MAGMA-seq human Fab libraries vs influenza H1 HA. antibody_type = Fab (was IgG).
ab_H1HA_kd: K_D in nM -> rank_label = -log10(K_D[M]); experimental.
ab_H1HA_binary: fitness in {0,1}; label_kind binary. The two tables must stay
in distinct groups. H1-HA sequence is unresolved -> antigen_source missing.
"""
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _run(table_id: str, raw_name: str, out_dir: Path) -> pd.DataFrame:
    conv = ROOT / "scripts/prepare/binding/peterson2024integrated" / table_id / "convert.py"
    src = ROOT / "data/binding" / raw_name
    subprocess.run([sys.executable, str(conv), str(src), str(out_dir)],
                   check=True, cwd=ROOT, capture_output=True)
    return pd.read_parquet(out_dir / "records.parquet")


def test_kd_table_fab_unit_and_label(tmp_path):
    df = _run("ab_H1HA_kd", "peterson2024integrated_ab_H1HA_kd.csv", tmp_path)
    assert len(df) == 1040
    assert df["record_id"].is_unique
    r0 = df.iloc[0]
    assert r0["antibody_type"] == "Fab"
    # K_D unit nM: row0 fitness=2490.81 nM -> -log10(2490.81e-9) = 5.6035
    assert abs(r0["rank_label"] - (-math.log10(2490.81e-9))) < 1e-9
    assert r0["metric_name"] == "neg_log10_kd_M"
    assert r0["metric_unit"] == "-log10(KD/M)"
    assert r0["metric_direction"] == "higher_is_better"
    assert r0["label_kind"] == "experimental"
    assert r0["group_id"] == "peterson2024integrated/ab_H1HA_kd/H1_HA/neg_log10_kd_M/experimental"
    # H1-HA construct unresolved -> kept missing, not invented
    assert r0["antigen_source"] == "missing"
    assert r0["antigen_sequence"] is None or (isinstance(r0["antigen_sequence"], float) and math.isnan(r0["antigen_sequence"]))


def test_binary_table_fab_and_semantics(tmp_path):
    df = _run("ab_H1HA_binary", "peterson2024integrated_ab_H1HA_binary.csv", tmp_path)
    assert len(df) == 1071
    r0 = df.iloc[0]
    assert r0["antibody_type"] == "Fab"
    assert r0["label_kind"] == "binary"
    assert r0["metric_name"] == "binary"
    assert r0["group_id"] == "peterson2024integrated/ab_H1HA_binary/H1_HA/binary/binary"
    # binary 0/1 semantics preserved
    vals = set(pd.to_numeric(df.loc[df["keep_for_training"], "rank_label"]).unique())
    assert vals <= {0.0, 1.0}
    assert r0["antigen_source"] == "missing"


def test_binary_and_continuous_in_distinct_groups(tmp_path):
    kd = _run("ab_H1HA_kd", "peterson2024integrated_ab_H1HA_kd.csv", tmp_path / "kd")
    bn = _run("ab_H1HA_binary", "peterson2024integrated_ab_H1HA_binary.csv", tmp_path / "bn")
    assert set(kd["group_id"]).isdisjoint(set(bn["group_id"]))
    assert set(kd["label_kind"]) == {"experimental"}
    assert set(bn["label_kind"]) == {"binary"}


def test_deterministic(tmp_path):
    a = _run("ab_H1HA_kd", "peterson2024integrated_ab_H1HA_kd.csv", tmp_path / "a")
    b = _run("ab_H1HA_kd", "peterson2024integrated_ab_H1HA_kd.csv", tmp_path / "b")
    assert a.equals(b)
