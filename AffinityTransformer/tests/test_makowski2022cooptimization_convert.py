"""Regression tests for makowski2022cooptimization converters (batch 2A).

Contract: scripts/prepare/binding/makowski2022cooptimization/conversion_contract.yaml
emibetuzumab CDR panels vs MET (ANT, on-target) and Ovalbumin (OVA, off-target).
metric = rel_binding_signal (used directly). OVA tables are OFF-TARGET: the
labels must stay measurement-faithful (higher signal = more OVA binding) and NOT
be inverted. The iso_ant raw file is intentionally misspelled 'makowksi' and must
remain traceable.
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# table_id -> (raw_filename, antigen_key, count)
TABLES = {
    "igg_ant": ("makowski2022cooptimization_igg_ant.csv", "MET", 96),
    "igg_ova": ("makowski2022cooptimization_igg_ova.csv", "Ovalbumin", 96),
    "iso_ant": ("makowksi2022cooptimization_iso_ant.csv", "MET", 126),   # misspelled on purpose
    "iso_ova": ("makowski2022cooptimization_iso_ova.csv", "Ovalbumin", 126),
}


def _run(table_id: str, out_dir: Path) -> pd.DataFrame:
    conv = ROOT / "scripts/prepare/binding/makowski2022cooptimization" / table_id / "convert.py"
    src = ROOT / "data/binding" / TABLES[table_id][0]
    subprocess.run([sys.executable, str(conv), str(src), str(out_dir)],
                   check=True, cwd=ROOT, capture_output=True)
    return pd.read_parquet(out_dir / "records.parquet")


def test_counts_antigens_and_metric(tmp_path):
    for table_id, (_, antigen_key, count) in TABLES.items():
        df = _run(table_id, tmp_path / table_id)
        assert len(df) == count, table_id
        assert df["record_id"].is_unique, table_id
        r0 = df.iloc[0]
        assert r0["antigen_key"] == antigen_key
        assert r0["metric_name"] == "rel_binding_signal"
        assert r0["metric_direction"] == "higher_is_better"
        assert r0["label_kind"] == "experimental"


def test_ova_direction_is_measurement_faithful_not_inverted(tmp_path):
    df = _run("igg_ova", tmp_path)
    # rank_label must equal the raw OVA signal (no negation / inversion)
    raw = pd.to_numeric(df["metric_value_numeric"])
    rank = pd.to_numeric(df["rank_label"])
    keep = df["keep_for_training"].astype(bool)
    assert (raw[keep] == rank[keep]).all()
    # higher raw OVA signal -> higher rank (monotonic, not inverted)
    a, b = df.iloc[0], df[pd.to_numeric(df["metric_value_numeric"]) > 0].iloc[0]
    assert float(a["rank_label"]) == float(a["metric_value_numeric"])  # 0.0
    assert float(b["rank_label"]) == float(b["metric_value_numeric"])
    assert df.iloc[0]["metric_direction"] == "higher_is_better"


def test_misspelled_iso_ant_source_is_traceable(tmp_path):
    df = _run("iso_ant", tmp_path)
    r0 = df.iloc[0]
    assert "makowksi2022cooptimization_iso_ant.csv" in r0["source_file"]
    assert r0["antigen_key"] == "MET"
    # source-row traceability preserved
    assert r0["source_row"] == 2


def test_ant_on_target_faithful(tmp_path):
    df = _run("igg_ant", tmp_path)
    r0 = df.iloc[0]  # ANT binding 0.8
    assert abs(float(r0["rank_label"]) - 0.8) < 1e-9
    assert r0["antigen_key"] == "MET"


def test_deterministic(tmp_path):
    a = _run("iso_ova", tmp_path / "a")
    b = _run("iso_ova", tmp_path / "b")
    assert a.equals(b)
