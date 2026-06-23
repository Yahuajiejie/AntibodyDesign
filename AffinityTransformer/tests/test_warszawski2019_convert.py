"""Regression tests for warszawski2019 converter (batch 2D).

Contract: scripts/prepare/binding/warszawski2019/conversion_contract.yaml
D44.1 = anti-hen-egg-white-lysozyme (PDB 1MLC). The prior converter wrongly
labelled the antigen VEGF-A (copy-paste from koenig). Antigen MUST be HEL,
UniProt P00698 mature aa19-147. rank_label = -log10(K_D[nM]).
"""
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/binding/warszawski2019_d44_Kd.csv"
HEL = ("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRN"
       "LCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL")


def _run(out_dir: Path) -> pd.DataFrame:
    conv = ROOT / "scripts/prepare/binding/warszawski2019/d44_Kd/convert.py"
    subprocess.run([sys.executable, str(conv), str(RAW), str(out_dir)],
                   check=True, cwd=ROOT, capture_output=True)
    return pd.read_parquet(out_dir / "records.parquet")


def test_antigen_is_hel_never_vegf(tmp_path):
    df = _run(tmp_path)
    r0 = df.iloc[0]
    assert r0["antigen_key"] == "HEL"
    assert r0["antigen_name"] == "Hen egg lysozyme"
    assert r0["antigen_sequence"] == HEL
    assert len(r0["antigen_sequence"]) == 129
    assert r0["antigen_source"] == "retrieved"
    assert "/HEL/" in r0["group_id"]
    # No VEGF identity or sequence may remain anywhere in the table.
    blob = df.astype(str).to_csv()
    assert "VEGF" not in blob
    assert "APMAEGGG" not in blob  # start of the old VEGF sequence


def test_schema_counts_label_and_provenance(tmp_path):
    df = _run(tmp_path)
    assert len(df) == 2048
    assert df["record_id"].is_unique
    r0 = df.iloc[0]
    assert r0["source_row"] == 2
    # rank_label = fitness column = -log10(Kd[nM]); row0 = 0.2649196339...
    assert abs(r0["rank_label"] - 0.2649196339406016) < 1e-9
    assert r0["metric_name"] == "neg_log10_kd_nM"
    assert r0["metric_unit"] == "-log10(KD/nM)"
    assert r0["metric_direction"] == "higher_is_better"
    assert r0["label_kind"] == "experimental"
    assert r0["group_id"] == "warszawski2019/d44_Kd/HEL/neg_log10_kd_nM/experimental"
    assert r0["antibody_type"] == "Fv"
    assert r0["heavy_chain"].startswith("QVQL")  # VH


def test_deterministic(tmp_path):
    a = _run(tmp_path / "a")
    b = _run(tmp_path / "b")
    assert a.equals(b)
