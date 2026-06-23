"""Regression tests for jain2024assessment converters (batch 2E).

Contract: scripts/prepare/binding/jain2024assessment/conversion_contract.yaml
Germline antibodies vs lysozyme controls (Octet BLI). Antigen sequences verified
against UniProt P00698 (hen) aa19-147 and P08905 (mouse) aa19-148.
rank_label = fitness = -log10(K_D[M]), higher_is_better, experimental.
"""
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HEN = ("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRN"
       "LCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL")
MOUSE = ("KVYERCEFARTLKRNGMAGYYGVSLADWVCLAQHESNYNTRATNYNRGDQSTDYGIFQINSRYWCNDGKTPRAVN"
         "ACGINCSALLQDDITAAIQCAKRVVRDPQGIRAWVAWRAHCQNRDLSQYIRNCGV")


def _run(table_id: str, raw_name: str, out_dir: Path) -> pd.DataFrame:
    conv = ROOT / "scripts/prepare/binding/jain2024assessment" / table_id / "convert.py"
    src = ROOT / "data/binding" / raw_name
    subprocess.run([sys.executable, str(conv), str(src), str(out_dir)],
                   check=True, cwd=ROOT, capture_output=True)
    return pd.read_parquet(out_dir / "records.parquet")


def test_hen_lysozyme_antigen_and_label(tmp_path):
    df = _run("Hen_Lys_kd", "jain2024assessment_Hen_Lys_kd.csv", tmp_path)
    assert len(df) == 31
    assert df["record_id"].is_unique
    r0 = df.iloc[0]
    # antigen = UniProt P00698 mature aa19-147 (129 aa), verified exact
    assert r0["antigen_sequence"] == HEN
    assert len(r0["antigen_sequence"]) == 129
    assert r0["antigen_source"] == "retrieved"
    assert r0["antigen_key"] == "HEL"
    # rank_label = fitness = neg_log_kd; row0 = 9.0
    assert abs(r0["rank_label"] - 9.0) < 1e-9
    assert r0["metric_direction"] == "higher_is_better"
    assert r0["label_kind"] == "experimental"
    assert r0["group_id"] == "jain2024assessment/Hen_Lys_kd/HEL/neg_log10_kd_M/experimental"
    assert r0["heavy_chain"].startswith("EVQLV")  # VH


def test_mouse_lysozyme_antigen_and_count(tmp_path):
    df = _run("mouse_Ly_kd", "jain2024assessment_mouse_Ly_kd.csv", tmp_path)
    assert len(df) == 2
    r0 = df.iloc[0]
    # mouse lysozyme P08905 mature aa19-148 (130 aa); Fc fusion partner omitted
    assert r0["antigen_sequence"] == MOUSE
    assert len(r0["antigen_sequence"]) == 130
    assert r0["antigen_source"] == "retrieved"
    assert abs(r0["rank_label"] - 7.0) < 1e-9
    assert r0["group_id"] == "jain2024assessment/mouse_Ly_kd/Mouse_Lysozyme/neg_log10_kd_M/experimental"


def test_hen_deterministic(tmp_path):
    a = _run("Hen_Lys_kd", "jain2024assessment_Hen_Lys_kd.csv", tmp_path / "a")
    b = _run("Hen_Lys_kd", "jain2024assessment_Hen_Lys_kd.csv", tmp_path / "b")
    assert a.equals(b)
