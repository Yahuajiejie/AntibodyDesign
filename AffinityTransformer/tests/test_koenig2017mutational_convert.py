"""Regression tests for koenig2017mutational converter (batch 2D).

Contract: scripts/prepare/binding/koenig2017mutational/conversion_contract.yaml
G6.31 anti-VEGF-A. TARGET (VEGF-A) is verified; the exact assay construct is
UNRESOLVED. The contract preserves the current sequence TEMPORARILY, flagged as
unverified. These tests guard that koenig did NOT gain a newly guessed construct
(sequence unchanged) while preserving label/unit/direction.
"""
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/binding/koenig2017mutational_kd_g6.csv"
# The exact sequence currently retained (unverified) -- must remain unchanged.
RETAINED_VEGF = (
    "APMAEGGGQNHHEVVKFMDVYQRSYCHPIETLVDIFQEYPDEIEYIFKPSCVPLMRCGGCCNDEGLECVPTEES"
    "NITMQIMRIKPHQGQHIGEMSFLQHNKCECRPKKDRARQENPCGPCSERRKHLFVQDPQTCKCSCKNTDSRCKA"
    "RQLELNERTCRCDKPRR"
)


def _run(out_dir: Path) -> pd.DataFrame:
    conv = ROOT / "scripts/prepare/binding/koenig2017mutational/kd_g6/convert.py"
    subprocess.run([sys.executable, str(conv), str(RAW), str(out_dir)],
                   check=True, cwd=ROOT, capture_output=True)
    return pd.read_parquet(out_dir / "records.parquet")


def test_target_preserved_and_no_new_guessed_construct(tmp_path):
    df = _run(tmp_path)
    r0 = df.iloc[0]
    assert r0["antigen_key"] == "VEGF_A"          # target verified, unchanged
    assert r0["antigen_source"] == "retrieved"     # contract: preserve temporarily
    # sequence must be EXACTLY the retained value -- not replaced by a guessed
    # VEGF165 or VEGF8-109 construct.
    assert r0["antigen_sequence"] == RETAINED_VEGF
    assert len(r0["antigen_sequence"]) == 165
    assert len(r0["antigen_sequence"]) != 109      # not silently swapped to RBD


def test_schema_counts_label_and_direction(tmp_path):
    df = _run(tmp_path)
    assert len(df) == 4275
    assert df["record_id"].is_unique
    r0 = df.iloc[0]
    assert r0["source_row"] == 2
    # rank_label = fitness = -log10(K_D[M]); row0 = 8.441265283
    assert abs(r0["rank_label"] - 8.441265283) < 1e-6
    assert r0["metric_name"] == "neg_log10_kd_M"
    assert r0["metric_unit"] == "-log10(KD/M)"
    assert r0["metric_direction"] == "higher_is_better"
    assert r0["label_kind"] == "experimental"
    assert r0["group_id"] == "koenig2017mutational/kd_g6/VEGF_A/neg_log10_kd_M/experimental"
    assert r0["antibody_type"] == "Fv"
    assert r0["heavy_chain"].startswith("EAQL")  # VH variant


def test_deterministic(tmp_path):
    a = _run(tmp_path / "a")
    b = _run(tmp_path / "b")
    assert a.equals(b)
