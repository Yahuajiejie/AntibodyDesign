"""Tests for the root predict.py command wrapper."""

from __future__ import annotations

import sys

import pandas as pd

import predict


def test_predict_main_reads_input_and_writes_rankings(tmp_path, monkeypatch):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "rankings.csv"
    input_path.write_text(
        "query_id,antibody_id,antigen_sequence,heavy_chain,light_chain,single_chain_sequence,antibody_type\n"
        "q1,ab1,MKTAYIAKQRQISFVKSHFSRQLE,QVQLVQSGAEVKKPGASVKVSCKAS,,,VHH\n",
        encoding="utf-8",
    )

    def fake_rank_antibody_table(input_table, model_name="best"):
        assert model_name == "best"
        assert input_table["query_id"].tolist() == ["q1"]
        return pd.DataFrame({
            "query_id": ["q1"],
            "antibody_id": ["ab1"],
            "score": [0.5],
            "rank": [1],
            "model_name": ["best"],
        })

    monkeypatch.setattr(predict, "rank_antibody_table", fake_rank_antibody_table)
    monkeypatch.setattr(
        sys,
        "argv",
        ["predict.py", "--model", "best", "--input", str(input_path), "--output", str(output_path)],
    )

    predict.main()

    result = pd.read_csv(output_path)
    assert result.to_dict(orient="records") == [{
        "query_id": "q1",
        "antibody_id": "ab1",
        "score": 0.5,
        "rank": 1,
        "model_name": "best",
    }]
