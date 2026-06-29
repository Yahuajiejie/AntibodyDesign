import importlib.util
from pathlib import Path


def _load_abrank_convert():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/prepare/binding/AbRank/dataset/convert.py"
    )
    spec = importlib.util.spec_from_file_location("abrank_convert", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_derive_sars_cov_2_rbd_mutant_uses_abrank_offset():
    module = _load_abrank_convert()

    seq = module._derive_sars_cov_2_rbd_mutant("SARS_CoV_2_E483K")

    assert seq is not None
    assert len(seq) == 223
    assert seq[484 - 319] == "K"


def test_derive_sars_cov_2_rbd_mutant_rejects_reference_mismatch():
    module = _load_abrank_convert()

    assert module._derive_sars_cov_2_rbd_mutant("SARS_CoV_2_V483K") is None


def test_antigen_sequence_and_source_prefers_provided_sequence():
    module = _load_abrank_convert()

    seq, source = module._antigen_sequence_and_source(
        {"Ag_seq": "ACDEFGHIK"}, "SARS_CoV_2_E483K"
    )

    assert seq == "ACDEFGHIK"
    assert source == "provided"
