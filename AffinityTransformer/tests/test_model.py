"""Tests for affinity_transformer.model (spec §5.4 / §7.2)."""

from __future__ import annotations

import torch

from affinity_transformer.dataloader import collate_rank_batch
from affinity_transformer.dataset import AffinityExample
from affinity_transformer.model import AffinityRanker

D_MODEL = 16


def _example(**overrides: object) -> AffinityExample:
    """Build an `AffinityExample` with sensible defaults, overridden as needed."""
    defaults: dict[str, object] = dict(
        record_id="r1",
        dataset_id="studyA/tableA",
        heavy_chain=None,
        light_chain=None,
        single_chain_sequence=None,
        antibody_type="Fv",
        antigen_sequence=None,
        antigen_key=None,
        rank_label=1.0,
        label_kind="experimental",
        group_id="studyA/tableA/agA/neg_log10_kd_M/experimental",
    )
    defaults.update(overrides)
    return AffinityExample(**defaults)  # type: ignore[arg-type]


def _fv() -> AffinityExample:
    return _example(
        record_id="fv1",
        heavy_chain="QVQLVQSGAEVKKPGASVKVSCKAS",
        light_chain="DIQMTQSPSSLSASVGDRVTITC",
        antigen_sequence="MKTAYIAKQRQISFVKSHFSRQLE",
    )


def _vhh() -> AffinityExample:
    return _example(
        record_id="vhh1",
        antibody_type="VHH",
        heavy_chain="QVKLEESGGGLVQAGGSLRLSCAAS",
        light_chain=None,
        antigen_sequence="MSTNPKPQRKTKRNTNRRPQDVKFPGGGQ",
        group_id="studyB/tableB/agB/neg_log10_kd_M/experimental",
    )


def _missing_antigen() -> AffinityExample:
    return _example(
        record_id="missing1",
        heavy_chain="QVQLVQSGAEVKKPGASVKVSCKAS",
        light_chain="DIQMTQSPSSLSASVGDRVTITC",
        antigen_sequence=None,
        group_id="studyC/tableC/unknown_antigen/neg_log10_kd_M/experimental",
    )


# ── basic forward shapes (spec §5.4 验收 1-3) ────────────────────────────────


def test_forward_fv(antibody_tokenizer, antigen_tokenizer, make_fake_encoder):
    batch = collate_rank_batch([_fv()], antibody_tokenizer, antigen_tokenizer)
    model = AffinityRanker(
        make_fake_encoder(D_MODEL), make_fake_encoder(D_MODEL), D_MODEL, use_cross_attention=False
    )

    score = model(batch)

    assert score.shape == (1,)
    assert torch.isfinite(score).all()


def test_forward_vhh(antibody_tokenizer, antigen_tokenizer, make_fake_encoder):
    batch = collate_rank_batch([_vhh()], antibody_tokenizer, antigen_tokenizer)
    model = AffinityRanker(
        make_fake_encoder(D_MODEL), make_fake_encoder(D_MODEL), D_MODEL, use_cross_attention=True
    )

    score = model(batch)

    assert score.shape == (1,)
    assert torch.isfinite(score).all()


def test_forward_mixed_fv_vhh(antibody_tokenizer, antigen_tokenizer, make_fake_encoder):
    batch = collate_rank_batch([_fv(), _vhh()], antibody_tokenizer, antigen_tokenizer)
    model = AffinityRanker(
        make_fake_encoder(D_MODEL), make_fake_encoder(D_MODEL), D_MODEL, use_cross_attention=True
    )

    score = model(batch)

    assert score.shape == (2,)
    assert torch.isfinite(score).all()


# ── missing antigen (spec §5.4 验收 4 / rule 2) ──────────────────────────────


def test_forward_missing_antigen_whole_batch_no_nan(antibody_tokenizer, antigen_tokenizer, make_fake_encoder):
    examples = [_missing_antigen(), _missing_antigen()]
    batch = collate_rank_batch(examples, antibody_tokenizer, antigen_tokenizer)
    assert batch.antigen_tokens is None  # whole batch has no antigen

    model = AffinityRanker(
        make_fake_encoder(D_MODEL), make_fake_encoder(D_MODEL), D_MODEL, use_cross_attention=True
    )
    score = model(batch)

    assert score.shape == (2,)
    assert torch.isfinite(score).all()


def test_forward_missing_antigen_mixed_row_no_nan(antibody_tokenizer, antigen_tokenizer, make_fake_encoder):
    batch = collate_rank_batch([_fv(), _missing_antigen()], antibody_tokenizer, antigen_tokenizer)
    assert batch.antigen_tokens is not None
    assert not batch.antigen_mask[1].any()  # second row's antigen is missing

    model = AffinityRanker(
        make_fake_encoder(D_MODEL), make_fake_encoder(D_MODEL), D_MODEL, use_cross_attention=True
    )
    score = model(batch)

    assert score.shape == (2,)
    assert torch.isfinite(score).all()


def test_forward_no_antigen_encoder_is_antibody_only(antibody_tokenizer, antigen_tokenizer, make_fake_encoder):
    batch = collate_rank_batch([_fv()], antibody_tokenizer, antigen_tokenizer)

    model = AffinityRanker(make_fake_encoder(D_MODEL), None, D_MODEL, use_cross_attention=True)
    score = model(batch)

    assert score.shape == (1,)
    assert torch.isfinite(score).all()


# ── backward (spec §5.4 验收 5) ──────────────────────────────────────────────


def test_forward_backward_no_nan_gradients(antibody_tokenizer, antigen_tokenizer, make_fake_encoder):
    batch = collate_rank_batch(
        [_fv(), _vhh(), _missing_antigen()], antibody_tokenizer, antigen_tokenizer
    )
    model = AffinityRanker(
        make_fake_encoder(D_MODEL), make_fake_encoder(D_MODEL), D_MODEL, use_cross_attention=True
    )

    score = model(batch)
    score.sum().backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads  # at least some parameters received gradients
    for grad in grads:
        assert torch.isfinite(grad).all()


# ── score is unbounded (spec §5.4 rule 1) ────────────────────────────────────


def test_head_applies_no_sigmoid_softmax_or_clamp(antibody_tokenizer, antigen_tokenizer, make_fake_encoder):
    model = AffinityRanker(
        make_fake_encoder(D_MODEL), make_fake_encoder(D_MODEL), D_MODEL, use_cross_attention=False
    )

    bounded_modules = (torch.nn.Sigmoid, torch.nn.Softmax, torch.nn.Tanh, torch.nn.Hardtanh)
    assert not any(isinstance(module, bounded_modules) for module in model.modules())
