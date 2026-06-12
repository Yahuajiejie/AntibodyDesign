"""Tests for affinity_transformer.dataloader (spec §5.3 / §7.2)."""

from __future__ import annotations

import pytest
import torch

from affinity_transformer.dataloader import (
    PairBatch,
    RankBatch,
    collate_pair_batch,
    collate_rank_batch,
)
from affinity_transformer.dataset import AffinityExample, AffinityPairExample


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


# ── collate_rank_batch: basic shapes ────────────────────────────────────────


def test_collate_rank_batch_batch_size_one(antibody_tokenizer):
    example = _example(heavy_chain="QVQLVQSGAEVKKPGASVKVSCKAS", light_chain="DIQMTQSPSSLSASVGDRVTITC")

    batch = collate_rank_batch([example], antibody_tokenizer)

    assert isinstance(batch, RankBatch)
    assert batch.antibody_tokens.shape[0] == 1
    assert batch.antibody_mask.shape == batch.antibody_tokens.shape
    assert batch.antibody_mask.dtype == torch.bool
    assert batch.antigen_tokens is None
    assert batch.antigen_mask is None
    assert batch.labels.tolist() == [1.0]
    assert batch.record_ids == ["r1"]
    assert batch.group_ids == [example.group_id]


def test_collate_rank_batch_rejects_empty(antibody_tokenizer):
    with pytest.raises(ValueError):
        collate_rank_batch([], antibody_tokenizer)


def test_collate_rank_batch_mixed_fv_vhh(antibody_tokenizer):
    fv = _example(
        record_id="fv1", heavy_chain="QVQLVQSGAEVKKPGASVKVSCKAS", light_chain="DIQMTQSPSSLSASVGDRVTITC"
    )
    vhh = _example(
        record_id="vhh1",
        antibody_type="VHH",
        heavy_chain="QVKLEESGGGLVQAGGSLRLSCAAS",
        light_chain=None,
        group_id="studyB/tableB/agB/neg_log10_kd_M/experimental",
    )

    batch = collate_rank_batch([fv, vhh], antibody_tokenizer)

    assert batch.antibody_tokens.shape[0] == 2
    assert batch.antibody_mask.shape == batch.antibody_tokens.shape
    # Shorter (VHH-only-heavy) sequence is padded -> its mask has at least one False.
    assert not batch.antibody_mask[1].all()
    assert batch.record_ids == ["fv1", "vhh1"]


# ── antibody sequence construction rules (spec §5.3) ────────────────────────


def test_antibody_sequence_single_chain_used_as_is(antibody_tokenizer):
    example = _example(single_chain_sequence="EVQLVESGGGLVQ", heavy_chain="IGNORED", light_chain="IGNORED")

    batch = collate_rank_batch([example], antibody_tokenizer)
    expected = antibody_tokenizer(["EVQLVESGGGLVQ"], padding=True, return_tensors="pt")

    assert torch.equal(batch.antibody_tokens, expected["input_ids"])


def test_antibody_sequence_heavy_and_light_joined_with_pipe(antibody_tokenizer):
    example = _example(heavy_chain="QVQLVQSGAEVKKPGASVKVSCKAS", light_chain="DIQMTQSPSSLSASVGDRVTITC")

    batch = collate_rank_batch([example], antibody_tokenizer)
    expected = antibody_tokenizer(
        ["QVQLVQSGAEVKKPGASVKVSCKAS|DIQMTQSPSSLSASVGDRVTITC"], padding=True, return_tensors="pt"
    )

    assert torch.equal(batch.antibody_tokens, expected["input_ids"])


def test_antibody_sequence_heavy_only_for_vhh(antibody_tokenizer):
    example = _example(antibody_type="VHH", heavy_chain="QVKLEESGGGLVQAGGSLRLSCAAS", light_chain=None)

    batch = collate_rank_batch([example], antibody_tokenizer)
    expected = antibody_tokenizer(["QVKLEESGGGLVQAGGSLRLSCAAS"], padding=True, return_tensors="pt")

    assert torch.equal(batch.antibody_tokens, expected["input_ids"])


def test_antibody_sequence_light_only(antibody_tokenizer):
    example = _example(heavy_chain=None, light_chain="DIQMTQSPSSLSASVGDRVTITC")

    batch = collate_rank_batch([example], antibody_tokenizer)
    expected = antibody_tokenizer(["DIQMTQSPSSLSASVGDRVTITC"], padding=True, return_tensors="pt")

    assert torch.equal(batch.antibody_tokens, expected["input_ids"])


def test_antibody_sequence_raises_when_no_chain_available(antibody_tokenizer):
    example = _example(heavy_chain=None, light_chain=None, single_chain_sequence=None)

    with pytest.raises(ValueError):
        collate_rank_batch([example], antibody_tokenizer)


# ── antigen tokenize rules (spec §5.3) ──────────────────────────────────────


def test_collate_rank_batch_all_antigen_missing(antibody_tokenizer, antigen_tokenizer):
    examples = [
        _example(record_id="r1", heavy_chain="QVQLVQSGAEVKKPGASVKVSCKAS", antigen_sequence=None),
        _example(record_id="r2", heavy_chain="QVQLVQSGAEVKKPGASVKVSCKAS", antigen_sequence=None),
    ]

    batch = collate_rank_batch(examples, antibody_tokenizer, antigen_tokenizer)

    assert batch.antigen_tokens is None
    assert batch.antigen_mask is None


def test_collate_rank_batch_no_antigen_tokenizer(antibody_tokenizer):
    examples = [
        _example(record_id="r1", heavy_chain="QVQLVQSGAEVKKPGASVKVSCKAS", antigen_sequence="MKTAYIAKQRQ"),
    ]

    batch = collate_rank_batch(examples, antibody_tokenizer, antigen_tokenizer=None)

    assert batch.antigen_tokens is None
    assert batch.antigen_mask is None


def test_collate_rank_batch_mixed_antigen_missing(antibody_tokenizer, antigen_tokenizer):
    with_antigen = _example(
        record_id="with_ag", heavy_chain="QVQLVQSGAEVKKPGASVKVSCKAS", antigen_sequence="MKTAYIAKQRQISFVKSHFSRQLE"
    )
    without_antigen = _example(
        record_id="without_ag", heavy_chain="QVQLVQSGAEVKKPGASVKVSCKAS", antigen_sequence=None
    )

    batch = collate_rank_batch([with_antigen, without_antigen], antibody_tokenizer, antigen_tokenizer)

    assert batch.antigen_tokens is not None
    assert batch.antigen_mask is not None
    assert batch.antigen_mask.shape == batch.antigen_tokens.shape
    # Row 0 (antigen present) has at least one valid token.
    assert batch.antigen_mask[0].any()
    # Row 1 (antigen missing) is entirely False, regardless of the placeholder tokenization.
    assert not batch.antigen_mask[1].any()


def test_collate_rank_batch_mask_shape_matches_tokens(antibody_tokenizer, antigen_tokenizer):
    examples = [
        _example(
            record_id="r1",
            heavy_chain="QVQLVQSGAEVKKPGASVKVSCKAS",
            light_chain="DIQMTQSPSSLSASVGDRVTITC",
            antigen_sequence="MKTAYIAKQRQISFVKSHFSRQLE",
        ),
        _example(
            record_id="r2",
            antibody_type="VHH",
            heavy_chain="QVKLEESGGGLVQAGGSLRLSCAAS",
            light_chain=None,
            antigen_sequence="MSTNPKPQRKTKRNTNRRPQDVKFPGGGQ",
            group_id="studyB/tableB/agB/neg_log10_kd_M/experimental",
        ),
    ]

    batch = collate_rank_batch(examples, antibody_tokenizer, antigen_tokenizer)

    assert batch.antibody_mask.shape == batch.antibody_tokens.shape
    assert batch.antigen_mask.shape == batch.antigen_tokens.shape


# ── collate_pair_batch ───────────────────────────────────────────────────────


def test_collate_pair_batch_splits_left_right(antibody_tokenizer, antigen_tokenizer):
    left = _example(
        record_id="left",
        heavy_chain="QVQLVQSGAEVKKPGASVKVSCKAS",
        light_chain="DIQMTQSPSSLSASVGDRVTITC",
        antigen_sequence="MKTAYIAKQRQISFVKSHFSRQLE",
        rank_label=2.0,
    )
    right = _example(
        record_id="right",
        antibody_type="VHH",
        heavy_chain="QVKLEESGGGLVQAGGSLRLSCAAS",
        light_chain=None,
        antigen_sequence=None,
        rank_label=1.0,
    )
    pair = AffinityPairExample(pair_id="p1", group_id=left.group_id, left=left, right=right, y_ij=1.0)

    batch = collate_pair_batch([pair], antibody_tokenizer, antigen_tokenizer)

    assert isinstance(batch, PairBatch)
    assert isinstance(batch.left, RankBatch)
    assert isinstance(batch.right, RankBatch)
    assert batch.left.record_ids == ["left"]
    assert batch.right.record_ids == ["right"]
    assert batch.left.labels.tolist() == [2.0]
    assert batch.right.labels.tolist() == [1.0]
    assert batch.y_ij.tolist() == [1.0]
    # left/right are collated independently: every "right" example in this
    # batch has antigen_sequence=None, so right's antigen tensors are the
    # whole-batch-missing None (rule 1), even though left's are not.
    assert batch.left.antigen_tokens is not None
    assert batch.right.antigen_tokens is None
    assert batch.right.antigen_mask is None


def test_collate_pair_batch_rejects_empty(antibody_tokenizer):
    with pytest.raises(ValueError):
        collate_pair_batch([], antibody_tokenizer)
