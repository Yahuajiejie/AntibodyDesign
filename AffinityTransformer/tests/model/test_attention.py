"""Tests for model attention construction helpers."""

from affinity_transformer.model.attention import build_cross_attention, select_num_heads


def test_select_num_heads_uses_largest_supported_divisor():
    assert select_num_heads(480) == 8
    assert select_num_heads(10) == 2
    assert select_num_heads(7) == 1


def test_build_cross_attention_uses_batch_first():
    attention = build_cross_attention(16)

    assert attention.embed_dim == 16
    assert attention.num_heads == 8
    assert attention.batch_first is True
