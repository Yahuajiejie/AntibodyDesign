"""Batch collation for ranking examples (spec docs/programming_spec.md §5.3).

This module turns `AffinityExample` / `AffinityPairExample` (spec §5.2) into
padded tensors ready for `AffinityRanker` (spec §5.4). It does not build pairs
or groups (that is `dataset.py`) and does not construct or own any tokenizer
or pretrained model -- those are supplied by the caller (eventually
`trainer.py`, spec §5.7, not yet implemented).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

import torch

from .dataset import AffinityExample, AffinityPairExample


class Tokenizer(Protocol):
    """Minimal interface a tokenizer must satisfy to be used here.

    HuggingFace `PreTrainedTokenizerBase` instances (e.g. for ESM2 or
    AbLang-2) satisfy this protocol directly. Any other tokenizer can be used
    as long as it is wrapped to match this `__call__` signature.
    """

    def __call__(
        self, sequences: list[str], padding: bool = True, return_tensors: str = "pt"
    ) -> Mapping[str, torch.Tensor]:
        """Tokenize and pad a batch of sequences.

        Args:
            sequences: Raw sequence strings (see `_antibody_sequence` for how
                antibody sequences are constructed from `AffinityExample`).
            padding: Whether to pad to the longest sequence in the batch.
            return_tensors: Tensor framework for the returned arrays; always
                `"pt"` here.

        Returns:
            A mapping containing at least:
                - `"input_ids"`: `LongTensor[B, L]`.
                - `"attention_mask"`: `LongTensor[B, L]`, where `1` marks a
                  real token and `0` marks padding.
        """
        ...


@dataclass
class RankBatch:
    """One batch of pointwise/listwise ranking examples.

    Attributes:
        antibody_tokens: `LongTensor[B, L_ab]` antibody token ids.
        antibody_mask: `BoolTensor[B, L_ab]`. `True` = real token, `False` =
            padding.
        antigen_tokens: `LongTensor[B, L_ag]` antigen token ids, or `None` if
            no example in the batch has an antigen sequence (or no
            `antigen_tokenizer` was supplied).
        antigen_mask: `BoolTensor[B, L_ag]` matching `antigen_tokens`, or
            `None` under the same condition. Rows corresponding to examples
            with `antigen_sequence is None` are entirely `False`.
        labels: `FloatTensor[B]` of `rank_label` values.
        record_ids: `record_id` of each example, in batch order.
        group_ids: `group_id` of each example, in batch order.
    """

    antibody_tokens: torch.Tensor
    antibody_mask: torch.Tensor
    antigen_tokens: torch.Tensor | None
    antigen_mask: torch.Tensor | None
    labels: torch.Tensor
    record_ids: list[str]
    group_ids: list[str]


@dataclass
class PairBatch:
    """One batch of pairwise ranking examples.

    Attributes:
        left: `RankBatch` built from the `left` side of each
            `AffinityPairExample`.
        right: `RankBatch` built from the `right` side of each
            `AffinityPairExample`.
        y_ij: `FloatTensor[B]` of `y_ij` targets, in `{0.0, 1.0}`.
    """

    left: RankBatch
    right: RankBatch
    y_ij: torch.Tensor


def _antibody_sequence(example: AffinityExample) -> str:
    """Build the string fed to `antibody_tokenizer` for one example.

    Implements the spec §5.3 "抗体序列拼接规则":

    1. If `single_chain_sequence` is set, use it as-is.
    2. Else if both `heavy_chain` and `light_chain` are set, join them as
       `f"{heavy_chain}|{light_chain}"` (heavy first, `|`-separated --
       AbLang-2's paired-chain input convention).
    3. Else use whichever of `heavy_chain` / `light_chain` is set (the
       typical case for VHH: `heavy_chain` only).

    Args:
        example: A single `AffinityExample`.

    Returns:
        The sequence string to tokenize.

    Raises:
        ValueError: If `single_chain_sequence`, `heavy_chain`, and
            `light_chain` are all `None`. `filter_trainable_records` should
            never produce such a record.
    In theory, the specific type of antibody to be used should be determined based 
    on the type of antibody. However, since the dataset comes from many different 
    articles or many different public datasets, in order to increase robustness, 
    we adopt a method based on the existence of chains to construct
    """
    if example.single_chain_sequence is not None:
        return example.single_chain_sequence
    if example.heavy_chain is not None and example.light_chain is not None:
        return f"{example.heavy_chain}|{example.light_chain}"
    if example.heavy_chain is not None:
        return example.heavy_chain
    if example.light_chain is not None:
        return example.light_chain
    raise ValueError(
        f"AffinityExample {example.record_id!r} has no usable antibody sequence "
        "(single_chain_sequence, heavy_chain, and light_chain are all None)"
    )


def _tokenize_antibody(
    examples: Sequence[AffinityExample], tokenizer: Tokenizer
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize the antibody sequence of every example.
    Antibody must exists, so does antibody tokenizer
    Args:
        examples: Examples to tokenize, in batch order.
        tokenizer: Tokenizer to apply (spec §5.3 `Tokenizer` protocol).

    Returns:
        `(antibody_tokens, antibody_mask)`: `LongTensor[B, L]` and
        `BoolTensor[B, L]` (`True` = real token).
    """
    sequences = [_antibody_sequence(example) for example in examples]
    encoded = tokenizer(sequences, padding=True, return_tensors="pt")
    tokens = encoded["input_ids"]
    mask = encoded["attention_mask"].bool()
    return tokens, mask


def _tokenize_antigen(
    examples: Sequence[AffinityExample], tokenizer: Tokenizer | None
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Tokenize the antigen sequence of every example, if applicable.

    Implements the spec §5.3 "抗原 tokenize 规则":

    1. If `tokenizer is None`, or every example has `antigen_sequence is
       None`, the whole batch has no antigen and `(None, None)` is returned.
    2. Otherwise every example is tokenized (examples with
       `antigen_sequence is None` use an empty-string placeholder), and the
       mask row for any example with `antigen_sequence is None` is forced to
       all-`False` regardless of what the tokenizer produced for the
       placeholder.

    Args:
        examples: Examples to tokenize, in batch order.
        tokenizer: Antigen tokenizer, or `None` if the model has no antigen
            encoder.

    Returns:
        `(antigen_tokens, antigen_mask)`, each `None` or
        `(LongTensor[B, L], BoolTensor[B, L])`.
    """
    sequences = [example.antigen_sequence for example in examples]
    if tokenizer is None or all(seq is None for seq in sequences):
        return None, None

    placeholders = [seq if seq is not None else "" for seq in sequences]
    encoded = tokenizer(placeholders, padding=True, return_tensors="pt")
    tokens = encoded["input_ids"]
    mask = encoded["attention_mask"].bool()

    missing = torch.tensor([seq is None for seq in sequences], dtype=torch.bool)
    if missing.any():
        mask[missing] = False

    return tokens, mask


def collate_rank_batch(
    examples: Sequence[AffinityExample],
    antibody_tokenizer: Tokenizer,
    antigen_tokenizer: Tokenizer | None = None,
) -> RankBatch:
    """Collate `AffinityExample`s into one `RankBatch`.

    Args:
        examples: Examples to collate, in the desired batch order. Must be
            non-empty.
        antibody_tokenizer: Tokenizer applied to the antibody sequence of
            each example (built per `_antibody_sequence`).
        antigen_tokenizer: Tokenizer applied to `antigen_sequence`, or `None`
            if the model has no antigen encoder.

    Returns:
        A `RankBatch` with padded antibody/antigen tensors and per-example
        metadata, in the same order as `examples`.

    Raises:
        ValueError: If `examples` is empty, or if some example has no
            usable antibody sequence (see `_antibody_sequence`).
    """
    if not examples:
        raise ValueError("collate_rank_batch requires at least one example")

    antibody_tokens, antibody_mask = _tokenize_antibody(examples, antibody_tokenizer)
    antigen_tokens, antigen_mask = _tokenize_antigen(examples, antigen_tokenizer)

    labels = torch.tensor([example.rank_label for example in examples], dtype=torch.float32)
    record_ids = [example.record_id for example in examples]
    group_ids = [example.group_id for example in examples]

    return RankBatch(
        antibody_tokens=antibody_tokens,
        antibody_mask=antibody_mask,
        antigen_tokens=antigen_tokens,
        antigen_mask=antigen_mask,
        labels=labels,
        record_ids=record_ids,
        group_ids=group_ids,
    )


def collate_pair_batch(
    examples: Sequence[AffinityPairExample],
    antibody_tokenizer: Tokenizer,
    antigen_tokenizer: Tokenizer | None = None,
) -> PairBatch:
    """Collate `AffinityPairExample`s into one `PairBatch`.

    Args:
        examples: Pair examples to collate, in the desired batch order. Must
            be non-empty.
        antibody_tokenizer: Forwarded to `collate_rank_batch` for both
            `left` and `right`.
        antigen_tokenizer: Forwarded to `collate_rank_batch` for both `left`
            and `right`.

    Returns:
        A `PairBatch` whose `left`/`right` are independently-collated
        `RankBatch`es (so `left` and `right` may have different padded
        lengths) plus the stacked `y_ij` targets.

    Raises:
        ValueError: If `examples` is empty, or propagated from
            `collate_rank_batch` if a `left`/`right` example has no usable
            antibody sequence.
    """
    if not examples:
        raise ValueError("collate_pair_batch requires at least one example")

    left = collate_rank_batch(
        [example.left for example in examples], antibody_tokenizer, antigen_tokenizer
    )
    right = collate_rank_batch(
        [example.right for example in examples], antibody_tokenizer, antigen_tokenizer
    )
    y_ij = torch.tensor([example.y_ij for example in examples], dtype=torch.float32)

    return PairBatch(left=left, right=right, y_ij=y_ij)
