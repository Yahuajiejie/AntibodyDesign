"""Training loader sampling behavior."""

from __future__ import annotations

from affinity_transformer.config import (
    Config,
    DataConfig,
    EncoderConfig,
    InteractionConfig,
    ModelConfig,
    ObjectiveConfig,
    TrainConfig,
)
from affinity_transformer.training.loaders import (
    build_cached_train_loader,
    build_online_train_loader,
)
from affinity_transformer.training.samplers import GroupShuffleSampler


def _config(path, seed: int = 23) -> Config:
    return Config(
        data=DataConfig(
            train_path=path,
            valid_path=None,
            max_pairs_per_group=50,
            seed=seed,
        ),
        model=ModelConfig(
            antibody_encoder=EncoderConfig(
                name="fake",
                revision="revision-1",
                tokenizer_revision="revision-1",
                mode="frozen_online",
                embedding_layer=-1,
                cache_dir=None,
                max_length=None,
                long_sequence_strategy="error",
            ),
            antigen_encoder=None,
            interaction=InteractionConfig(
                kind="antibody_only",
                d_model=8,
                num_layers=0,
                num_heads=1,
                ffn_multiplier=4.0,
                dropout=0.0,
                pooling="masked_mean",
                bidirectional=False,
            ),
            objective=ObjectiveConfig(
                name="pairwise_ranknet",
                temperature=1.0,
                sigma=1.0,
                pointwise_loss="huber",
            ),
        ),
        train=TrainConfig(batch_size=2, lr=1e-3, epochs=2, device="cpu"),
    )


def test_online_train_loader_shuffles_reproducibly(
    tmp_path, toy_records, antibody_tokenizer
):
    path = tmp_path / "records.csv"
    toy_records.to_csv(path, index=False)
    config = _config(path)

    _, first_loader = build_online_train_loader(
        path, config, antibody_tokenizer, None
    )
    _, second_loader = build_online_train_loader(
        path, config, antibody_tokenizer, None
    )

    assert isinstance(first_loader.sampler, GroupShuffleSampler)
    first_epoch = list(iter(first_loader.sampler))
    second_epoch = list(iter(first_loader.sampler))
    recreated_first_epoch = list(iter(second_loader.sampler))
    assert sorted(first_epoch) == list(range(len(first_loader.dataset)))
    assert sorted(second_epoch) == list(range(len(first_loader.dataset)))
    assert first_epoch != second_epoch
    assert first_epoch == recreated_first_epoch


def test_cached_train_loader_uses_group_shuffle_sampler(toy_records):
    config = _config(None)
    loader = build_cached_train_loader(
        toy_records,
        config,
        antibody_store=object(),
        antigen_store=object(),
    )

    assert isinstance(loader.sampler, GroupShuffleSampler)
