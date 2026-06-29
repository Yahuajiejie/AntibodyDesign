"""Minimal end-to-end integration test (spec §7.3).

spec §7.3 requires a toy processed table covering 1 Fv group, 1 VHH group,
1 missing-antigen group, 1 binary-label group, and 1 single-label group
(all present in the `toy_records` fixture, see tests/conftest.py), and a
pipeline that runs:

    load config -> load records -> build pairs -> create dataloader ->
    forward -> ranknet loss -> one optimizer step -> evaluate

without error.
"""

from __future__ import annotations

import yaml
import torch
from torch.utils.data import DataLoader

from affinity_transformer.config import load_config
from affinity_transformer.dataloader import collate_pair_batch, collate_rank_batch
from affinity_transformer.dataset import (
    AffinityRecordDataset,
    PairwiseAffinityDataset,
    build_pairs,
    filter_trainable_records,
    load_records,
)
from affinity_transformer.model import AffinityRanker
from affinity_transformer.model.losses import ranknet_loss
from affinity_transformer.trainer import Trainer

D_MODEL = 16


def test_full_pipeline_load_config_to_evaluate(tmp_path, toy_records, antibody_tokenizer, make_fake_encoder):
    """spec §7.3 minimal integration test."""

    # -- toy processed table on disk (5 groups: Fv, VHH, missing-antigen,
    # binary, single-label -- see tests/conftest.py:toy_records).
    records_path = tmp_path / "records.csv"
    toy_records.to_csv(records_path, index=False)

    # -- matching YAML config. `model.antibody_encoder="fake"` is a
    # placeholder: this test builds the model directly with FakeEncoder
    # (build_model_and_tokenizers + real ESM2 names are covered, without
    # network access, by tests/test_trainer.py).
    config_dict = {
        "data": {
            "train_path": str(records_path),
            "valid_path": str(records_path),
            "max_pairs_per_group": 50,
            "seed": 0,
        },
        "model": {
            "antibody_encoder": "fake",
            "antigen_encoder": None,
            "d_model": D_MODEL,
            "use_cross_attention": False,
        },
        "train": {
            "batch_size": 4,
            "lr": 1.0e-3,
            "epochs": 1,
            "device": "cpu",
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_dict))

    # -- load config -> load records -> build pairs.
    config = load_config(config_path)
    records = load_records(config.data.train_path)
    trainable = filter_trainable_records(records)
    pairs = build_pairs(trainable, config.data.max_pairs_per_group, config.data.seed)
    assert len(pairs) > 0

    # -- create dataloaders.
    train_dataloader = DataLoader(
        PairwiseAffinityDataset(trainable, pairs),
        batch_size=config.train.batch_size,
        shuffle=False,
        collate_fn=lambda examples: collate_pair_batch(examples, antibody_tokenizer, None),
    )
    valid_dataloader = DataLoader(
        AffinityRecordDataset(trainable),
        batch_size=config.train.batch_size,
        shuffle=False,
        collate_fn=lambda examples: collate_rank_batch(examples, antibody_tokenizer, None),
    )
    valid_record_metadata = trainable[["record_id", "dataset_id", "label_kind"]]

    # -- build model (FakeEncoder; see comment on config["model"] above).
    model = AffinityRanker(
        antibody_encoder=make_fake_encoder(config.model.d_model),
        antigen_encoder=None,
        d_model=config.model.d_model,
        use_cross_attention=config.model.use_cross_attention,
    )

    # -- forward -> ranknet loss -> one optimizer step, on the first batch,
    # exercised directly (independently of Trainer.fit's internal loop).
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    batch = next(iter(train_dataloader))
    score_i = model(batch.left)
    score_j = model(batch.right)
    loss = ranknet_loss(score_i, score_j, batch.y_ij)
    assert torch.isfinite(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # -- evaluate, via Trainer (spec §5.7), after a full epoch of training.
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        valid_record_metadata=valid_record_metadata,
    )
    trainer.fit()

    metrics = trainer.evaluate(trainer.valid_dataloader)

    for key in ("valid_macro_spearman", "valid_weighted_spearman", "n_valid_groups", "n_skipped_groups"):
        assert key in metrics
    # toy_records has 5 groups total (see tests/conftest.py docstring).
    assert metrics["n_valid_groups"] + metrics["n_skipped_groups"] == 5.0
