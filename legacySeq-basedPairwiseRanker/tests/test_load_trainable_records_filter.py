"""Tests for `training.data.load_trainable_records` applying
`config.data.record_filter` in explicit-path mode (`split_strategy="none"`).

Before this change, `resolve_data_paths` only applied `record_filter` in
automatic-split mode -- every `configs/v065/*.yaml` uses explicit paths, so
the filter was silently a no-op for them regardless of what `data.filter`
said. This was found while excluding `label_kind="predicted"` groups
(ML-model-predicted affinity scores, not real measurements) from the
`noise_aware_multiscale` configs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from affinity_transformer.config import DataConfig
from affinity_transformer.record_filter import RecordFilterConfig
from affinity_transformer.training.data import load_trainable_records

_BASE_ROW = {
    "dataset_id": "studyA/tableA",
    "study_id": "studyA",
    "table_id": "tableA",
    "source_file": "data/binding/studyA/tableA.csv",
    "source_row": 0,
    "antibody_id": "ab",
    "antibody_type": "Fv",
    "heavy_chain": "QVQLVQSGAEVKKPGASVKVSCKAS",
    "light_chain": "DIQMTQSPSSLSASVGDRVTITC",
    "single_chain_sequence": None,
    "antigen_key": "agA",
    "antigen_name": "Antigen A",
    "antigen_sequence": "MKTAYIAKQRQISFVKSHFSRQLE",
    "antigen_source": "provided",
    "assay_name": "SPR",
    "assay_type": "binding",
    "metric_name": "neg_log10_kd_M",
    "metric_value_raw": "1.0",
    "metric_value_numeric": 1.0,
    "metric_unit": "-log10(KD/M)",
    "metric_direction": "higher_is_better",
    "transform_rule": "rank_label = neg_log10_kd_M",
    "keep_for_training": True,
    "drop_reason": None,
}


def _mixed_label_kind_records() -> pd.DataFrame:
    rows = []
    for i in range(5):
        rows.append({
            **_BASE_ROW, "record_id": f"exp/{i}", "rank_label": float(i),
            "label_kind": "experimental", "group_id": "g/experimental",
        })
    for i in range(5):
        rows.append({
            **_BASE_ROW, "record_id": f"pred/{i}", "rank_label": float(i),
            "label_kind": "predicted", "group_id": "g/predicted",
        })
    return pd.DataFrame(rows)


def _minimal_config(record_filter: RecordFilterConfig) -> object:
    from affinity_transformer.config import (
        Config, EncoderConfig, InteractionConfig, ModelConfig, ObjectiveConfig, TrainConfig,
    )
    data = DataConfig(
        train_path=None, valid_path=None, max_pairs_per_group=10, seed=0,
        record_filter=record_filter,
    )
    encoder = EncoderConfig(
        name="dummy", revision="main", tokenizer_revision="main", mode="frozen_online",
        embedding_layer=-1, cache_dir=None, max_length=None, long_sequence_strategy="error",
    )
    model = ModelConfig(
        antibody_encoder=encoder, antigen_encoder=None,
        interaction=InteractionConfig(
            kind="antibody_only", d_model=8, num_layers=0, num_heads=1,
            ffn_multiplier=4.0, dropout=0.1, pooling="masked_mean", bidirectional=False,
        ),
        objective=ObjectiveConfig(name="pairwise_ranknet", temperature=1.0, sigma=1.0, pointwise_loss="huber"),
    )
    train = TrainConfig(batch_size=4, lr=1e-3, epochs=1, device="cpu")
    return Config(data=data, model=model, train=train)


def test_load_trainable_records_applies_exclude_label_kinds(tmp_path: Path):
    records_path = tmp_path / "records.parquet"
    _mixed_label_kind_records().to_parquet(records_path)

    config = _minimal_config(RecordFilterConfig(exclude_label_kinds=("predicted",)))
    loaded = load_trainable_records(records_path, config)

    assert set(loaded["label_kind"]) == {"experimental"}
    assert len(loaded) == 5
    assert "g/predicted" not in set(loaded["group_id"])


def test_load_trainable_records_no_filter_keeps_everything(tmp_path: Path):
    records_path = tmp_path / "records.parquet"
    _mixed_label_kind_records().to_parquet(records_path)

    config = _minimal_config(RecordFilterConfig())  # empty filter, the default
    loaded = load_trainable_records(records_path, config)

    assert set(loaded["label_kind"]) == {"experimental", "predicted"}
    assert len(loaded) == 10


def test_load_trainable_records_raises_if_filter_removes_everything(tmp_path: Path):
    records_path = tmp_path / "records.parquet"
    _mixed_label_kind_records().to_parquet(records_path)

    config = _minimal_config(
        RecordFilterConfig(exclude_label_kinds=("experimental", "predicted"))
    )
    with pytest.raises(ValueError, match="removed every trainable record"):
        load_trainable_records(records_path, config)
