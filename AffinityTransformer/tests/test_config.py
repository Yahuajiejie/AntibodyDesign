"""Tests for affinity_transformer.config (spec §5.1)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from affinity_transformer.config import Config, load_config


def _write_config(
    tmp_path: Path,
    overrides: dict[str, dict[str, object]],
    *,
    train_path: Path,
    valid_path: Path | None = None,
) -> Path:
    payload: dict[str, dict[str, object]] = {
        "data": {
            "train_path": str(train_path),
            "valid_path": None if valid_path is None else str(valid_path),
            "max_pairs_per_group": 50,
            "seed": 0,
        },
        "model": {
            "antibody_encoder": "esm2_t12_35M",
            "antigen_encoder": None,
            "d_model": 480,
            "use_cross_attention": False,
        },
        "train": {
            "batch_size": 16,
            "lr": 1.0e-4,
            "epochs": 10,
            "device": "cpu",
        },
    }
    payload.update(overrides) # 如果 overrides 中存在与 payload 相同的键，payload 中原本的值就会被替换为 overrides 中的新值。
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload)) # safe_dump()将 Python 对象（如字典、列表）安全地序列化为 YAML 格式的字符串4。
    return config_path


@pytest.fixture()
def existing_train_path(tmp_path: Path) -> Path:
    train_path = tmp_path / "records.parquet"
    train_path.write_bytes(b"")  # only existence is checked by load_config
    return train_path

# tmp_path是 Pytest 自带的一个官方顶级夹具（Built-in Fixture）。
# 每次只要有一个新的测试函数运行，Pytest 就会在系统的临时目录下，为这个测试函数独立创建一个专属的、名字随机的临时文件夹。
# 由于该函数带有@pytest.fixture()，它在 Pytest 后台已经登记注册了。只要函数的参数有这个函数的名字，pytest就会调用它

def test_load_config_valid(tmp_path, existing_train_path):
    # 测试正常config能不能加载
    config_path = _write_config(tmp_path, {}, train_path=existing_train_path)

    config = load_config(config_path)

    assert isinstance(config, Config)
    assert config.data.train_path == existing_train_path
    assert config.data.valid_path is None
    assert config.data.max_pairs_per_group == 50
    assert config.data.large_group_threshold == 10_000
    assert config.data.pair_enumeration_limit == 100_000
    assert config.data.label_block_count == 5
    assert config.data.intra_block_pairs_per_large_group == 50
    assert config.data.discrete_label_unique_threshold == 32
    assert config.data.discrete_label_ratio_threshold == pytest.approx(0.05)
    assert config.data.seed == 0
    assert config.model.antibody_encoder.name == "esm2_t12_35M"
    assert config.model.antibody_encoder.mode == "frozen_online"
    assert config.model.antigen_encoder is None
    assert config.model.interaction.kind == "antibody_only"
    assert config.model.objective.name == "pairwise_ranknet"
    assert config.model.use_cross_attention is False
    assert config.train.device == "cpu"
    assert config.train.lr == pytest.approx(1.0e-4)


def test_load_config_parses_v065_cached_concat_model(tmp_path, existing_train_path):
    antibody_cache = tmp_path / "ab-cache"
    antigen_cache = tmp_path / "ag-cache"
    antibody_cache.mkdir()
    antigen_cache.mkdir()
    model = {
        "antibody_encoder": {
            "name": "igbert",
            "revision": "ab-rev-123",
            "tokenizer_revision": "ab-tokenizer-123",
            "mode": "frozen_cached",
            "embedding_layer": -1,
            "cache_dir": str(antibody_cache),
            "max_length": 256,
            "long_sequence_strategy": "truncate",
        },
        "antigen_encoder": {
            "name": "esm2_t12_35M",
            "revision": "ag-rev-456",
            "tokenizer_revision": "ag-tokenizer-456",
            "mode": "frozen_cached",
            "embedding_layer": -1,
            "cache_dir": str(antigen_cache),
            "max_length": 1024,
            "long_sequence_strategy": "truncate",
        },
        "interaction": {
            "kind": "concat",
            "d_model": 64,
            "num_layers": 0,
            "num_heads": 8,
            "ffn_multiplier": 4.0,
            "dropout": 0.1,
            "pooling": "masked_mean",
            "bidirectional": True,
        },
        "objective": {
            "name": "pairwise_ranknet",
            "temperature": 1.0,
            "sigma": 2.0,
            "pointwise_loss": "huber",
        },
    }
    config_path = _write_config(
        tmp_path,
        {"model": model},
        train_path=existing_train_path,
    )

    config = load_config(config_path)

    assert config.model.antibody_encoder.cache_dir == antibody_cache
    assert config.model.antigen_encoder is not None
    assert config.model.antigen_encoder.tokenizer_revision == "ag-tokenizer-456"
    assert config.model.interaction.kind == "concat"
    assert config.model.interaction.num_layers == 0
    assert config.model.objective.sigma == pytest.approx(2.0)


def test_cached_config_rejects_floating_revision(tmp_path, existing_train_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    model = {
        "antibody_encoder": {
            "name": "igbert",
            "revision": "main",
            "mode": "frozen_cached",
            "embedding_layer": -1,
            "cache_dir": str(cache_dir),
            "max_length": None,
            "long_sequence_strategy": "error",
        },
        "antigen_encoder": None,
        "interaction": {
            "kind": "antibody_only", "d_model": 16, "num_layers": 0,
            "num_heads": 1, "ffn_multiplier": 4.0, "dropout": 0.1,
            "pooling": "masked_mean", "bidirectional": False,
        },
        "objective": {
            "name": "pairwise_ranknet", "temperature": 1.0,
            "sigma": 1.0, "pointwise_loss": "huber",
        },
    }
    config_path = _write_config(tmp_path, {"model": model}, train_path=existing_train_path)

    with pytest.raises(ValueError, match="immutable"):
        load_config(config_path)


def test_concat_config_requires_antigen_encoder(tmp_path, existing_train_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    model = {
        "antibody_encoder": {
            "name": "igbert", "revision": "rev-1", "mode": "frozen_cached",
            "embedding_layer": -1, "cache_dir": str(cache_dir), "max_length": None,
            "long_sequence_strategy": "error",
        },
        "antigen_encoder": None,
        "interaction": {
            "kind": "concat", "d_model": 16, "num_layers": 0,
            "num_heads": 1, "ffn_multiplier": 4.0, "dropout": 0.1,
            "pooling": "masked_mean", "bidirectional": True,
        },
        "objective": {
            "name": "pairwise_ranknet", "temperature": 1.0,
            "sigma": 1.0, "pointwise_loss": "huber",
        },
    }
    config_path = _write_config(tmp_path, {"model": model}, train_path=existing_train_path)

    with pytest.raises(ValueError, match="requires model.antigen_encoder"):
        load_config(config_path)


def test_load_config_missing_file(tmp_path):
    # 测试文件不存在时，会怎么样
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_load_config_train_path_must_exist(tmp_path, existing_train_path):
    # 测试data_config:train_path 必须存在，valid_path 只要声明就要存在，不声明写None
    overrides = {
        "data": {
            "train_path": str(tmp_path / "missing.parquet"),
            "valid_path": None,
            "max_pairs_per_group": 50,
            "seed": 0,
        }
    }
    config_path = _write_config(tmp_path, overrides, train_path=existing_train_path)

    with pytest.raises(FileNotFoundError):
        load_config(config_path)


def test_load_config_valid_path_must_exist(tmp_path, existing_train_path):
    # valid_path 只要声明就要存在，不声明写None
    missing_valid = tmp_path / "valid_missing.parquet"
    config_path = _write_config(
        tmp_path, {}, train_path=existing_train_path, valid_path=missing_valid
    )

    with pytest.raises(FileNotFoundError):
        load_config(config_path)


def test_load_config_missing_required_field_raises_value_error(tmp_path, existing_train_path):
    # config必须要存在seed字段
    config_path = _write_config(tmp_path, {}, train_path=existing_train_path)
    raw = yaml.safe_load(config_path.read_text())
    del raw["data"]["seed"]
    config_path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="seed"):
        load_config(config_path)


def test_load_config_missing_section_raises_value_error(tmp_path, existing_train_path):
    # config必须包含model config
    config_path = _write_config(tmp_path, {}, train_path=existing_train_path)
    raw = yaml.safe_load(config_path.read_text())
    del raw["model"]
    config_path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="model"):
        load_config(config_path)


def test_load_config_does_not_apply_hidden_seed_default(tmp_path, existing_train_path):
    # 种子可以换，但不能一直用默认种子
    overrides = {
        "data": {
            "train_path": str(existing_train_path),
            "valid_path": None,
            "max_pairs_per_group": 50,
            "seed": 12345,
        }
    }
    config_path = _write_config(tmp_path, overrides, train_path=existing_train_path)

    config = load_config(config_path)

    assert config.data.seed == 12345


def test_load_config_parses_large_group_pair_sampling_options(tmp_path, existing_train_path):
    overrides = {
        "data": {
            "train_path": str(existing_train_path),
            "valid_path": None,
            "max_pairs_per_group": 200,
            "seed": 0,
            "large_group_threshold": 5000,
            "pair_enumeration_limit": 20000,
            "label_block_count": 4,
            "intra_block_pairs_per_large_group": 25,
            "discrete_label_unique_threshold": 16,
            "discrete_label_ratio_threshold": 0.1,
        }
    }
    config_path = _write_config(tmp_path, overrides, train_path=existing_train_path)

    config = load_config(config_path)

    assert config.data.large_group_threshold == 5000
    assert config.data.pair_enumeration_limit == 20000
    assert config.data.label_block_count == 4
    assert config.data.intra_block_pairs_per_large_group == 25
    assert config.data.discrete_label_unique_threshold == 16
    assert config.data.discrete_label_ratio_threshold == pytest.approx(0.1)


def test_load_config_parses_record_filter(tmp_path, existing_train_path):
    overrides = {
        "data": {
            "train_path": str(existing_train_path),
            "valid_path": None,
            "max_pairs_per_group": 50,
            "seed": 0,
            "filter": {
                "include_dataset_ids": ["studyA/tableA"],
                "include_antigen_keys": ["agA"],
                "require_antigen_sequence": True,
                "min_unique_labels_per_group": 2,
            },
        }
    }
    config_path = _write_config(tmp_path, overrides, train_path=existing_train_path)

    config = load_config(config_path)

    assert config.data.record_filter.include_dataset_ids == ("studyA/tableA",)
    assert config.data.record_filter.include_antigen_keys == ("agA",)
    assert config.data.record_filter.require_antigen_sequence is True
    assert config.data.record_filter.min_unique_labels_per_group == 2


def test_load_config_not_a_mapping(tmp_path):
    config_path = tmp_path / "invalid_format.yaml"
    config_path.write_text(yaml.safe_dump(["not", "a", "dict"])) # 写入一个列表
    # 再检查文件有没有崩溃之前，这个程序就崩溃了
    with pytest.raises(ValueError, match="top-level mapping"):
        load_config(config_path)
