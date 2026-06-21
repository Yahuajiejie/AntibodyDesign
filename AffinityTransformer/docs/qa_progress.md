# 质检进度跟踪

快照时间：2026-06-20，HEAD = `01dcab6`。按实际文件目录结构列出，不分优先级、不分层——勾上 = 这次会话里读过这个文件，没勾 = 还没看。

**2026-06-22 更新**：这轮会话（实现 `noise_aware_multiscale` + tau 注册表 + 接入 `build_pairs`）新增/改动了下面这些文件。新增的标了「新增」；原来打勾、内容已经变了的改回未勾并标「已改动，需重新质检」（不是从没看过，是看过的版本过期了）；原来就没打勾、这轮又改了的标了「本轮也改动」，避免漏掉。本仓库在这个沙箱里看不到 git，没法给新 HEAD。

## affinity_transformer/

- [x] `config.py` （已改动，需重新质检：新增 6 个 `noise_aware_*` 字段 + 解析逻辑）
- [ ] `dataloader.py`
- [x] `metrics.py`
- [ ] `record_filter.py`
- [x] `splits.py`
- [ ] `trainer.py`
- [ ] `user_entry.py`
- [ ] `utils.py`

## affinity_transformer/dataset/

- [x] `__init__.py`
- [x] `datasets.py`
- [x] `examples.py`
- [x] `groups.py`
- [ ] `pairs.py` （已改动，需重新质检：加了 `noise_aware_multiscale` 分支 + `antigen_key` 必填校验）
- [x] `records.py`
- [x] `schema.py`

## affinity_transformer/dataset/pair_sampling/

- [ ] `__init__.py` （已改动，需重新质检：新增导出 `_noise_aware_multiscale_pairs`/`resolve_tau_for_group`）
- [x] `blocks.py`
- [ ] `common.py`
- [ ] `heap_tree.py`
- [x] `labels.py`
- [x] `large_group.py`
- [ ] `noise_aware_multiscale.py` （新增）
- [ ] `noise_floor_tree.py` （新增，已废弃：原型有单链聚类 bug，已被 `noise_aware_multiscale.py` 取代，内容清空只留说明，沙箱权限不让真删）
- [ ] `randomized_tree.py`
- [ ] `tau_registry.py` （新增）
- [ ] `tree.py`
- [x] `two_label.py`
- [ ] `validation.py` （本轮也改动：新增 `noise_aware_*` 参数校验）

## affinity_transformer/embeddings/

- [x] `__init__.py`
- [x] `collate.py`
- [x] `extractors.py`
- [x] `huggingface.py`
- [x] `pipeline.py`
- [x] `schema.py`
- [x] `store.py`
- [ ] `validation.py`

## affinity_transformer/model/

- [x] `__init__.py`
- [x] `attention.py`
- [x] `blocks.py`
- [x] `embedding_ranker.py`
- [x] `factory.py`
- [x] `heads.py`
- [x] `interaction.py`
- [x] `losses.py`
- [x] `pooling.py`
- [x] `projections.py`
- [x] `ranker.py`

## affinity_transformer/training/

- [ ] `__init__.py`
- [x] `artifacts.py`
- [ ] `cached.py` （本轮也改动：`load_trainable_records`三处调用都多传了`config`）
- [ ] `cross_validation.py`
- [ ] `data.py` （已改动，需重新质检：`load_trainable_records`新增`config`参数，接入`record_filter`——之前你打勾看的是没有这处改动的版本）
- [ ] `evaluation.py`
- [ ] `loaders.py` （本轮也改动：新增`_load_trainable_records_for_loader`，给在线模式接入`record_filter`）
- [ ] `online.py`
- [x] `samplers.py`

## 根目录

- [ ] `train.py`
- [ ] `predict.py`
