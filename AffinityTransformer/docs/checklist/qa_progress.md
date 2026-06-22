# 质检进度跟踪

快照时间：2026-06-20，HEAD = `01dcab6`。按实际文件目录结构列出，不分优先级、不分层——勾上 = 这次会话里读过这个文件，没勾 = 还没看。

**2026-06-22 更新**：这轮会话（实现 `noise_aware_multiscale` + tau 注册表 + 接入 `build_pairs`）新增/改动了下面这些文件。新增的标了「新增」；原来打勾、内容已经变了的改回未勾并标「已改动，需重新质检」（不是从没看过，是看过的版本过期了）；原来就没打勾、这轮又改了的标了「本轮也改动」，避免漏掉。本仓库在这个沙箱里看不到 git，没法给新 HEAD。

**2026-06-22 再更新**：同一天另一轮会话又改了 `splits.py`，新增了两套切分协议——`build_within_antigen_split`（3.2节辅助协议，按做法一：每个group各自独立切，允许同一抗体跨不同group出现在不同集合，理由见函数docstring）和 `antigen_cluster_holdout_split`（新的主协议，按抗原序列相似度聚类切分，取代`group_holdout_split`的主协议地位）。新增了 `antigen_clustering.py`（抗原聚类，用average/complete链接法，不用single，理由见模块docstring）。`scripts/data/build_splits.py` 也跟着改了，加了三个新CLI参数和两个新的strategy分支。这几个文件之前打勾看过的版本已经过期，下面标记已改回未勾。

## affinity_transformer/

- [x] `config.py` （已改动，需重新质检：新增 6 个 `noise_aware_*` 字段 + 解析逻辑）
- [ ] `antigen_clustering.py` （新增：抗原序列聚类，给 `antigen_cluster_holdout_split` 用）
- [ ] `dataloader.py`
- [x] `metrics.py`
- [ ] `record_filter.py`
- [ ] `splits.py` （已改动，需重新质检：新增 `build_within_antigen_split`/`antigen_cluster_holdout_split` 两套协议，`group_holdout_split` 降级为备用项）
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
- [x] `cached.py` （本轮也改动：`load_trainable_records`三处调用都多传了`config`）
- [x] `cross_validation.py`
- [x] `data.py` （已改动，需重新质检：`load_trainable_records`新增`config`参数，接入`record_filter`——之前你打勾看的是没有这处改动的版本）
- [ ] `evaluation.py`
- [ ] `loaders.py` （本轮也改动：新增`_load_trainable_records_for_loader`，给在线模式接入`record_filter`）
- [x] `online.py`
- [x] `samplers.py`

## 根目录

- [ ] `train.py`
- [ ] `predict.py`
