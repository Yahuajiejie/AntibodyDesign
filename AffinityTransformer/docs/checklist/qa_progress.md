# 质检进度跟踪

快照时间：2026-06-20，HEAD = `01dcab6`。按实际文件目录结构列出，不分优先级、不分层——勾上 = 这次会话里读过这个文件，没勾 = 还没看。

**2026-06-22 更新**：这轮会话（实现 `noise_aware_multiscale` + tau 注册表 + 接入 `build_pairs`）新增/改动了下面这些文件。新增的标了「新增」；原来打勾、内容已经变了的改回未勾并标「已改动，需重新质检」（不是从没看过，是看过的版本过期了）；原来就没打勾、这轮又改了的标了「本轮也改动」，避免漏掉。本仓库在这个沙箱里看不到 git，没法给新 HEAD。

**2026-06-22 再更新**：同一天另一轮会话又改了 `splits.py`，新增了两套切分协议——`build_within_antigen_split` 和 `antigen_cluster_holdout_split`。其中 `build_within_antigen_split` 现已改为按 antigen context 聚合后在 context 内切抗体单位/component，不再是每个 group 各自独立切的做法一；`antigen_cluster_holdout_split` 仍按抗原序列相似度聚类切分。新增了 `antigen_clustering.py`（抗原聚类，用average/complete链接法，不用single，理由见模块docstring）。`scripts/data/build_splits.py` 也跟着改了，加了三个新CLI参数和两个新的strategy分支。这几个文件之前打勾看过的版本已经过期，下面标记已改回未勾。

**2026-06-23 更新**：当前审查范围收窄为“四种协议从数据处理到 split 导出”：
`group_holdout`、`antibody_cold_start`、`antigen_cold_start`、`dual_cold_start`。
pair holdout 暂不进入本轮审查；训练、embedding cache、checkpoint 和对照实验也先不进入本轮审查。
这一轮重点不再是单个 `splits.py` 文件，而是下面这条流水线是否能在真实 `all_records.parquet` 上跑通：

```text
all_records.parquet
→ entity_annotations.parquet
→ 可选 representation_annotations.parquet
→ 四个协议目录
→ train.parquet / valid.parquet / test.parquet / audit artifacts
```

因此待审查文件清单需要新增 `annotations/`、`splitting/` 子包、三条 cold-start 专用脚本，以及对应测试。
旧的 `scripts/data/build_splits.py --strategy antibody_cold_start_split/antigen_cold_start_split`
仍需看作兼容入口；canonical 协议导出优先审查三条专用脚本。

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

## affinity_transformer/annotations/

本轮新增为四协议 split 导出的必查项。重点看 annotation 是否保持窄表、是否只通过 `record_id` 临时 join、是否不会污染 base records schema。

- [x] `__init__.py`
- [x] `io.py`

## affinity_transformer/splitting/

本轮新增为四协议 split 导出的核心审查范围。重点看协议语义、component 构造、leakage audit、artifact writer 和 facade 兼容关系。

- [x] `__init__.py`
- [x] `common.py`
- [x] `dispatch.py`
- [x] `results.py`
- [x] `audits.py` （除非数据表出现奇怪问题，否则不检查）
- [x] `debug.py`
- [x] `group.py`
- [ ] `antigen_cluster.py`
- [ ] `within_antigen.py`
- [x] `entity_cold_start.py`
- [ ] `dual_cold_start.py`

## affinity_transformer/dataset/

- [x] `__init__.py`
- [x] `datasets.py`
- [x] `examples.py`
- [x] `groups.py`
- [ ] `pairs.py` （已改动，需重新质检：加了 `noise_aware_multiscale` 分支 + `antigen_key` 必填校验）
- [x] `records.py`
- [x] `schema.py`

## affinity_transformer/dataset/pair_sampling/

- [x] `__init__.py` （已改动，需重新质检：新增导出 `_noise_aware_multiscale_pairs`/`resolve_tau_for_group`）
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
- [x] `validation.py`

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

## scripts/data/

本轮只审查“数据处理到 split 导出”相关脚本，不审训练脚本。

- [ ] `inspect_records.py`
- [ ] `filter_records.py`
- [ ] `build_splits.py` （兼容入口；注意它的 entity cold-start 分支和专用脚本的产物契约不完全一样）
- [ ] `build_antibody_cold_start_split.py` （canonical antibody cold-start 导出入口）
- [ ] `build_antigen_cold_start_split.py` （canonical antigen cold-start 导出入口）
- [ ] `build_dual_cold_start_split.py` （canonical dual cold-start 导出入口）

## scripts/prepare/binding/

- [ ] `merge_records.py` （确认 `all_records.parquet` 的生成、`record_id` 稳定性和 summary 产物）
- [ ] `gen_manifest.py` （只审与 label_kind / dataset metadata 进入 records 相关的部分）

## scripts/runs/ 与 scripts/slurm/

- [ ] `scripts/runs/g00_qc_and_splits.sh` （目前主要跑 group holdout；需确认是否仍代表当前四协议目标）
- [ ] `scripts/slurm/g00_qc_and_splits.sbatch`
- [ ] `scripts/slurm/submit_g00_g01_chain.sh`

## tests/：四协议 split 相关

- [ ] `tests/test_splits.py`
- [ ] `tests/test_entity_cold_start_splits.py`
- [ ] `tests/test_antibody_cold_start_annotations.py`
- [ ] `tests/test_antigen_cold_start_annotations.py`
- [ ] `tests/test_dual_cold_start.py`
- [ ] `tests/test_splits_facade_compat.py`
- [ ] `tests/test_antigen_clustering.py`
- [ ] `tests/test_splits_within_antigen.py`
- [ ] `tests/test_record_filter.py`

## 当前缺失、后续新增后必须审查的文件

这些文件目前不是完整实现清单，而是四协议真实数据全流程要补齐的生产入口。新增后必须加入本表。

- [ ] `scripts/data/build_entity_annotations.py`（缺失：从 `all_records.parquet` 生成 `entity_annotations.parquet`）
- [ ] `scripts/data/build_protocol_splits.py`（缺失：一键导出 group / antibody / antigen / dual 四个协议目录）
- [ ] `scripts/data/build_antibody_clusters.py`（缺失：正式抗体 cluster 窄表与 manifest）
- [ ] `scripts/data/build_antigen_clusters.py`（缺失：正式抗原 cluster 窄表与 manifest；可先复用现有 `antigen_clustering.py` 能力）
- [ ] `scripts/data/build_representation_annotations.py`（缺失：需要 effective-input audit 时生成 `representation_annotations.parquet`）

## 根目录

- [ ] `train.py`
- [ ] `predict.py`
