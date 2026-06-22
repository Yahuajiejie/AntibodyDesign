# AffinityTransformer 工程说明

本文档说明 AffinityTransformer 的任务边界、数据契约、切分协议、代码结构和验收要求。科学问题以项目 README 为准，工程状态以当前仓库代码为准。

文中使用三种状态：

| 状态 | 含义 |
| --- | --- |
| 已接入 | 已进入公开入口或训练主链路，并有相应测试 |
| 部分接入 | 底层代码已经存在，但协议、配置、审计或端到端流程尚未完整 |
| 规划中 | README 已提出，但当前代码还不能执行 |

工程文档不能把“底层函数已经写好”直接写成“项目已经支持”。只有数据入口、配置、训练、评估、产物和测试形成闭环，才算完整接入。

## 1. 项目目标与范围

本项目根据抗体序列和抗原信息，为候选抗体输出排序分数：

```text
score = f(antibody, antigen)
```

这个分数没有 Kd、IC50 或 EC50 等物理单位，只用于同一可比实验背景内的相对排序。

项目采用排序学习，主要是因为不同研究的指标、单位和实验条件并不统一。将它们强行合并成全局回归目标，模型容易学到数据来源和标签尺度，而不是可以迁移的抗体—抗原匹配关系。

新版 README 将泛化问题分成三层：

1. 已见实体之间的比较补全；
2. 已知抗原上的新抗体排序；
3. 未见抗原与新抗体排序。

三层结果必须分别报告。一个模型在关系补全上表现良好，不能据此宣称它能处理新抗体或新抗原。

本仓库当前聚焦序列模型。交叉注意力是序列特征交互，不是三维复合物结构模拟。结构模型位于独立项目路线，不属于本仓库的实现范围。

当前抗原输入只有精确序列路径。README 中的 MSA、ESM-MSA 和 Exact + MSA 属于研究方案，尚未在本仓库实现。抗体专用编码器目前接入的是 IgBERT；README 提到的 AbLang-2 尚未接入。

## 2. 数据单位与实体键

### 2.1 record、group 和 pair

| 单位 | 含义 | 作用 |
| --- | --- | --- |
| record | 一条抗体—抗原测量记录 | 数据清洗、实体规范化、切分和逐条预测 |
| group | 一组可以互相比较排名的 records | 决定训练样本的可比范围和 Spearman 的计算范围 |
| pair | 从同一 group 的两条 records 派生出的比较样本 | RankNet 训练 |

`group_id` 回答“哪些 records 可以比较”，split key 回答“验证时哪些实体必须保持未见”。两者不是同一个概念。

同一抗原可能因为 assay、metric、study 或数据来源不同而形成多个 group。因此，group holdout 不能自动解释为未见抗原评估。

### 2.2 科学协议需要的实体键

新版 README 为切分和审计定义了更完整的实体层。当前实现情况如下。

| 字段或实体 | 用途 | 当前状态 |
| --- | --- | --- |
| `record_id` | 标识原始记录 | 已进入标准表 |
| `measurement_family_id` | 归并技术重复、同一样本或派生测量 | 规划中 |
| `antibody_sequence_key` | 标识规范化后的精确抗体序列 | 可通过 `antibody_sequence_hashes` 临时计算，未进入标准表 |
| `antibody_cluster_id` | 隔离相同或近似抗体序列 | 规划中 |
| `antigen_sequence_key` | 标识规范化后的精确抗原序列 | 规划中；当前主要使用 `antigen_key` |
| `antigen_cluster_id` | 隔离相同或近似抗原序列 | 部分接入；由独立聚类表生成，不是标准表字段 |
| `interaction_key` | 标识规范化后的抗原—抗体组合 | 规划中 |
| `group_id` | 标识排序可比性分组 | 已进入标准表 |

`antibody_sequence_hashes` 计算的是重链、轻链和单链字段的精确哈希。它可以识别完全相同的输入序列，但不能代替抗体序列聚类。

聚类 ID 属于派生数据。任何聚类产物都必须和算法、阈值、输入数据版本及哈希一起保存，否则相同的 `cluster_id` 名字不能证明两次实验使用了相同的实体划分。

## 3. 数据切分协议

切分协议共同遵守以下原则。

除专门的 pair holdout 诊断外，训练和评估必须先切 records，再在各 split 内构造训练样本：

```text
原始 records
  -> 实体规范化和聚类
  -> 按评估协议切分 records
  -> 在 train / valid / test 内分别构造 pointwise、pairwise 或 listwise 视图
```

普通训练不得先生成全部 pairs，再把 pairs 随机分到不同集合。否则同一 record 会通过不同配对对象同时进入训练和验证。

### 3.1 Pair holdout

Pair holdout 允许训练集和验证集共享抗原、抗体和 record，只要求完整比较关系不重复：

```text
(antigen_key, min(antibody_i, antibody_j), max(antibody_i, antibody_j))
```

它测试的是已见实体之间的关系补全，只适合检查排序损失、pair 方向和训练流程。结果不能解释为新抗体或新抗原泛化。

当前代码没有实现独立的 pair holdout 构造器和 pair-level 泄露报告。`debug_record_split` 是随机切 records，不是 pair holdout，二者不能混用名称。

状态：规划中。

### 3.2 Within-antigen antibody holdout

科学协议允许抗原和 group 跨集合，但要求验证抗体在训练中保持未见。目标约束包括：

- `record_id` 不重叠；
- `measurement_family_id` 不重叠；
- `interaction_key` 不重叠；
- `antibody_sequence_key` 不重叠；
- `antibody_cluster_id` 不重叠；
- 抗体簇隔离应在整个数据集范围内成立，不只在单个 group 内成立。

当前 `build_within_antigen_split` 的实现比上述协议宽松。它在每个 `group_id` 内独立按精确抗体序列哈希切分，并保证同一 group 内的同一精确抗体不跨 train、valid、test；同一抗体仍可通过另一个 group 出现在不同集合。它没有抗体聚类，也没有 `measurement_family_id` 和 `interaction_key` 审计。

该函数会把无法稳定切分的小 group 全部放入 train，并输出 `pinned_groups.csv`。`min_eval_records` 控制一个 group 至少要给 valid 和 test 各贡献多少条记录。

因此，当前实现应称为“group-local exact-antibody holdout”，不能直接当作 README 定义的完整 within-antigen antibody holdout。

状态：部分接入。

### 3.3 Dual cold-start

主科学协议要求验证集中的抗原簇和抗体簇都不出现在训练集中，同时隔离测量重复和抗原—抗体组合：

- `antigen_cluster_id` 不重叠；
- `antibody_cluster_id` 不重叠；
- `interaction_key` 不重叠；
- `measurement_family_id` 不重叠。

当前 `antigen_cluster_holdout_split` 只按 `antigen_cluster_id` 切分。它能够阻止近似抗原分散到不同集合，但没有隔离抗体簇，也没有测量家族和 interaction 审计。

因此，该策略是 antigen-cluster holdout，不是完整的 dual cold-start。只有补齐抗体聚类和其余实体审计后，才可以把结果命名为 dual cold-start。

状态：部分接入。

### 3.4 当前切分入口

| 入口或策略 | 实际行为 | 可用于说明什么 |
| --- | --- | --- |
| `debug_record_split` | 随机切分 records，record 不重叠 | 流程调试 |
| `group_holdout_split` | 同一 `group_id` 整体进入一个集合 | 未见 group |
| `build_within_antigen_split` | 每个 group 内按精确抗体哈希切分 | group-local 新抗体 |
| `antigen_cluster_holdout_split` | 按抗原序列簇切分 | 未见抗原簇 |
| `build_group_kfolds` | 按 `group_id` 做 K 折 | group-level 交叉验证 |

`scripts/data/build_splits.py` 已能离线生成 group、within-antigen 和 antigen-cluster 切分。训练 YAML 的自动切分目前只接受 `debug_record_split` 和 `group_holdout_split`；新策略尚未接入 `config.py`、`training/data.py` 和 K 折流程。

现阶段使用新策略训练时，应先通过脚本生成固定 parquet，再在 YAML 中设置 `split_strategy: none` 并显式填写 train、valid、test 路径。

## 4. 抗原聚类

`affinity_transformer/antigen_clustering.py::compute_antigen_clusters` 根据 `antigen_sequence` 生成 `antigen_cluster_id`。

当前算法：

1. 要求一个 `antigen_key` 只能对应一条非空抗原序列；
2. 先按序列长度分桶；
3. 只在同长度桶内计算归一化 Hamming 距离；
4. 使用 average 或 complete linkage 做层次聚类；
5. 禁止 single linkage，避免近邻链把大量样本串成一个大簇。

当前限制：

- 长度不同的序列完全不比较，即使只相差一个插入或缺失也不会进入同一簇；
- 同长度大桶仍需要成对距离和层次聚类，时间与内存开销需要在真实全量数据上检查；
- 函数默认相似度阈值为 `0.75`，CLI 默认值为 `0.9`，两处尚未统一；
- `antigen_clusters.csv` 目前没有同时保存阈值、linkage、输入文件哈希和代码版本；
- 聚类编号只在同一次聚类结果内部有意义，不能脱离元数据跨实验比较。

正式实验必须显式指定阈值和 linkage，并保存聚类元数据。阈值应根据当前数据的簇大小分布选择，不能直接照搬其他数据集。

## 5. 切分审计

不同协议允许的重叠不同，审计结果必须按协议解释。

| 检查项 | Pair holdout | Within-antigen | Dual cold-start |
| --- | --- | --- | --- |
| 完整 comparison pair | 必须为零 | 必须为零 | 必须为零 |
| `record_id` | 允许 | 必须为零 | 必须为零 |
| `measurement_family_id` | 允许，仅作诊断 | 必须为零 | 必须为零 |
| 精确抗体序列 | 允许 | 必须为零 | 必须为零 |
| `antibody_cluster_id` | 允许 | 必须为零 | 必须为零 |
| 精确抗原序列 | 允许 | 允许 | 必须为零 |
| `antigen_cluster_id` | 允许 | 允许 | 必须为零 |
| `interaction_key` | 允许已见实体，但完整比较关系不能重复 | 必须为零 | 必须为零 |
| `group_id` | 允许 | 允许 | 应为零 |

除重叠检查外，还应报告各 split 的 study、assay、metric、label kind、group 大小和记录数分布。分布差异不一定是泄露，但会影响指标解释。

当前代码的审计范围：

- `debug_record_split`：检查 `record_id`；
- `group_holdout_split`：检查 `record_id` 和 `group_id`；
- `build_within_antigen_split`：报告中只检查 `record_id`，同 group 内精确抗体隔离由构造过程保证但未独立审计；
- `antigen_cluster_holdout_split`：检查 `record_id`、`group_id` 和 `antigen_cluster_id`。

这套审计还不足以满足新版 README 的完整协议。

## 6. 代码分层与目录

```text
AffinityTransformer/
├── affinity_transformer/
│   ├── config.py                  # YAML 配置解析和校验
│   ├── record_filter.py           # 标准表过滤和精确抗体序列哈希
│   ├── antigen_clustering.py      # 抗原序列聚类
│   ├── splits.py                  # 固定切分、辅助切分、K 折和泄露报告
│   ├── metrics.py                 # 分组 Spearman 及汇总
│   ├── trainer.py                 # 通用训练循环
│   ├── user_entry.py              # 对外推理接口
│   ├── dataset/                   # records、groups、pairs 和 Dataset
│   ├── embeddings/                # 冻结编码器及离线缓存
│   ├── model/                     # 投影、交互、池化和打分网络
│   └── training/                  # loader、runner、评估和产物
├── configs/                       # 数据、模型和实验配置
├── scripts/
│   ├── prepare/                   # 原始数据转换和质检
│   ├── data/                      # 过滤、检查和切分
│   ├── embeddings/                # embedding 缓存构建
│   ├── experiments/              # 批量实验和结果汇总
│   ├── runs/                      # 本地实验组入口
│   └── slurm/                     # 集群任务脚本
├── tests/                         # 单元测试和集成测试
├── train.py                       # 训练入口
├── predict.py                     # 推理入口
└── docs/                          # 工程说明和专题分析
```

代码分三层：

- 数据准备层处理原始字段、单位、标签方向和单个数据集的特殊规则；
- 通用框架层只消费标准表，不解析某篇论文的原始 CSV；
- 实验入口层读取配置并组织流程，不重新实现 dataset、model 或 metric 的内部逻辑。

`processed/` 和 `outputs/` 是生成产物目录，不应提交大数据、embedding 或 checkpoint。

## 7. 标准数据表

`affinity_transformer/dataset/schema.py` 定义当前必需字段。

| 类别 | 字段 |
| --- | --- |
| 记录来源 | `record_id`, `dataset_id`, `study_id`, `table_id`, `source_file`, `source_row` |
| 抗体 | `antibody_id`, `antibody_type`, `heavy_chain`, `light_chain`, `single_chain_sequence` |
| 抗原 | `antigen_key`, `antigen_name`, `antigen_sequence`, `antigen_source` |
| 实验 | `assay_name`, `assay_type`, `metric_name`, `metric_unit` |
| 标签 | `metric_value_raw`, `metric_value_numeric`, `metric_direction`, `transform_rule`, `rank_label`, `label_kind` |
| 训练控制 | `group_id`, `keep_for_training`, `drop_reason` |

### 7.1 标签方向

`rank_label` 约定为数值越大越好。Kd、IC50、EC50 等“小值更优”的指标必须在各数据集的 `convert.py` 中完成方向转换。

方向统一只保证组内排序语义一致，不表示不同 metric 可以跨 group 比较。框架层不会再次猜测标签方向，因此转换规则必须对照原始论文和原始数据人工核查。

### 7.2 group_id

`group_id` 是排序可比性单位。常见形式为：

```text
{study_id}/{table_id}/{antigen_key}/{metric_name}/{label_kind}
```

同一 group 内应具有一致的抗原背景、指标含义和实验条件。pairwise 和 listwise 样本不跨 group 构造。

### 7.3 可训练记录

`load_records` 检查标准列是否齐全。`filter_trainable_records` 只保留：

- `keep_for_training` 为真；
- `rank_label` 为有限数。

它不会重新检查抗体序列是否合法。重链存在性和序列字符主要由 `validate_processed_table.py` 保证，不能绕过数据质检后指望训练层自动兜底。

数据准备质检允许序列中出现 `X`，而 `utils.validate_amino_acid_sequence` 只接受 20 种标准氨基酸。两处规则尚未统一。

## 8. 数据准备与运行流程

```text
raw CSV
  -> scripts/prepare/binding/<study>/<table>/convert.py
  -> records.parquet
  -> scripts/prepare/validate_processed_table.py
  -> merge_records.py
  -> all_records.parquet
  -> scripts/data/filter_records.py
  -> 实体键和序列簇构建
  -> scripts/data/build_splits.py
  -> train / valid / test parquet + 审计产物
  -> embedding 缓存
  -> train.py
  -> 预测、分组指标和运行产物
```

所有 pair、list 或 pointwise 训练视图都应从已经切好的 records 生成。切分产物必须可以复用，不能在每次训练时悄悄换一套实体划分。

`validate_processed_table.py` 至少检查必需列、ID 唯一性、标签有限性、抗体序列字符、枚举值和来源行号。检查失败时必须返回非零退出码。

## 9. Pair 构造与采样

`dataset/pairs.py::build_pairs` 是 pairwise 样本的统一入口。它只在同一 `group_id` 内构造 pair，标签完全相同的记录不生成硬排序关系。

```text
label_i > label_j  -> y_ij = 1.0
label_i < label_j  -> y_ij = 0.0
```

当前可配置策略：

| 策略 | 说明 | 状态 |
| --- | --- | --- |
| `absolute_cap` | 每组最多抽取固定数量的 pair | 已接入 |
| `capped_proportional` | 按候选 pair 比例抽样并设置上下限 | 已接入 |
| `balanced_tree` | 按标签顺序构造平衡树 | 已接入 |
| `randomized_bst` | 构造随机二叉搜索树 | 已接入 |
| `noise_aware_multiscale` | 按测量噪声阈值过滤近似不可分 pair，并构造多尺度、度数受控的比较图 | 已接入 |

`noise_aware_multiscale` 已进入 `build_pairs`、配置、loader、测试和专用实验 YAML。它通过 `tau_registry.py` 按 `antigen_key` 推断数据来源和噪声阈值。这里存在三个需要保留的限制：

- `antigen_key` 只是数据来源的代理，不是显式 source 字段；
- 部分 tau 来自间接换算，默认 tau 没有充分实验依据；
- 当前只支持跳过无法跨越 tau 的记录，不支持软标签或置信度权重。

旧的 `noise_floor_tree.py` 只保留废弃说明，不再是可用策略。该文件末尾仍有“replacement 未接入”的过时描述，应以 `pairs.py` 和配置校验中的实际 dispatch 为准，并在后续代码整理时修正。

比较采样策略时，至少应报告 pair 数量、record coverage、标签差分布、degree 分布、构造时间、内存占用、pairwise accuracy、signed margin 和最终 group-level Spearman。当前训练产物没有自动汇总全部这些诊断项，仍需补充统一报告。

当 `weight_pairs_by_group_size` 开启时，训练按 group 的记录数和实际 pair 数计算权重。权重计算和 loader 构造会分别调用一次 `build_pairs`，两次调用的 seed 和采样参数必须完全一致。

## 10. Embedding 子系统

项目保留在线编码和离线缓存两条路径。

### 10.1 离线缓存

`frozen_cached` 模式先用冻结编码器生成逐残基 embedding，再训练交互和打分网络。

```text
<cache_dir>/
├── metadata.yaml
├── manifest.parquet
└── shards/
    └── shard_00000.pt
```

训练前会核对编码器名称、模型 revision、tokenizer revision、embedding 层、最大长度、长序列策略、序列覆盖率和 shard 内容。配置与缓存不一致时应在模型和优化器创建前报错。

ESM2 和 IgBERT 的重轻链输入方式不同，拼接规则由各自 extractor 决定。当前代码没有实现 README 所描述的统一“抗体类型标记 + 链边界标记”协议，不能在工程文档中把它写成已落地能力。

`ShardedEmbeddingStore` 会把所需 shard 载入内存并尝试使用 mmap。大缓存内存占用和多 worker 并发读取仍需要压力测试。

### 10.2 在线编码

在线路径在训练前向中运行编码器，目前主要服务历史 ESM2 配置和现有推理入口。在线路径和缓存路径使用不同模型构造流程，checkpoint 不能混用。

## 11. 模型与训练目标

`EmbeddingAffinityRanker` 支持三种模型结构：

- `antibody_only`；
- `concat`；
- `deep_cross_attention`。

当前 `run_cached_ranknet` 只接入 `concat` 和 `deep_cross_attention`。antibody-only 的缓存模型可以在 factory 层构造，但尚未接入缓存训练 runner；在线训练路径可以使用 antibody-only 配置。

缺失抗原通过 mask 处理。无抗原样本不会把占位向量当作有效 token 参与注意力。

科学方案包含 pointwise、pairwise 和 listwise 三类任务头。代码状态如下：

| 训练目标 | 底层损失和数据结构 | 训练 runner |
| --- | --- | --- |
| Pointwise | `pointwise_ranking_loss` 已实现 | 未接入 |
| Pairwise RankNet | pair Dataset、loader、loss 已实现 | 已接入，是当前主线 |
| Listwise ListNet | `build_groups`、Listwise Dataset、`listnet_loss` 已实现 | 未接入 |

配置层允许三种 objective，但 `Trainer` 只执行 `pairwise_ranknet`。pointwise 和 listwise 在 runner 接通、端到端测试通过前，只能写成部分实现，不能写成项目已经支持三类训练。

## 12. 配置与训练编排

`config.py` 同时兼容历史扁平 YAML 和嵌套模型 YAML。`train.py` 根据 `model.antibody_encoder.mode` 选择在线或缓存 runner，再根据 `cross_validation.enabled` 选择固定切分或 K 折。

```text
train.py
  -> load_config
  -> resolve_data_paths / run_group_kfold_cross_validation
  -> run_online_training 或 run_cached_ranknet
  -> 读取已经切好的 records
  -> 构造训练视图和 valid/test record loader
  -> Trainer.fit
  -> 分组评估
  -> 写出 checkpoint、预测和指标
```

自动切分、离线切分脚本和 K 折目前支持的 split key 不一致。新增协议时必须同时检查：

- `DataConfig` 和合法值校验；
- `scripts/data/build_splits.py`；
- `training/data.py`；
- K 折实现；
- 泄露报告；
- YAML 示例；
- 单元测试和端到端测试。

## 13. 评估指标

`compute_group_spearman` 按 `group_id` 计算 Spearman：

```python
label_rank = group["rank_label"].rank()
score_rank = group["score"].rank()
spearman = label_rank.corr(score_rank)
```

即先计算真实标签和预测分数的排名，再计算两个排名序列的 Pearson 相关系数。

真实标签少于两个不同值，或模型分数为常数时，组内 Spearman 为 `NaN`。该 group 仍保留在结果表中，用于统计跳过组数量。

汇总指标包括：

- `macro_spearman`：有效 group 的等权平均；
- `weighted_spearman`：按 `n_records` 加权的组内 Spearman 平均。

weighted Spearman 仍然是“组内相关系数的加权平均”，不是把所有 records 混在一起计算全局 Spearman。二分类与连续标签应按 `label_kind` 分开报告。

每个指标必须带上协议名称、split artifact、实体聚类参数和随机种子。pair holdout、within-antigen、antigen-cluster holdout 和 dual cold-start 的数字不能混成一个总分，也不能在表格中使用含糊的 `valid_spearman` 名称代替协议说明。

## 14. 推理接口

`user_entry.py` 和 `predict.py` 接收抗原序列及候选抗体，输出 `query_id`、`antibody_id`、`score`、`rank` 和 `model_name`。

当前推理接口只支持在线 ESM2 路径。缓存 embedding 路径训练出的 `EmbeddingAffinityRanker` checkpoint 尚不能自动为新抗原和新抗体生成 embedding 后完成推理。

完整推理流程需要：加载 checkpoint 和编码器配置、生成新序列 embedding、核对 metadata，再调用缓存模型打分。这仍是主要交付缺口。

## 15. 实验产物与可复现性

一次训练至少保留：

```text
checkpoint.pt
config.yaml
metrics.json
history.csv
run.log
predictions.csv
group_metrics.csv
```

配置 test 集时，还应输出 `test_predictions.csv` 和 `test_group_metrics.csv`。缓存路径另外写出 embedding metadata 和资源统计。

涉及实体切分的实验还应保留：

```text
train.parquet / valid.parquet / test.parquet
split_summary.csv
leakage_report.csv
pinned_groups.csv              # 若使用 within-antigen 辅助切分
antigen_clusters.csv           # 若使用抗原聚类
cluster_metadata.yaml          # 待补：阈值、linkage、输入哈希、代码版本
```

结果汇总必须记录配置名、随机种子、协议名、split key、模型结构、采样策略和指标定义。

## 16. 编码、测试与验收

公开函数应有 docstring，说明参数、返回值、异常和职责边界。私有 helper 使用下划线前缀，不从包的公开接口导出。

核心模块不得在 import 时下载模型或读取大文件。新增配置字段时，要同步修改 dataclass、解析、合法值校验、调用透传和测试。

提交训练相关改动前至少运行：

```bash
python -m py_compile train.py predict.py affinity_transformer/*.py \
    affinity_transformer/dataset/*.py \
    affinity_transformer/dataset/pair_sampling/*.py \
    affinity_transformer/embeddings/*.py \
    affinity_transformer/model/*.py \
    affinity_transformer/training/*.py
pytest -q
```

切分与采样改动还要单独验证：

- 输入顺序变化不影响相同 seed 的结果；
- train、valid、test 没有记录丢失或重复；
- 每种协议的必查实体都进入泄露报告；
- 小 group 和超大实体簇的行为可解释；
- 聚类阈值和方法写入产物；
- pair 构造不跨 split；
- loader 与 group weight 两次调用 `build_pairs` 的参数一致；
- sampler 的 record coverage、degree 和标签差满足设计约束。

## 17. 禁止项

1. 不在 `affinity_transformer/` 中解析某个数据集的原始 CSV、换算单位或翻转标签方向。
2. 不跨 `group_id` 构造排序样本。
3. 除明确命名的 pair holdout 诊断外，不先生成 pair 再随机切 pair。
4. 不把 `debug_record_split` 称为 pair holdout。
5. 不把 group holdout 称为未见抗原评估。
6. 不把 antigen-cluster holdout 称为 dual cold-start，除非抗体簇等约束也已满足。
7. 不把当前 group-local 的 within-antigen 实现解释为全局新抗体簇泛化。
8. 不把未接入 runner 的 pointwise、listwise、MSA 或 AbLang-2 写成已支持能力。
9. 不把二分类和连续标签混成一个无法解释的指标。
10. 不在 import 时加载预训练模型或大数据。
11. 不用隐藏默认值掩盖配置错误。
12. 不在同一次改动中同时大规模重构目录和改变训练算法。

## 18. 当前开发优先级

| 优先级 | 问题 | 验收目标 |
| --- | --- | --- |
| P0 | 主协议缺少抗体聚类、测量家族和 interaction 实体 | 完成实体键，输出真正的 dual cold-start split 和全量审计 |
| P0 | within-antigen 实现只做 group-local 精确序列隔离 | 改为全局 antibody cluster 隔离，或永久使用更准确的协议名称 |
| P0 | 新切分策略未接入训练配置和 K 折 | 配置、CLI、自动切分、K 折和产物使用同一协议定义 |
| P0 | 缓存模型没有新样本推理入口 | 打通在线生成 embedding 到 `EmbeddingAffinityRanker` 的推理流程 |
| P1 | 抗原聚类默认值不一致且缺元数据 | 统一默认行为并写出完整 `cluster_metadata.yaml` |
| P1 | pointwise/listwise 只有底层实现 | 补 loader、runner、端到端测试和同预算对照实验 |
| P1 | pair 采样缺统一诊断产物 | 自动输出 coverage、标签差、degree、时间和内存统计 |
| P1 | tau registry 部分依据较弱 | 保存每组解析到的 tau 和规则，补充直接测量依据 |
| P1 | mmap、group weight 和新切分缺完整环境测试 | 补并发、回归和真实小规模端到端测试 |
| P2 | 序列字符集对 `X` 的规则不一致 | 统一数据准备、训练和推理校验 |
| P2 | MSA、AbLang-2 和统一抗体标记仍是方案 | 实现前保持为规划中，不混入当前能力说明 |

本文件只维护当前有效的工程约定。历史讨论留在 Git 和专题分析文档中，不在正文开头罗列旧版本沿革。代码能力、科学协议或目录结构变化时，应直接更新对应章节。
