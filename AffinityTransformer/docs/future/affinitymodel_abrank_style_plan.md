# affinitymodel：AbRank-style 训练流程修改方案

本文档定义 `affinitymodel/` 目录下的新实验版本。它不是对原
`affinity_transformer/` 主线的就地重构，而是一个更接近 AbRank 思路的
ranking 训练分支。

核心目标：保留现有 raw → records → all_records 的数据准备成果，但在
`all_records.parquet` 之后重新定义训练用的轻量数据表、序列聚类、切分、
pair 构造和分阶段训练流程。

## 1. 总原则

1. 不修改原始 converter 的职责边界。
2. 不把所有 label 强行当成同一种物理量。
3. `all_records.parquet` 仍然作为输入事实表，不在第一阶段修改。
4. `affinitymodel/` 内部只消费从 `all_records.parquet` 派生出的窄表。
5. predicted、experimental、binary 不再一锅炖。
6. 主评估以 experimental 数据为准，predicted 主要作为弱监督预训练。

## 2. 新流程总览

```text
raw table / fasta / method note
  -> per-table records.parquet                    # 保留现有流程
  -> all_records.parquet                          # 保留现有流程
  -> affinitymodel dataset table                  # 新增：训练用窄表
  -> antibody / antigen sequence clustering        # 新增
  -> split by selected protocol                    # 新增/重写
  -> pair/list dataset construction                # 新增/重写
  -> staged training / validation / testing         # 新增/重写
```

原始 `all_records.parquet` 保留完整字段。`affinitymodel` 的训练表只保留模型
与构造 pair 必需的信息，避免继续把 converter、annotation、split 和 trainer
揉在一起。

## 3. affinitymodel dataset table

### 3.1 输入

```text
processed/binding/all_records.parquet
```

### 3.2 输出

建议输出：

```text
processed/affinitymodel/datasets/base_affinity_table.parquet
processed/affinitymodel/datasets/base_affinity_table_manifest.json
processed/affinitymodel/datasets/base_affinity_table_qc.csv
```

### 3.3 保留字段

第一版只保留：

```text
record_id
dataset_id
study_id
table_id
source_file
source_row

heavy_chain
light_chain
single_chain_sequence
antibody_type

antigen_key
antigen_name
antigen_sequence
antigen_source

assay_name
assay_type
metric_name
metric_unit
metric_value_raw
metric_value_numeric
metric_direction
rank_label
label_kind
group_id
keep_for_training
drop_reason
```

### 3.4 派生字段

在该窄表中新增：

```text
metric_family
metric_scale
affinity_score
score_direction
supervision_tier
```

含义：

| 字段 | 含义 |
| --- | --- |
| `metric_family` | `kd`、`ic50`、`ec50`、`predicted_affinity`、`binary`、`other` |
| `metric_scale` | 例如 `M`、`nM`、`ugml`、`dimensionless` |
| `affinity_score` | 越大表示越强；不保证所有 family 之间物理可比 |
| `score_direction` | 第一版统一为 `higher_is_better` |
| `supervision_tier` | `experimental_affinity`、`predicted_affinity`、`binary`、`other` |

### 3.5 第一版保留策略

`affinitymodel` 第一版保留：

```text
metric_family in {kd, ic50, ec50, predicted_affinity}
```

暂时排除：

```text
binary
rel_binding_signal
escape
fitness
log(Kd_ratio)
other
```

但 binary 不删除出事实表；只是不进入第一版 ranking 训练。

## 4. 指标归一化规则

### 4.1 KD

KD 是最干净的跨表亲和力指标。统一目标为：

```text
affinity_score = -log10(KD_M)
metric_family = kd
metric_scale = M
```

当前需要特别处理：

```text
neg_log10_kd_M  -> 已是 -log10(KD_M)
neg_log10_kd_nM -> 如果确认为 -log10(KD_nM)，则 affinity_score = rank_label + 9
```

`neg_log10_kd_nM` 在进入跨组 pair 前必须审查 converter；不能假设已经正确转成 M。

### 4.2 IC50

IC50 不是 KD，但可以作为 affinity/effect proxy。

第一版：

```text
metric_family = ic50
affinity_score = 当前 rank_label
```

不把 IC50 强行换算成 KD。IC50 与 KD 的混合 pair 必须单独开关，并在结果中单独报告。

### 4.3 EC50

EC50 与 IC50 类似，作为 effect proxy。

第一版：

```text
metric_family = ec50
affinity_score = 当前 rank_label
```

不与 KD 默认混合。

### 4.4 predicted_affinity

当前 `pred_affinity` 来自：

```text
engelhart2022dataset/scFv-SARS-CoV-2_affinity
li2023machine/affinity1
li2023machine/affinity2
```

本地 converter 记录：

```text
metric_unit = predicted score (dimensionless)
transform_rule = rank_label = Pred_affinity
metric_direction = higher_is_better
```

因此：

```text
metric_family = predicted_affinity
metric_scale = dimensionless
affinity_score = rank_label
```

predicted affinity 不能直接当成 `-log10(KD_M)`，也不能默认和 experimental
KD/IC50/EC50 组成跨指标 pair。它第一阶段作为弱监督预训练数据使用。

## 5. 数据分层

训练数据分为至少五层：

| 层 | 数据 | 用途 |
| --- | --- | --- |
| P1 | `engelhart2022dataset` predicted affinity | predicted 预训练 |
| P2 | `li2023machine/affinity1` predicted affinity | predicted 预训练 |
| P3 | `li2023machine/affinity2` predicted affinity | predicted 预训练 |
| E1 | experimental KD | 主微调与主评估 |
| E2 | experimental IC50/EC50 | 扩展微调与辅助评估 |

注意：不同 predicted 数据集的分数未必直接可比。第一版 predicted pair 只在同一
`group_id` 或同一 predicted dataset 内构造，不跨 predicted 来源混合。

## 6. 序列聚类

在 base affinity table 之后执行序列聚类。

### 6.1 抗体聚类

输入：

```text
heavy_chain
light_chain
single_chain_sequence
```

第一版：

```text
antibody_sequence_key = hash(normalized_heavy + "|" + normalized_light + "|" + normalized_single_chain)
antibody_cluster_id_100 = antibody_sequence_key
```

第二版再加入相似度阈值：

```text
antibody_cluster_id_high
antibody_cluster_id_low
```

阈值可参考：

```text
0.95
0.75
```

但必须写入 manifest。

### 6.2 抗原聚类

输入：

```text
antigen_sequence
```

第一版：

```text
antigen_sequence_key = hash(normalized_antigen_sequence)
antigen_cluster_id_100 = antigen_sequence_key
```

第二版再加入相似度阈值。

### 6.3 complex group

AbRank-style 版本需要显式区分：

```text
measurement_group_id = 原 group_id
complex_group_id = antibody_cluster_id + antigen_cluster_id
```

`measurement_group_id` 表示严格实验可比性；`complex_group_id` 表示序列相似性盒子。
这两个概念不能再混用。

## 7. 切分策略

切分必须先切 records，再在各 split 内构造 pair。

第一阶段支持：

### 7.1 group holdout baseline

```text
同一 measurement_group_id 不跨 train/valid/test
```

用途：工程基线。

### 7.2 predicted pretrain split

predicted 数据按 dataset/group 切分，避免同一 predicted source 的完全重复记录泄漏。

### 7.3 experimental KD split

只使用 E1 层：

```text
metric_family = kd
supervision_tier = experimental_affinity
```

用于主微调和主评估。

### 7.4 hard Ab / hard Ag 后续版本

待抗体/抗原聚类稳定后，再实现：

```text
hard_ab: antibody_cluster_id 不跨 split
hard_ag: antigen_cluster_id 不跨 split
dual: antibody_cluster_id 与 antigen_cluster_id 都不跨 split
```

这些策略不能阻塞第一版 staged training。

## 8. pair 构造

pair builder 是新版本的核心，不再默认只依赖原项目的 group 内 pair。

### 8.1 strict measurement pair

```text
record_i.group_id == record_j.group_id
```

这是最保守、最干净的 pair。

### 8.2 predicted source-local pair

```text
label_kind = predicted
dataset_id 相同
或 group_id 相同
```

不同 predicted 来源的分数不默认互比。

### 8.3 experimental metric-family pair

第一版：

```text
KD 只和 KD 比
IC50 只和 IC50 比
EC50 只和 EC50 比
```

跨 group pair 必须满足：

```text
abs(affinity_score_i - affinity_score_j) >= margin
```

默认 margin 可从 1.0 开始，即约 10-fold 差距。KD/IC50/EC50 混合 pair 单独开关。

### 8.4 AbRank-style mixed pair

后续可选：

```text
KD / IC50 / EC50 共同作为 affinity proxy
只保留大 margin pair
pair manifest 记录 metric_family_i / metric_family_j
```

该模式必须单独报告，不能和严格 experimental KD 主结果混为一个指标。

## 9. 分阶段训练

第一版训练分三步：

### Stage A：predicted pretraining

数据：

```text
P1 + P2 + P3
```

约 110 万可训练记录。

约束：

- 不和 experimental 组成 pair；
- 不同 predicted source 默认不互比；
- 只作为弱监督预训练；
- 输出 checkpoint 标记为 `predicted_pretrain`。

### Stage B：experimental KD fine-tuning

数据：

```text
E1
```

约 17–18 万 KD 记录。

用途：主微调。

### Stage C：experimental KD/IC50/EC50 fine-tuning

数据：

```text
E1 + E2
```

约 21–22 万记录。

用途：扩展微调。KD、IC50、EC50 分开报告，不直接假装为同一物理单位。

## 10. 评估原则

主评估只看 experimental。

必须至少报告：

```text
KD metrics
IC50 metrics
EC50 metrics
experimental overall
predicted sanity metrics
```

不能用 predicted valid/test 作为主结果。

关键实验：

| 实验 | 目的 |
| --- | --- |
| experimental KD from scratch | 干净基线 |
| predicted pretrain -> KD finetune | 检验 predicted 是否有帮助 |
| predicted pretrain -> KD/IC50/EC50 finetune | 检验扩展实验指标是否有帮助 |
| predicted only sanity check | 检查是否只是学会 teacher score |

## 11. 第一阶段验收

完成以下产物才算第一版闭环：

```text
processed/affinitymodel/datasets/base_affinity_table.parquet
processed/affinitymodel/datasets/base_affinity_table_manifest.json
processed/affinitymodel/clusters/sequence_clusters.parquet
processed/affinitymodel/splits/<strategy>/train.parquet
processed/affinitymodel/splits/<strategy>/valid.parquet
processed/affinitymodel/splits/<strategy>/test.parquet
processed/affinitymodel/pairs/<stage>/train_pairs.parquet
outputs/affinitymodel/<run>/metrics.json
```

并且：

- `all_records.parquet` 不被修改；
- predicted 与 experimental 不直接混 pair；
- KD、IC50、EC50 的 metric family 可追踪；
- 每个 split 和 pair 文件都有 manifest；
- 主评估只使用 experimental。

