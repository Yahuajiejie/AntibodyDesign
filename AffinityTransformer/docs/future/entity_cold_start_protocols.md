# 实体泛化协议与数据接口改造方案

## 1. 改动背景

项目最初只有 record、group 和 pair 三层数据结构：record 表示一条抗体—抗原观测，group 决定哪些 records 可以互相比较，pair 是在 group 内由两条 records 派生出的 RankNet 训练样本。这套结构足以跑通 group holdout，但不足以回答 README 中已经明确区分的四类实体泛化问题：抗原是否见过、抗体是否见过，两者需要分别判断。

当前代码已经增加了 Antibody cold-start 和 Antigen cold-start 的固定切分与 K 折算法，也接入了命令行和交叉验证分派。但这些函数假定输入 records 已经带有全部实体字段，真实数据准备、抗体聚类、测量家族、冲突处理、有效模型输入审计和正式运行脚本都没有接通。换句话说，现在完成的是切分算法，不是可以直接用于真实数据的端到端协议。

前一版方案还有三个设计问题。

第一，它把基础 records、切分 annotation、表征 annotation 和审计报告混在一张表里。如果把所有字段都塞进标准 schema，不仅八十多套 converter 会被迫一起修改，主训练流程也会长期携带大量只在切分阶段使用的列。

第二，它把 `protocol`、`partition_unit_id` 等算法内部概念写成了数据字段。实际运行会按协议建立独立目录，切分函数内部也可以使用临时 component，无需让每条 record 重复保存协议名称和切分单元。

第三，它只考虑了“不同完整序列是否重叠”，没有把“模型实际看到了什么”放在正确的数据层。当前抗原缓存采用有限长度并允许截断；不同完整抗原可能在 tokenizer 和截断后变成相同输入。这个问题需要审计，但它属于特定 encoder 配置下的表征信息，不属于抗原本身，也不应成为所有 cold-start 函数的无条件必填字段。

本次改造的目标是重新划清这些边界，并在不破坏现有 `group_holdout_split` 的前提下，把实体协议、pair 构造和训练入口真正接起来。

## 2. 本次确定的设计原则

1. 基础 records schema 暂不扩充。全局实体信息放入独立 annotation 表，通过 `record_id` 或 sequence key 关联。
2. 各 converter 继续负责来源特有的解析和标签转换；全局去重、聚类和跨数据集身份判断必须在 merge 之后完成。
3. 只有原始表中存在、merge 后无法恢复的实验信息，才定点修改对应 converter，例如技术重复编号、同一样本的派生测量关系和 teacher 批次。
4. 除 Pair holdout 外，始终先切 records，再在各 split 内构造 pairs。
5. `group_id` 是排序可比性单位，不是通用切分单位。
6. 实体字段用于生成和审计 split，切分完成后原则上退出主训练链。
7. 不建立一张包含所有信息的“万能 annotation 表”。实体、聚类、表征、冲突和 predicted 质量门控分别保存为窄表或 manifest。
8. `protocol` 由目录和 `split_manifest.yaml` 表示，不作为逐 record 字段；切分内部使用临时 `_component_id`，不持久化 `partition_unit_id`。
9. 冲突记录不能只做备注。原始 records 必须保留，但任何会产生自相矛盾监督的 pair 都必须被阻止。
10. 目录重构放在接口和真实数据行为稳定之后。重构前需要单独通知项目负责人，保留旧导入路径和调用方式的兼容层。

## 3. 数据对象和键的含义

### 3.1 五种基本对象

| 对象 | 含义 | 是否直接进入模型训练 |
| --- | --- | --- |
| record | 一条原始抗体—抗原观测 | 是，经过过滤后进入 Dataset |
| group | 可以互相比较排名的一组 records | 是，决定 pair 构造和 Spearman 范围 |
| entity annotation | record 对应的精确实体、实体簇和测量家族 | 通常否，只用于切分和审计 |
| interaction | 一个规范化抗体输入与一个规范化抗原输入的组合 | 主要用于冲突和泄露审计 |
| pair | 同一 group 内两条 records 派生出的排序关系 | 是，进入 RankNet |

### 3.2 六个实体字段

实体 annotation 保留以下字段：

| 字段 | 含义 | 主要用途 |
| --- | --- | --- |
| `measurement_family_id` | 技术重复、同一样本或同一来源派生测量的家族 | 防止相关测量跨 split |
| `antibody_sequence_key` | 规范化后的精确抗体输入 | 精确重复审计、抗体聚类、评估组有效输入计数 |
| `antibody_cluster_id` | 相同或近似抗体序列簇 | Antibody cold-start 隔离、Antigen cold-start 的“抗体已见”判断 |
| `antigen_sequence_key` | 规范化后的完整抗原或真实 assay construct | 精确重复审计、抗原聚类、已知抗原判断 |
| `antigen_cluster_id` | 相同或近似抗原序列簇 | Antigen cold-start 和 Dual cold-start 隔离 |
| `interaction_key` | 规范化抗体—抗原组合 | 发现相同输入、标签冲突和 interaction overlap |

这些字段不加入 `dataset/schema.py::REQUIRED_COLUMNS`。它们由 merge 后的全局 annotation 流程生成，并以 `record_id` 关联基础 records。

`interaction_key` 可以确定性计算：

```text
interaction_key = hash(antibody_sequence_key, antigen_sequence_key)
```

因此它可以写入 entity annotation 方便审计，也可以在需要时临时生成；切分算法不能依赖人工填写的 interaction 名称。

### 3.3 表征字段

表征 annotation 不按 record 或 interaction 重复保存，而按“序列 × 表征配置”保存：

| 字段 | 含义 |
| --- | --- |
| `sequence_type` | `antibody` 或 `antigen` |
| `sequence_key` | 对应的 antibody/antigen sequence key |
| `representation_id` | encoder、tokenizer、最大长度和长序列策略的确定性配置哈希 |
| `effective_input_hash` | tokenizer 和截断后实际 token 输入的哈希 |
| `raw_length` | 原始序列长度 |
| `effective_length` | 模型实际编码的有效长度 |
| `was_truncated` | 是否发生截断，可由前两项推导 |

`representation_id` 不是人工实验标签。它用于避免把不同 tokenizer 或不同 `max_length` 下生成的 `effective_input_hash` 混为一谈：

```text
representation_id = hash(
    encoder_name,
    encoder_revision,
    tokenizer_revision,
    max_length,
    long_sequence_strategy,
    input_format_version,
)
```

当前缓存 manifest 已有 `sequence_length` 和 `embedding_length`，实现时应优先复用，不重复制造同义列。`was_truncated` 可以在报告阶段计算。

`effective_input_hash` 应按单条序列的非 padding token IDs 和对应 attention mask 计算，不能直接哈希带有 batch padding 的整行 tensor，否则同一序列放进不同 batch 可能得到不同结果。当前代码中的 `effective_antigen_input_hash` 在接口再版后迁移为这里的通用名称；抗原审计使用 `sequence_type="antigen"` 的记录。

完整序列与有效输入的关系按以下规则处理：

1. 优先使用真实 assay construct，不用无关的全长蛋白代替实验构建体；
2. 在显存和 encoder 位置长度允许时提高 `max_length`；
3. 更长序列采用功能区域、chunk 或支持更长上下文的 encoder；
4. 只要仍存在截断，就在 Antigen/Dual cold-start 产物上检查 effective-input overlap；
5. `effective_input_hash` 不再是 Antibody cold-start 的必填字段，也不加入基础 records。

多卡训练不能替代这项审计。DDP 只复制模型并分配 batch，主要提高吞吐量；如果更深模型或更长输入单卡放不下，需要 FSDP、ZeRO、Tensor/Sequence Parallel、activation checkpointing 等真正的内存拆分方案。无论使用几张 GPU，只要最终发生截断，就仍需检查不同抗原是否变成相同有效输入。

## 4. 最终映射结构

### 4.1 基础表与 annotation

基础 records 继续保持当前标准列：

```text
all_records.parquet
record_id | group_id | antibody chains | antigen_sequence | rank_label | ...
```

实体 annotation 是每条 record 的窄映射：

```text
entity_annotations.parquet
record_id
measurement_family_id
antibody_sequence_key
antibody_cluster_id
antigen_sequence_key
antigen_cluster_id
interaction_key
```

聚类采用 sequence key 到 cluster ID 的独立映射：

```text
antibody_clusters.parquet
antibody_sequence_key | antibody_cluster_id

antigen_clusters.parquet
antigen_sequence_key | antigen_cluster_id
```

表征采用 sequence key 到有效输入的独立映射：

```text
representation_annotations.parquet
sequence_type | sequence_key | representation_id
effective_input_hash | raw_length | effective_length | was_truncated
```

这种结构避免在同一抗原对应数十万抗体时，将相同的抗原表征信息复制数十万次。

### 4.2 Record、interaction 和 pair 的关联

```text
record_id
  ├── group_id
  ├── antibody_sequence_key ──> antibody_cluster_id
  ├── antigen_sequence_key  ──> antigen_cluster_id
  ├── measurement_family_id
  └── interaction_key = hash(antibody_sequence_key, antigen_sequence_key)

pair_id
  ├── record_id_i ──> interaction_i / antibody_cluster_i / measurement_family_i
  └── record_id_j ──> interaction_j / antibody_cluster_j / measurement_family_j
```

pair 表仍保持当前紧凑结构，不复制所有实体字段：

```text
pair_id | group_id | record_id_i | record_id_j | label_i | label_j | y_ij
```

需要做 pair 审计时，通过 `record_id_i` 和 `record_id_j` join entity annotation，临时生成：

```text
pair_entity_audit.parquet
pair_id
interaction_key_i
interaction_key_j
antibody_cluster_id_i
antibody_cluster_id_j
antigen_cluster_id_i
antigen_cluster_id_j
measurement_family_id_i
measurement_family_id_j
```

该表是诊断产物，不进入 DataLoader。

### 4.3 冲突集合

标签冲突只在相同可比背景内定义。检查键为：

```text
(group_id, interaction_key)
```

同一 interaction 出现在不同 assay、metric 或 group 中，不自动视为标签冲突，因为这些标签可能本来就不能直接比较。

冲突报告至少包含：

```text
conflict_id
group_id
interaction_key
record_ids
rank_labels
measurement_family_ids
resolution
```

仅删除两个冲突 records 之间的直接 pair 不够。例如相同输入分别标为 1 和 3，第三条输入标为 2，即使删除冲突双方的 pair，仍会同时生成“该输入低于第三条”和“该输入高于第三条”的矛盾监督。

因此采用以下规则：

1. 确认属于同一 measurement family 的技术重复时，按预先规定的方法聚合标签；
2. 无法可靠聚合时，整个冲突集合退出 pair 构造；
3. 原始 records 不物理删除，全部写入 conflict report；
4. 若选择排除 records，复用现有 `keep_for_training=False` 和 `drop_reason`；
5. `build_pairs` 必须保证不会从未解决的冲突集合产生任何硬排序关系。

## 5. 协议再版

README 中的四种实体问题与 group 工程基线分别实现。协议名称、输出目录和指标名称必须一致。

### 5.1 Pair holdout：已见抗原、已见抗体（不着急实现）

Pair holdout 只用于 transductive diagnostic。抗原、抗体、record 和 measurement family 可以重复，但同一个规范化比较关系不能同时进入 train 和 valid/test。

规范化比较键定义为：

```text
pair_relation_key = hash(
    group_id,
    min(interaction_key_i, interaction_key_j),
    max(interaction_key_i, interaction_key_j),
)
```

`group_id` 不能省略，否则来自不同 assay 或 metric 的两个 pair 会被误认为同一个比较关系。

它允许先构造规范化候选 pairs，再按完整比较键切分；这是“先切 records 再构 pair”规则的唯一例外。Pair holdout 不能用于声称新抗体或新抗原泛化。

### 5.2 Group holdout：工程基线

`group_holdout_split` 继续按 `group_id` 保持原子性。它测试未见同质实验组，不自动等于未见抗原、未见抗体或 dual cold-start。

现有接口、配置名和调用方法保持不变。

### 5.3 Antibody cold-start：已知抗原、未见抗体

回答的问题：

> 模型已经见过目标抗原及其部分实验记录后，能否对全局范围从未出现过的抗体簇排序？

流程：

```text
records + entity annotations
→ 按 antibody_cluster_id 与 measurement_family_id 构造连通 component
→ 按 component 记录数分配 train / valid / test
→ 以 train 为参照筛选 valid/test 的已知抗原 records
→ 检查每个评估 group 的记录数、不同抗体输入数和标签数
→ 在各 split 内独立构造 pairs
```

要求：

- `record_id` 不重叠；
- `measurement_family_id` 不重叠；
- `antibody_sequence_key` 不重叠；
- `antibody_cluster_id` 不重叠；
- `interaction_key` 不重叠；
- 主 valid/test record 的 `antigen_sequence_key` 在 train 出现；
- 默认要求主 valid/test 的 `group_id` 在 train 出现；
- 每个评估 group 至少有 `min_eval_records` 条记录、两个不同抗体输入和两个不同标签。

该协议不要求 `antigen_cluster_id` 或 `effective_input_hash` 作为切分函数的必填字段。

### 5.4 Antigen cold-start：未见抗原、已知抗体

回答的问题：

> 面对训练阶段从未出现的抗原簇，模型能否对已经在其他训练抗原上出现过的抗体簇排序？

流程：

```text
records + entity annotations
→ 按 antigen_cluster_id 与 measurement_family_id 构造连通 component
→ 按 component 分配 train / valid / test
→ 从 valid/test 中筛出 antibody_cluster_id 已在 train 出现的 records
→ 检查每个评估 group 是否可计算排序指标
→ 在各 split 内独立构造 pairs
```

要求：

- `record_id`、`measurement_family_id` 不重叠；
- `antigen_sequence_key`、`antigen_cluster_id` 不重叠；
- `interaction_key` 不重叠；
- 主 valid/test 中每条 record 的 `antibody_cluster_id` 在 train 出现；
- 每个评估 group 至少有 `min_eval_records` 条记录、两个不同抗体输入和两个不同标签；
- 若该模型表征存在截断，指定 representation 下的 effective input 不得跨 split 重叠。

被分配到未见抗原 holdout、但抗体簇也未见的 records 属于 Dual cold-start 子集，不能静默删除。它们写入 `excluded_records`，并标记原因。

### 5.5 Dual cold-start：未见抗原、未见抗体

回答的问题：

> 面对训练阶段未见的目标抗原，能否对一批同样未见的候选抗体排序？

该协议与比赛最终场景最接近，应作为正式模型选择的主协议。

Dual cold-start 不能分别随机切 antigen clusters 和 antibody clusters 后再取交集。一个 measurement family 或 interaction 可能连接多个实体，独立分配会产生矛盾。正式实现应将以下关系构成二部或多部图：

```text
antigen_cluster
antibody_cluster
measurement_family
interaction
```

图上的每个连通 component 是不可拆分单元，再由现有的按记录数平衡算法分配到 train/valid/test 或 K folds。

要求：

- `record_id`、measurement family、exact/cluster antibody、exact/cluster antigen 和 interaction 均不跨 split；
- 若发生截断，effective input 也不跨 split；
- 超大 component 必须报告 records、groups、抗原簇和抗体簇数量，不能静默固定到 train；
- valid/test 内仍需按 group 检查至少两个有效输入和两个标签。

### 5.6 协议目录与 manifest

每套协议使用独立目录：

```text
processed/binding/splits/
├── pair_holdout/
├── group_holdout/
├── antibody_cold_start/
├── antigen_cold_start/
└── dual_cold_start/
```

目录本身负责区分协议，不在每条 record 上增加 `protocol` 字段。每个目录仍需保存 `split_manifest.yaml`：

```yaml
protocol: antibody_cold_start
seed: 42
input_records_hash: ...
entity_annotations_hash: ...
antibody_cluster_manifest_hash: ...
antigen_cluster_manifest_hash: ...
representation_manifest_hash: ...
code_commit: ...
```

manifest 防止目录复制、改名或部分文件替换后失去真实语义。

## 6. Pair 构造契约

实体协议统一执行：

```text
records
→ entity annotation
→ cluster/component
→ train / valid / test records
→ 冲突处理
→ 各 split 内分别 build_pairs
```

`build_pairs` 必须满足：

1. 两个端点来自同一个 split；
2. 两个端点属于同一个 `group_id`；
3. 标签相同不构造硬排序关系；
4. 未解决的 `(group_id, interaction_key)` 冲突集合不构造任何 pair；
5. pair 只保存 record 端点，不重复保存整套 entity annotation；
6. valid/test pairs 仅用于 pair accuracy 等诊断，Spearman 主指标直接对 records 打分。

第一版不新增 `pair_confidence`、`pair_weight`、`record_loss_weight` 或 `training_quality_score`。没有逐 record ensemble 方差或实验误差时，这些字段没有可靠来源。若以后获得可信不确定性，再单独扩展 `PAIR_COLUMNS`、Dataset、collate 和 loss。

## 7. 聚类产物

### 7.1 抗体聚类

抗体不能直接复用抗原的全长 Hamming 聚类。正式方案至少需要考虑：

- heavy/light/single-chain 的规范化规则；
- 抗体类型；
- VH/VL 全长；
- heavy CDR3 和 light CDR3；
- 缺链、VHH 和异常残基；
- ANARCI 或等价编号失败时的明确处理。

输出只保存 `antibody_sequence_key → antibody_cluster_id`。算法、阈值和失败统计写入 manifest。

### 7.2 抗原聚类

现有 `compute_antigen_clusters` 需要改为以 `antigen_sequence_key` 为输入，不能继续要求一个 `antigen_key` 只映射一条序列。还需要补充长度不同、少量插入缺失和不同 construct 的处理。

输出只保存 `antigen_sequence_key → antigen_cluster_id`。`cluster_size` 可运行时计算，不复制到 entity annotation。

### 7.3 Cluster manifest

每次聚类至少保存：

```text
entity_type
algorithm
algorithm_version
similarity_threshold
linkage_method（适用时）
input_records_hash
code_commit
n_sequences
n_clusters
cluster_size_summary
```

## 8. Predicted 数据和 noise-aware sampler

### 8.1 Predicted 质量门控

Predicted 第一版采用批次级门控，不向百万条 records 复制相同的统计字段。每个 teacher 批次生成一行质量报告：

```text
predicted_batch_id 或 dataset_id
teacher_model
teacher_revision
n_matched
spearman_rho
pearson_r
rho_ci_lower
rho_ci_upper
permutation_p_value
bh_fdr_q_value
gate_status
gate_reason
```

如果一个 `dataset_id` 就对应一个 teacher 批次，可以直接复用 `dataset_id`；只有同一数据集中混入多个 teacher 批次时才新增 `predicted_batch_id`。

第一版优先比较：

```text
通过门控的 predicted 预训练
→ experimental 微调
```

这一路径可以复用现有 `label_kind`，不需要逐 record loss weight。没有 ensemble 方差时，predicted pair 通过最小分数差或跨分位区间规则筛选，不伪造 pair confidence。

### 8.2 Noise-aware sampler

Noise-aware sampler 已经能通过 `resolve_tau_for_group` 得到 `tau`、rule label 和 basis，本次不新增标准字段，也不修改主训练 schema。

后续只需把每组实际解析到的以下信息写入 sampler 审计报告：

```text
group_id
resolved_tau
tau_rule
tau_basis
```

这属于运行产物，不是新的训练输入。

## 9. 协议接口再版

### 9.1 基础加载接口

保留：

```python
load_records(path: Path) -> pd.DataFrame
```

新增独立加载和 join 接口：

```python
load_entity_annotations(path: Path) -> pd.DataFrame
load_representation_annotations(path: Path) -> pd.DataFrame
join_entity_annotations(records, annotations) -> pd.DataFrame
validate_entity_annotations(records, annotations) -> None
```

`load_records` 继续只检查基础 records schema，不能因为新协议而要求所有旧数据文件包含实体列。

### 9.2 固定切分接口

保留旧接口：

```python
build_splits(
    records,
    strategy="group_holdout_split",
    valid_fraction=...,
    test_fraction=...,
    seed=...,
)
```

实体协议使用具名函数，显式传入 annotation：

```python
build_antibody_cold_start_split(
    records,
    entity_annotations,
    *,
    valid_fraction,
    test_fraction,
    seed,
    min_eval_records,
    require_train_group=True,
)

build_antigen_cold_start_split(
    records,
    entity_annotations,
    *,
    valid_fraction,
    test_fraction,
    seed,
    min_eval_records,
    representation_annotations=None,
)

build_dual_cold_start_split(
    records,
    entity_annotations,
    *,
    valid_fraction,
    test_fraction,
    seed,
    min_eval_records,
    representation_annotations=None,
)
```

`representation_annotations` 只在指定表征发生截断、需要 effective-input 审计时提供。不能把它重新变成所有协议的无条件依赖。

函数内部可以 join 为临时 DataFrame，但输出的 `train/valid/test.parquet` 默认保持基础 records schema，实体信息保存在 split 目录旁的 annotation/审计产物中。

### 9.3 K-fold 接口

保留 `build_group_kfolds`。实体 K 折采用与固定切分相同的 annotation 参数：

```python
build_antibody_cold_start_kfolds(records, entity_annotations, ...)
build_antigen_cold_start_kfolds(records, entity_annotations, ...)
build_dual_cold_start_kfolds(records, entity_annotations, ...)
```

正式 test 先冻结，不参与 K-fold 轮转。每折 valid 都要相对该折 train 重新计算“抗原已见”或“抗体已见”的资格，不能只按静态 fold assignment 判断。

### 9.4 产物接口

每套实体协议至少输出：

```text
train.parquet
valid.parquet
test.parquet
split_manifest.yaml
component_assignments.parquet
eligibility_report.csv
excluded_records.parquet
leakage_report.csv
summary.csv
```

`excluded_records` 必须保留原始分配 split 和排除原因。任何 records 数量减少都应能由该表解释。

## 10. 建议新增模块和目录

以下是目标结构，不在写本文档时立即移动文件：

```text
affinity_transformer/
├── annotations/
│   ├── schema.py                  # entity/representation/conflict 表契约
│   ├── entities.py                # sequence key、interaction、measurement family
│   ├── representation.py          # effective input 与截断审计
│   ├── conflicts.py               # (group_id, interaction_key) 冲突处理
│   └── io.py                      # annotation 加载、校验和 join
├── clustering/
│   ├── antibody.py
│   ├── antigen.py
│   └── manifests.py
├── splitting/
│   ├── common.py                  # component 和按权重分配算法
│   ├── group.py
│   ├── antibody_cold_start.py
│   ├── antigen_cold_start.py
│   ├── dual_cold_start.py
│   ├── audits.py
│   └── results.py
├── weak_supervision/
│   └── predicted_quality.py
├── dataset/
│   ├── records.py
│   ├── pairs.py
│   └── pair_sampling/
└── splits.py                      # 迁移期兼容 facade，保留旧导入路径
```

脚本侧建议新增：

```text
scripts/data/
├── build_entity_annotations.py
├── audit_record_conflicts.py
├── build_antibody_clusters.py
├── build_antigen_clusters.py
├── audit_predicted_quality.py
└── build_splits.py
```

表征 annotation 优先在现有 cache 构建流程内生成，避免再跑一遍 tokenizer：

```text
scripts/embeddings/build_v065_cache.py
→ embedding cache
→ representation_annotations.parquet
→ truncation/effective-input audit
```

数据产物建议按层组织：

```text
processed/binding/
├── all_records.parquet
├── annotations/
│   ├── entity_annotations.parquet
│   ├── conflict_report.parquet
│   └── predicted_quality_report.parquet
├── clusters/
│   ├── antibody_clusters.parquet
│   ├── antibody_clusters.yaml
│   ├── antigen_clusters.parquet
│   └── antigen_clusters.yaml
├── representations/
│   └── <representation_id>/
│       ├── representation_annotations.parquet
│       └── manifest.yaml
└── splits/
    └── <protocol>/
        └── <split_version>/
```

## 11. 受波及的函数和文件

### 11.1 必须新增或修改

| 文件/函数 | 修改内容 |
| --- | --- |
| `affinity_transformer/dataset/schema.py` | 保持基础 `REQUIRED_COLUMNS`；新增 annotation schema 应移到独立模块 |
| `affinity_transformer/dataset/records.py::load_records` | 保持原接口；新增 annotation loader/joiner，不让基础 loader 承担协议字段校验 |
| `affinity_transformer/record_filter.py::antibody_sequence_hashes` | 统一为规范化 antibody sequence key，确保规则与实际 encoder 输入一致 |
| `affinity_transformer/antigen_clustering.py::compute_antigen_clusters` | 改用 `antigen_sequence_key`，修复一 antigen key 多序列，补长度不同和 indel 处理；稳定后迁入 `clustering/` |
| `affinity_transformer/splits.py` 的四个 entity cold-start 函数 | 改为显式接收 entity annotation；按协议拆分必需字段；增加 Dual cold-start；迁移期保留导出 |
| `splits.py::_prepare_cold_start_records` | 删除“所有协议无条件要求全部字段”的逻辑，改为协议专用校验 |
| `splits.py::_derive_entity_components` | 保留内部临时 component，不输出 `partition_unit_id`；增加 dual component 图 |
| `splits.py::_select_protocol_eligible_records` | 分别实现已知抗原、已知抗体和 dual 资格规则 |
| `splits.py::_build_entity_protocol_leakage_report` | 按协议解释允许/禁止重叠；effective-input 检查变为条件项 |
| `scripts/data/build_splits.py` | 接收 entity/representation annotation 路径；写 manifest；增加 Dual cold-start |
| `affinity_transformer/config.py` | 增加 annotation、cluster manifest、split manifest 路径；自动切分合法值接入新协议 |
| `affinity_transformer/training/data.py::resolve_data_paths` | 自动切分模式加载 annotation；正式训练优先消费冻结 split artifact |
| `affinity_transformer/training/cross_validation.py::run_group_kfold_cross_validation` | 加载 entity annotation；增加 dual 分派；函数名在兼容期保留，后续再改名 |
| `affinity_transformer/embeddings/huggingface.py` | 缓存构建时暴露实际 token 输入哈希和有效长度 |
| `affinity_transformer/embeddings/pipeline.py::write_embedding_cache` | 同步写 representation annotation 或足够生成它的 manifest 字段 |
| `affinity_transformer/embeddings/validation.py::validate_embedding_cache` | 在已有截断统计上增加 effective-input collision 审计 |
| `scripts/embeddings/build_v065_cache.py` | 输出 representation annotation，不重复 tokenizer 工作 |
| `affinity_transformer/dataset/pairs.py::build_pairs` | 接收已解决的 conflict policy/集合，保证冲突输入不产生矛盾 pair |
| `affinity_transformer/record_filter.py::filter_records` | Predicted 门控接入时只消费批次 PASS/FAIL，不读取或复制整套相关性统计 |
| `affinity_transformer/training/artifacts.py` | 保存 split、entity、cluster、representation manifest 的哈希引用 |
| `scripts/runs/g00_qc_and_splits.sh` | 在 filter/split 前接入 annotation、冲突、聚类和协议产物 |
| `scripts/slurm/g00_qc_and_splits.sbatch` | 跟随新的 G00 流程和资源需求 |
| `tests/test_entity_cold_start_splits.py`、`tests/test_cross_validation.py` | 改为真实 annotation 接口，补协议专用字段、Dual、manifest 和兼容性回归 |
| 新增 annotation/cluster/conflict/representation 测试 | 覆盖键稳定性、一对一映射、冲突处理、聚类版本和有效输入碰撞 |

### 11.2 原则上保持不变

| 文件/模块 | 原因 |
| --- | --- |
| 大多数 `scripts/prepare/binding/*/convert.py` 和 `prepare.sh` | 全局实体和聚类不能在单数据集 converter 内正确生成 |
| `scripts/prepare/binding/prepare_all.sh` | 继续只负责逐来源转换和基础表验证；全局 annotation 放在 merge 后的新阶段 |
| `scripts/prepare/binding/merge_records.py` | 继续合并标准 records；只需在总流程中接到 annotation 阶段，可补输入哈希 |
| `scripts/prepare/validate_processed_table.py` | 继续验证基础 records；annotation 使用独立 validator |
| `dataset/datasets.py` 和 `dataset/examples.py` | 主训练仍消费基础 records，不把切分 annotation 塞进 Example |
| `training/loaders.py` 的公共返回类型 | pair 仍由 record 端点构成；只需把 conflict 处理放在 `build_pairs` 之前或内部 |
| 模型结构和 RankNet loss | 实体协议不应进入模型 forward；第一版不增加 pair weight |
| `metrics.py` 的 Spearman 计算 | 仍按 `group_id` 计算；只需由 artifact 层附加协议名称和分层结果 |
| noise-aware sampler 主算法 | 已经实现，本次只补审计输出，不新增 schema |

### 11.3 定点修改 converter 的判定规则

先对所有 ready 数据集做字段可恢复性审计。只有满足以下条件才修改具体 converter：

1. 原始数据明确提供 replicate/sample/clone/derived-measurement ID；
2. 当前标准 records 已经丢掉这项信息；
3. merge 后无法从现有字段可靠恢复；
4. 该信息确实会改变 `measurement_family_id`、冲突处理或 predicted teacher 批次。

不满足这些条件时，不批量修改 converter。

## 12. 实施顺序

### 阶段 A：冻结数据契约

1. 完成 entity、cluster、representation、conflict 和 predicted report schema；
2. 为六个实体字段写明规范化规则；
3. 确定无法恢复 measurement family 的数据采用何种显式状态，不用 `record_id` 静默冒充；
4. 确定 interaction 冲突的聚合和排除规则。

### 阶段 B：真实数据 annotation

1. merge 后生成 sequence keys 和 interaction key；
2. 审计需要定点补充实验元数据的 converters；
3. 生成 measurement families；
4. 生成 conflict report 并执行冲突策略。

### 阶段 C：聚类

1. 实现抗体聚类及失败处理；
2. 重写抗原聚类输入和长度/indel 逻辑；
3. 保存 mapping 和 manifest；
4. 检查 mega-cluster 和 cluster 稳定性。

### 阶段 D：协议接口 v2

1. 修改两个已实现 cold-start 函数的 annotation 接口；
2. 实现 Dual cold-start；
3. 保持 `group_holdout_split` 完全兼容；
4. 接入固定 split、K-fold、CLI 和 config；
5. 产出完整 eligibility、excluded 和 leakage 报告。

### 阶段 E：表征审计

1. 缓存构建同时生成 representation annotation；
2. 提高可行的 antigen `max_length`；
3. 报告截断率和 effective-input collision；
4. 对碰撞抗原修正输入或合并切分单元。

### 阶段 F：正式运行链

1. 修改 G00 和集群脚本；
2. 冻结 final test，只在最终 checkpoint 上评估一次；
3. 让训练产物记录全部 manifest 哈希；
4. 跑真实数据的小规模端到端测试；
5. 接入 predicted 质量门控和两阶段训练。

### 阶段 G：目录迁移

只有阶段 A 至 F 的接口和真实数据行为稳定后，才将 `splits.py`、`antigen_clustering.py` 等迁入新子包。迁移前需要明确通知，迁移后至少保留一个版本的兼容 facade 和回归测试。

## 13. 验收条件

完成以下条件后，才能称新协议已经正式落地：

1. 真实 `all_records.parquet` 可以生成完整 entity annotation，不依赖手工伪造测试列；
2. 所有 cluster mapping 都有可追踪 manifest；
3. 所有标签冲突都有 resolution，训练 pairs 中不存在相同输入导致的矛盾监督；
4. Antibody、Antigen 和 Dual cold-start 都能生成非空、可评估的 valid/test；
5. 每个协议的禁止重叠为零，允许重叠被明确报告；
6. 发生截断的表征完成 effective-input collision 审计；
7. pairs 只在 split 后、同 group 内构造；
8. K-fold 不读取 final test；
9. G00、集群 YAML、训练入口和结果汇总消费的是同一批冻结 split artifact；
10. `group_holdout_split` 的原接口和历史测试继续通过；
11. 文档、目录名、配置名和结果表使用相同协议名称；
12. 目录重构没有破坏旧导入路径，或已经提供明确迁移说明。

## 14. 当前实现与本文档的差距

截至本文档更新时，代码已经实现 Antibody/Antigen cold-start 的切分算法、K 折、CLI 分派、CV 分派和合成数据测试，但仍存在以下差距：

- `splits.py` 仍要求身份字段直接内嵌在 records；
- 两种协议仍共用过宽的 `COLD_START_IDENTITY_COLUMNS`；
- 没有真实数据 entity annotation 和 measurement family 流程；
- 没有抗体聚类；
- 现有抗原聚类仍以 `antigen_key` 为入口，并受一 key 多序列限制；
- 没有 Dual cold-start 实现；
- 没有 conflict resolution 接入 pair 构造；
- representation annotation 和 effective-input collision 尚未实现；
- 正式 G00 和现有实验配置仍主要使用 group holdout；
- predicted 质量门控尚未进入数据和训练链。

因此，当前代码只能视为协议算法原型，不能把它生成的合成测试结果当作真实数据协议已经完成的证明。
