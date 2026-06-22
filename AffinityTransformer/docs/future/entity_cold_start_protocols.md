# Antibody / Antigen Cold-start 实现设计

本文档定义两个优先实现的实体泛化协议：

1. 已知抗原、未见抗体（Antibody cold-start）；
2. 未见抗原、已知抗体（Antigen cold-start）。

本轮不改变 `group_holdout_split` 的接口、配置名称和调用方法，也不移动 `splits.py`。新协议先以具名函数接入现有切分模块；等接口和真实数据行为验证稳定后，再单独讨论是否迁移到 `splitting/` 子包。

## 1. 共同原则

### 1.1 切分顺序

这两种协议都必须先切 records，再构造 pairs：

```text
规范化 records
→ 构造实体簇和不可拆分 component
→ 分配 train / valid / test 或 K folds
→ 筛选符合协议的 valid/test records
→ 分别在各集合内构造 pairs
```

不允许将已采样的 pairs 独立随机分配到三个集合。如果为了审计预先计算了候选 pair，pair 的归属必须由两个端点的 record split 决定；端点分属不同 split 的 pair 直接禁用。

### 1.2 必需字段

两种协议不在切分函数内猜测实体身份。输入 records 必须事先包含：

| 字段 | 含义 |
| --- | --- |
| `record_id` | 原始记录唯一标识 |
| `group_id` | 排序可比性单位 |
| `dataset_id` | 数据集来源 |
| `measurement_family_id` | 技术重复和派生测量家族 |
| `antibody_sequence_key` | 规范化精确抗体输入 |
| `antibody_cluster_id` | 抗体近重复簇 |
| `antigen_sequence_key` | 规范化精确抗原序列 |
| `antigen_cluster_id` | 抗原近重复簇 |
| `interaction_key` | 抗原—抗体组合 |
| `effective_antigen_input_hash` | tokenizer 和截断后的真实抗原输入 |
| `rank_label` | 方向统一后的排序标签 |
| `keep_for_training` | 记录是否可进入训练/评估 |

任一身份字段缺失或存在空值时直接报错，不将 `record_id` 或 `group_id` 静默当成替代品。

### 1.3 不可拆分 component

单一 cluster ID 并不一定是最终切分单位。如果一个 `measurement_family_id` 连接了多个实体簇，这些簇必须合并到同一 component，否则技术重复或派生标签会跨 split。

- Antibody cold-start：按 `antibody_cluster_id + measurement_family_id` 的连通关系构造 component；
- Antigen cold-start：按 `antigen_cluster_id + measurement_family_id` 的连通关系构造 component。

每个 component 只能进入一个 split 或一个 validation fold。

### 1.4 标签的使用边界

切分不根据具体标签高低反复调整。`rank_label` 只用于判断某个评估 group 是否至少有两个不同标签，避免产生无法计算 Spearman 的空评估集。

## 2. Antibody cold-start

### 2.1 回答的问题

> 在模型已经见过某个抗原及其部分实验记录后，能否对全局范围从未出现过的抗体簇进行排序？

“未见抗体”是全局条件。某抗体簇只要在任意训练抗原下出现，就不能再进入 valid/test。

### 2.2 切分流程

```text
records
→ 构造全局 antibody components
→ 按 component 记录数分配 train / valid / test
→ 以 train 为参照筛选 valid/test
→ 输出主评估 records 和被排除 records
```

分配后的 valid/test record 首先需要满足：

1. `antigen_sequence_key` 在 train 出现；
2. 默认要求 `group_id` 也在 train 出现，以尽量排除未见 assay/metric group 的额外分布偏移；
3. 筛选后每个 group 至少有 `min_eval_records` 条记录；
4. 至少有两个不同 `antibody_sequence_key`；
5. 至少有两个不同有效 `rank_label`。

如果只要求已知抗原，不要求同一实验 group 在 train 出现，必须显式使用 `require_train_group=False`。这类结果应另行标记为“known antigen, new experimental group”。

### 2.3 泄露审计

以原始 component 分配结果审计，不只审计筛选后的评估子集：

- record overlap = 0；
- measurement family overlap = 0；
- exact antibody sequence overlap = 0；
- antibody cluster overlap = 0；
- interaction overlap = 0；
- valid/test 中的每个主评估 record 的 exact antigen 在 train 出现。

`group_id` 和 antigen 重叠是协议预期行为，应报告重叠数，但不应报错。

## 3. Antigen cold-start

### 3.1 回答的问题

> 对训练阶段从未出现的抗原簇，模型能否对已经在其他训练抗原下出现过的抗体簇进行排序？

“允许抗体重叠”不等于“抗体已见”。主评估子集必须逐条确认 `antibody_cluster_id` 在 train 中出现。

### 3.2 切分流程

```text
records
→ 构造 antigen components
→ 按 component 记录数分配 train / valid / test
→ 以 train antibody clusters 筛选 valid/test records
→ 在每个 group 内检查主评估记录是否足够
→ 输出 seen-antibody 主子集和 unseen-antibody 排除子集
```

主评估 valid/test group 需要：

1. 所有保留 record 的 `antibody_cluster_id` 在 train 出现；
2. 至少有 `min_eval_records` 条保留记录；
3. 至少有两个不同 `antibody_sequence_key`；
4. 至少有两个不同有效 `rank_label`。

抗原未见但抗体也未见的 records 不是脏数据，而是 dual cold-start 子集。本协议的主 valid/test 不使用它们，但必须将它们写入 `excluded_records` 并标明 `antibody_cluster_not_seen_in_train`，不能静默丢失。

### 3.3 泄露审计

- record overlap = 0；
- measurement family overlap = 0；
- exact antigen sequence overlap = 0；
- antigen cluster overlap = 0；
- effective antigen input overlap = 0；
- interaction overlap = 0；
- 主 valid/test 中每条 record 的 antibody cluster 都在 train 出现。

antibody cluster overlap 在本协议中是必要条件，不是泄露。

## 4. Pair 构造

切分函数返回 records，不在内部采样训练 pairs。调用方按现有契约分别执行：

```python
train_pairs = build_pairs(result.train, ...)
valid_pairs = build_pairs(result.valid, ...)  # 只用于 pair accuracy 等诊断
test_pairs = build_pairs(result.test, ...)
```

Spearman 主评估直接对 valid/test records 打分，不依赖 valid/test 采样出来的 pairs。

## 5. 产物契约

两种协议使用统一结果类型，至少包含：

| 产物 | 内容 |
| --- | --- |
| `train` | 所有分配到 train 的可训练 records |
| `valid` | 符合本协议主问题的 valid records |
| `test` | 符合本协议主问题的 test records |
| `unit_assignments` | component 到 split/fold 的映射及记录数 |
| `eligibility_report` | 每个 valid/test group 的筛选数量和原因 |
| `excluded_records` | 被分配到 holdout 但不符合主问题的 records |
| `leakage_report` | 禁止/允许重叠的审计 |
| `summary` | 各 split 的 records、groups、label kind 和可评估 group 数 |

`excluded_records` 必须保留 `_assigned_split` 和 `protocol_exclusion_reason`。任何记录数减少都应能从该表解释。

## 6. K-fold 契约

K-fold 不用 `group_id` 作通用原子单位：

- Antibody cold-start：每个 antibody component 恰好进入一次 validation fold；
- Antigen cold-start：每个 antigen component 恰好进入一次 validation fold。

每折的 valid 仍需要相对该折 train 重新进行协议资格筛选。因此，一个 component 被分配到 validation fold 不等于它的所有 records 都一定进入主指标。

最终 test 不参与 K-fold 轮转。正式流程应先按同一协议冻结 test，然后只对剩余 development pool 构造 folds。

## 7. 实现边界

本轮实现：

- 两种协议的固定 train/valid/test 切分；
- 两种协议的 K-fold 构造函数；
- component 构造、资格筛选、排除记录和泄露报告；
- 与现有 `build_pairs` 的集成测试。
- `cross_validation.protocol` 对两种新协议的分派；未填写时仍使用原 `group_holdout`。

本轮不实现：

- 抗体聚类算法本身；切分函数消费已审计的 `antibody_cluster_id`；
- 将新协议接入现有集群 YAML；
- 目录迁移。

`run_group_kfold_cross_validation` 的函数名和调用签名为了兼容现有入口暂时保留，但其内部已根据 `cross_validation.protocol` 选择 group、antibody 或 antigen K-fold。未来如果迁移到 `splitting/` 包，再单独更名并提供迁移期别名。
