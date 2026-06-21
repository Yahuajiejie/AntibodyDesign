# AffinityTransformer 工程说明

本文档说明当前项目要解决的问题、数据与评估边界、代码目录、主要模块和验收要求。内容以仓库中的实际代码为准；尚未接入训练流程的功能，会明确标成“未接入”或“后续工作”。

项目总 README 负责介绍比赛背景和建模思路，本文件负责回答开发时最常见的几个问题：数据应该在哪一层处理，哪些记录可以比较，训练集和验证集应该怎样隔离，pair 应该何时生成，以及各模块之间怎样传递数据。

## 1. 项目目标

本项目面向抗体候选排序。模型接收抗体序列和抗原信息，输出一个实数分数：

```text
score = f(antibody, antigen)
```

这个分数不表示具有物理单位的 Kd、IC50 或 EC50。它只用于判断同一可比背景下，哪些抗体应该排在前面。

项目采用排序学习，主要有两个原因：

1. 不同数据集的指标、单位和实验条件不同，绝对数值通常不能直接混在一起回归。
2. 比赛最终关注候选抗体的先后顺序，Spearman 等级相关系数比绝对误差更符合任务目标。

当前代码已经实现：标准数据表、组内 pair 构造、冻结编码器的 embedding 缓存、抗体—抗原交互模型、RankNet 训练、分组 Spearman 评估和实验脚本。

当前代码没有实现 README 中设想的 MSA 抗原路径。现有抗原表征只有精确序列路径；MSA、ESM-MSA 和 Exact + MSA 融合仍是后续方案，不能写成已经具备的能力。

## 2. 三种数据单位

项目中有三种容易混淆的数据单位。

| 单位 | 含义 | 主要用途 |
| --- | --- | --- |
| record | 一条抗体—抗原观测记录 | 数据清洗、切分、逐条预测 |
| group | 一组可以互相比较排名的 records | 决定哪些记录可以构造 pair，计算组内 Spearman |
| pair | 从同一 group 的两条 records 派生出的训练样本 | RankNet 训练 |

`group_id` 回答的是“哪些 records 可以比较”，split key 回答的是“train、valid、test 之间要隔离什么”。两者可能相同，也可能不同。

因此，不能从“pair 只在 group 内生成”直接推出“所有任务都必须按 group 切分”。切分方式应由评估目标决定。

### 2.1 pair 不是独立观测

同一条 record 可以和多条记录组成不同 pair。例如：

```text
(i, j), (i, k), (i, l)
```

如果先生成 pair，再随机切分，record `i` 很可能同时出现在训练集和验证集中。这属于直接数据污染。

项目必须按以下顺序处理：

```text
原始 records
  -> 清洗并形成标准表
  -> 按评估协议切分 records
  -> 在 train、valid、test 内部分别生成 pair 或逐条预测
```

禁止采用以下顺序：

```text
原始 records
  -> 生成全部 pairs
  -> 随机切分 pairs
```

当前训练 loader 会在读取已经切好的记录表后调用 `build_pairs`，顺序是正确的。后续新增数据入口时也必须保持这一点。

## 3. 评估协议与数据切分

### 3.1 主协议：未见抗原泛化

比赛后期才给出目标抗原，因此主评估应回答：模型面对训练时没有完整见过的新抗原，能否给候选抗体排序？

推荐的切分单位依次为：

1. `antigen_cluster_id`：由抗原序列聚类得到，最严格，可避免高度同源抗原分散到不同集合。
2. `antigen_key` 或经过人工校验的抗原身份：没有可靠聚类时使用。
3. `group_id`：只能作为当前代码的退化方案，不能自动代表未见抗原。

切分后必须重新在各集合内部构造 pair。报告结果时，应同时给出 group macro Spearman、按记录数加权的 Spearman，以及按抗原、数据来源和标签类型拆分的结果。

### 3.2 辅助协议：已知抗原上的新抗体

另一个有实际意义的问题是：模型已经见过某个抗原和一部分抗体后，能否给这个抗原的新候选抗体排序？

这个协议允许同一抗原出现在 train 和 valid，但必须满足：

- 同一条 record 不跨集合；
- 同一抗体序列及其近重复序列不跨集合；
- 先切 records，再分别构造 pairs；
- 只选择验证记录数足够的 group，避免用一两个点计算 Spearman；
- 结果命名为 `within-antigen generalization`，不能解释成未见抗原泛化。

这套协议适合作为辅助分析，不应代替比赛主验证。

### 3.3 当前代码能够保证什么

`affinity_transformer/splits.py` 当前提供两种固定切分：

| 策略 | 行为 | 使用范围 |
| --- | --- | --- |
| `debug_record_split` | 按 record 随机切分 | 只用于调试流程，不用于汇报模型效果 |
| `group_holdout_split` | 同一 `group_id` 整体进入一个集合 | 当前正式实验使用 |

`build_group_kfolds` 也只按 `group_id` 构造交叉验证折，并通过贪心分配尽量平衡各折的记录数。

当前泄露报告会检查 `record_id` 重叠；使用 group holdout 时还会检查 `group_id` 重叠。它不会检查 `antigen_key`、抗原序列或抗原序列簇是否跨集合。

所以，当前 `group_holdout_split` 的准确含义是“未见 group 的验证”，不一定是“未见抗原的验证”。一个抗原可能因为指标、实验方法或数据来源不同而形成多个 group，并被分到不同集合。

要使主评估与比赛场景严格一致，后续需要在 schema 和切分模块中加入稳定的抗原身份或抗原簇字段，并将其纳入切分与泄露检查。这是当前最高优先级的评估改动。

### 3.4 小 group 和超大 group

小 group 不适合机械地做组内 80/20 切分。一个 group 如果只有 9 条记录，验证集通常只有 2 条；此时 Spearman 几乎只能取 `+1` 或 `-1`，结果非常不稳定。只有 1 条记录时，Spearman 无法定义。

当前 `group_holdout_split` 会把大于 holdout 目标容量的超大 group 固定在训练集，以避免验证集被单个 group 占满。这是工程折中，但也会形成验证盲区。正式报告应列出哪些 group 从未进入验证集；必要时再做单独的 leave-one-group-out 实验。

## 4. 代码分层与目录

项目分为数据准备、通用框架和实验入口三层。依赖方向应保持单向。

```text
AffinityTransformer/
├── affinity_transformer/          # 通用训练与推理框架
│   ├── config.py                  # YAML 配置解析和校验
│   ├── record_filter.py           # 标准表过滤
│   ├── splits.py                  # 数据切分、交叉验证和泄露报告
│   ├── metrics.py                 # 分组 Spearman 及汇总
│   ├── trainer.py                 # 通用训练循环
│   ├── user_entry.py              # 对外推理接口
│   ├── dataset/                   # records、groups、pairs 和 Dataset
│   ├── embeddings/                # 冻结编码器及离线缓存
│   ├── model/                     # 投影、交互、池化和打分网络
│   └── training/                  # loader、runner、评估和产物写出
├── configs/                       # 模型、数据和实验配置
├── scripts/
│   ├── prepare/                   # 各原始数据集的转换与质检
│   ├── data/                      # 合并表的过滤、检查与切分
│   ├── embeddings/                # embedding 缓存构建
│   ├── experiments/              # 批量实验与结果汇总
│   ├── runs/                      # 本地实验组入口
│   └── slurm/                     # 集群任务脚本
├── tests/                         # 单元测试和集成测试
├── train.py                       # 训练命令行入口
├── predict.py                     # 推理命令行入口
└── docs/                          # 工程说明和专题文档
```

`processed/` 存放标准表、切分和 embedding 缓存；`outputs/` 存放训练结果。两类目录都属于生成产物，不应提交大文件到 Git。

### 4.1 各层职责

数据准备层位于 `scripts/prepare/`。这里处理原始 CSV 的列名、单位、标签方向、抗体链拆分、信号肽和单个数据集的特殊规则，最终产出统一的 `records.parquet`。

通用框架层位于 `affinity_transformer/`。这里不应知道某篇论文原始表格的列名，也不负责修正某个数据集的标签方向。它只消费标准表。

实验入口层包括 `train.py`、`predict.py` 和 `scripts/` 下的运行脚本。它负责读取配置和组织流程，不应重新实现 dataset、model 或 metric 内部逻辑。

## 5. 标准数据表

`affinity_transformer/dataset/schema.py` 定义标准表的必需字段。主要字段可分为六类。

| 类别 | 字段 |
| --- | --- |
| 记录来源 | `record_id`, `dataset_id`, `study_id`, `table_id`, `source_file`, `source_row` |
| 抗体 | `antibody_id`, `antibody_type`, `heavy_chain`, `light_chain`, `single_chain_sequence` |
| 抗原 | `antigen_key`, `antigen_name`, `antigen_sequence`, `antigen_source` |
| 实验 | `assay_name`, `assay_type`, `metric_name`, `metric_unit` |
| 标签 | `metric_value_raw`, `metric_value_numeric`, `metric_direction`, `transform_rule`, `rank_label`, `label_kind` |
| 训练控制 | `group_id`, `keep_for_training`, `drop_reason` |

### 5.1 标签方向

`rank_label` 是训练和评估使用的统一标签，约定数值越大越好。Kd、IC50、EC50 等“小值更优”的指标，必须在各数据集的 `convert.py` 中完成方向转换。

框架层不会再次猜测指标方向。如果 `transform_rule` 或转换代码写错，模型仍然可以正常运行，但会学到错误的排序。因此，标签方向必须在数据准备阶段用原始论文或原始表格人工核对。

### 5.2 group_id

`group_id` 是排序可比性单位。同一 group 内的记录应具有一致的抗原背景、指标含义和实验条件；跨 group 不构造 pair，也不计算同一个 Spearman。

项目中的常见形式是：

```text
{study_id}/{table_id}/{antigen_key}/{metric_name}/{label_kind}
```

这是一项数据约定，不代表 `group_id` 天然适合作为所有评估任务的切分键。新增数据集时，应分别回答：这条记录可以和谁比较？为了评估目标场景，又必须和谁隔离？

### 5.3 可训练记录

`load_records` 只检查必需列是否齐全，`filter_trainable_records` 再筛选：

- `keep_for_training` 为真；
- `rank_label` 是有限数。

训练层不会在这里重新校验抗体序列。重链是否存在、序列字符是否合法等条件，主要由数据准备脚本和 `validate_processed_table.py` 保证；如果绕过这一步直接把表交给训练层，`filter_trainable_records` 不会自动兜底。

数据准备脚本和表质检目前允许序列中出现 `X`，而 `affinity_transformer/utils.py` 的通用校验只接受 20 种标准氨基酸。两处规则不一致，后续应统一字符集规则，并明确 `X` 是保留、替换还是丢弃。

## 6. 数据准备与质检

每个原始数据集使用独立的 `convert.py`、`test.py` 和 `prepare.sh`。转换代码只处理本数据集的特殊情况，不应把通用训练逻辑复制进去。

完整流程如下：

```text
raw CSV
  -> scripts/prepare/binding/<study>/<table>/convert.py
  -> processed/binding/<study>/<table>/records.parquet
  -> scripts/prepare/validate_processed_table.py
  -> merge_records.py
  -> processed/binding/all_records.parquet
  -> scripts/data/filter_records.py
  -> scripts/data/build_splits.py
  -> processed/binding/splits/<name>/{train,valid,test}.parquet
```

`validate_processed_table.py` 至少检查必需列、ID 唯一性、标签有限性、序列字符、枚举值和来源行号。检查失败时必须返回非零退出码，不能只打印警告后继续训练。

## 7. pair 构造与采样

`affinity_transformer/dataset/pairs.py::build_pairs` 是 pair 构造的统一入口。它只在同一 `group_id` 内配对，标签相同的两条记录不生成 pair。

pair 的方向约定为：

```text
label_i > label_j  ->  y_ij = 1.0
label_i < label_j  ->  y_ij = 0.0
```

当前可配置的采样策略包括：

| 策略 | 说明 |
| --- | --- |
| `absolute_cap` | 每个 group 最多抽取固定数量的 pair |
| `capped_proportional` | 按候选 pair 比例抽样，并设置上下限 |
| `balanced_tree` | 按标签顺序构造平衡树，减少 pair 数量 |
| `randomized_bst` | 构造随机二叉搜索树，降低固定树结构偏差 |

大 group 会走专门的采样逻辑，避免枚举全部 `O(n^2)` 候选对。二分类或只有两个标签值的 group 使用独立逻辑，不强行套用连续标签的分位块。

`pair_sampling/noise_floor_tree.py` 尚未接入 `build_pairs`、配置校验和正式实验，且现有分析记录了已知问题。它目前只能视为实验草稿，不能在 YAML 中启用。

当 `weight_pairs_by_group_size` 开启时，训练会按 group 的记录数和实际采样 pair 数计算权重，避免大 group 因采样上限而失去过多影响力。权重计算和 loader 构造会分别调用一次 `build_pairs`，两次调用的 seed 和采样参数必须完全一致。

## 8. Embedding 子系统

项目有在线编码和离线缓存两条路径。

### 8.1 离线缓存路径

`frozen_cached` 模式先用冻结的预训练模型生成逐残基 embedding，再训练较小的交互与打分网络。这样可以避免每个 batch 重复运行大编码器。

缓存目录包含：

```text
<cache_dir>/
├── metadata.yaml
├── manifest.parquet
└── shards/
    └── shard_00000.pt
```

训练前会校验编码器名称、模型 revision、tokenizer revision、embedding 层、最大长度、长序列策略和序列覆盖率。配置与缓存不一致时应在模型和优化器创建前报错。

抗体缓存键由抗体类型及各链序列共同计算，抗原缓存键由抗原序列计算。ESM2 和 IgBERT 的链输入方式不同，具体格式由各自 extractor 决定，不能假设所有编码器都使用同一种拼接规则。

`ShardedEmbeddingStore` 会把所需 shard 载入内存，并尝试使用 mmap。大缓存的内存占用和多 worker 并发读取仍需要更充分的压力测试。

### 8.2 在线路径

在线路径在训练前向中直接运行编码器。目前主要服务历史 ESM2 配置和现有推理入口。在线路径和缓存路径仍然并存，配置与 checkpoint 不能混用。

## 9. 模型与训练目标

`EmbeddingAffinityRanker` 先把抗体和抗原 embedding 投影到统一维度，模型类本身支持三种结构：

- `antibody_only`：只使用抗体表示；
- `concat`：分别池化抗体和抗原后拼接；
- `deep_cross_attention`：先进行多层抗体—抗原交互，再池化和打分。

但当前 `run_cached_ranknet` 只接入了 `concat` 和 `deep_cross_attention`。`antibody_only` 在模型和 factory 层已有实现，缓存训练入口尚未接通；历史在线训练路径可以使用 antibody-only 配置。文档和实验配置应按 runner 的实际支持范围填写，不能只看模型类是否能构造。

缺失抗原通过 mask 显式处理。无抗原样本不会把全零占位当成有效 token 参与注意力，也不会因为同一个 batch 中其他样本有抗原而改变结果。

在线路径使用 `AffinityRanker`，内部包含在线编码器。两类模型分别由不同 factory 和 runner 构造，不是同一个 checkpoint 的两种加载方式。

`model/losses.py` 已实现 pointwise、RankNet 和 ListNet 三类损失，但当前训练 runner 只完整接入 `pairwise_ranknet`。pointwise 和 listwise 虽然能通过部分配置校验，也有底层函数或数据结构，实际训练会报 `NotImplementedError`。在 runner 接通并补齐端到端测试前，不能把它们列为可用训练目标。

## 10. 配置与训练流程

`affinity_transformer/config.py` 同时兼容两类 YAML：历史扁平配置对应在线编码路径，嵌套配置对应缓存 embedding 路径。`train.py` 根据 `model.antibody_encoder.mode` 选择 runner，再根据 `cross_validation.enabled` 决定跑固定切分还是 K 折。

主调用链如下：

```text
train.py
  -> load_config
  -> resolve_data_paths / run_group_kfold_cross_validation
  -> run_online_training 或 run_cached_ranknet
  -> 读取已切分的 records
  -> 构造 train pairs 和 valid/test record loader
  -> Trainer.fit
  -> 分组评估
  -> 写出 checkpoint、指标和预测结果
```

配置错误应尽量在训练开始前报出。新增配置字段时，要同时检查 dataclass、YAML 解析、合法值校验、runner 支持范围和测试，不能只在其中一处增加字段。

## 11. 评估指标

`compute_group_spearman` 接收逐条预测结果，按 `group_id` 计算 Spearman。代码的具体做法是：

```python
label_rank = group["rank_label"].rank()
score_rank = group["score"].rank()
spearman = label_rank.corr(score_rank)
```

也就是说，先分别计算真实标签排名和预测分数排名，再计算两个排名序列的 Pearson 相关系数；这正是 Spearman 等级相关系数的定义。

以下情况的组内 Spearman 记为 `NaN`，但该 group 仍保留在结果表中：

- 真实标签少于两个不同取值；
- 模型分数为常数，导致相关系数无法定义。

`summarize_group_spearman` 输出两类汇总：

- `macro_spearman`：对有效 group 的 Spearman 做算术平均，每个 group 权重相同，避免少数大 group 主导结果；
- `weighted_spearman`：用各 group 的 `n_records` 对 Spearman 加权，更强调记录数较多的 group。

`weighted_spearman` 仍然是“组内相关系数的加权平均”，不是把所有 records 混在一起计算一个全局 Spearman。跨 group 的标签尺度和实验条件不同，不能直接计算全局相关系数。

二分类标签和连续实验标签的含义不同，必须按 `label_kind` 分开报告。汇总结果还要包含总 group 数、有效 group 数和跳过 group 数，不能只报一个平均值。

任何指标都必须和切分协议一起解释。使用 group holdout 得到的分数不能自动写成“未见抗原性能”；使用组内 record split 得到的分数也不能与抗原隔离结果直接横向比较。

## 12. 推理接口

`affinity_transformer/user_entry.py` 和 `predict.py` 提供竞赛式接口：输入抗原序列和一组候选抗体，输出 `query_id`、`antibody_id`、`score`、`rank` 和 `model_name`。

当前推理接口只支持在线 ESM2 路径。缓存 embedding 路径训练出的 `EmbeddingAffinityRanker` checkpoint 还没有完整的对外推理流程，也不能自动为新抗体和新抗原生成临时 embedding。

这是当前主要交付缺口之一。后续应补一条统一流程：加载 checkpoint 和对应编码器配置，为新序列生成 embedding，完成缓存一致性校验，再调用 `EmbeddingAffinityRanker` 打分。

## 13. 实验产物

一次训练运行至少应保留：

```text
checkpoint.pt
config.yaml
metrics.json
history.csv
run.log
predictions.csv
group_metrics.csv
```

配置 test 集时，还应输出 `test_predictions.csv` 和 `test_group_metrics.csv`。缓存路径另外写出 embedding 元数据引用和资源统计。

批量实验使用 `scripts/runs/`、`scripts/experiments/` 和 `scripts/slurm/` 组织。实验汇总必须保留配置名、随机种子、split 名称、split key、模型结构和指标定义，避免只留下一个无法追溯的分数。

## 14. 编码、测试与验收

公开函数应有 docstring，说明参数、返回值和主要异常。模块开头应说明职责边界，尤其要写清楚本模块不负责什么。私有 helper 使用下划线前缀，不从包的公开接口导出。

核心模块不得在 import 时加载大模型或大数据文件。模型权重、缓存和标准表应在明确的函数调用中按需加载。

提交训练相关改动前，至少运行：

```bash
python -m py_compile train.py predict.py affinity_transformer/*.py \
    affinity_transformer/dataset/*.py \
    affinity_transformer/dataset/pair_sampling/*.py \
    affinity_transformer/embeddings/*.py \
    affinity_transformer/model/*.py \
    affinity_transformer/training/*.py
pytest -q
```

本地缺少 GPU、`transformers` 或模型权重时，应说明哪些测试没有运行及原因，不能用“环境问题”代替验收记录。

下列改动需要额外检查：

- split key、泄露报告和 pair 生成顺序；
- embedding store 的 mmap 与多 worker 读取；
- `group_weights=None` 的无权重回归路径；
- loader 与 group weight 两次调用 `build_pairs` 的参数一致性；
- 新采样策略是否同时接入配置、校验、构造入口和测试；
- checkpoint 与 embedding metadata hash 是否匹配。

## 15. 禁止项

1. 不在 `affinity_transformer/` 中解析某个数据集的原始 CSV、换算单位或翻转标签方向。
2. 不跨 `group_id` 构造 pair。
3. 不先生成 pair 再随机切分 pair。
4. 不把 `debug_record_split` 的结果作为正式模型效果汇报。
5. 不把 group 隔离结果直接称为抗原隔离，除非已经检查 `antigen_key` 或抗原簇没有重叠。
6. 不把二分类 group 和连续标签 group 混成一个无法解释的单一指标。
7. 不在 import 模块时下载或加载预训练模型。
8. 不用隐藏默认值掩盖配置拼写错误。
9. 不把未接入 runner 的功能写成已支持功能。
10. 不在一次改动中同时大规模调整目录结构和训练算法；两类改动应分开验收。

## 16. 当前问题与开发顺序

| 优先级 | 问题 | 建议动作 |
| --- | --- | --- |
| P0 | 主验证只隔离 `group_id`，未严格隔离抗原 | 定义 `antigen_key`/`antigen_cluster_id`，扩展 split key、K 折和泄露报告 |
| P0 | 主线缓存模型没有可用的新样本推理入口 | 打通在线生成 embedding 到 `EmbeddingAffinityRanker` 的推理流程 |
| P1 | 配置允许 pointwise/listwise，但 runner 未接入 | 前置校验不支持的组合，或补齐 loader、训练和端到端测试 |
| P1 | 超大 group 可能长期只在训练集 | 报告固定训练 group，并增加单独留一验证 |
| P1 | 交叉验证有单测但缺少正式实验验证 | 用小规模真实配置跑通完整链路并保存产物 |
| P1 | embedding mmap、group weight 测试不足 | 增加并发读取和无权重回归测试 |
| P2 | 数据准备层和框架层对 `X` 残基规则不一致 | 统一序列字符集与丢弃原因 |
| P2 | README 中的 MSA、Exact + MSA 仍是方案 | 实现前保持为“后续方向”，不要混入现状说明 |
| P2 | `noise_floor_tree` 未接入且有已知问题 | 修正算法、补测试后再决定是否进入配置 |

文档和代码应一起维护。新增模块、切分协议、训练目标或推理路径时，需要同步更新本文件中对应的“当前实现”和“已知问题”，不要继续通过增加新的历史版本文档来代替维护现行说明。
