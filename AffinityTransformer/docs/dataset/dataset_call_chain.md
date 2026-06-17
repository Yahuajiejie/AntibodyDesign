# Dataset 调用链

本文对应 spec v0.6 的 dataset / pair sampler 设计，解释 `affinity_transformer/dataset/` 的函数如何互相调用。

## 1. 标准表进入

调用入口：

```text
load_records(path)
```

调用逻辑：

```text
如果 path 不存在：
  抛 FileNotFoundError

如果 path 后缀是 .parquet：
  pd.read_parquet(path)
否则如果后缀是 .csv：
  pd.read_csv(path)
否则：
  抛 ValueError

检查 REQUIRED_COLUMNS：
  缺任何标准字段都抛 ValueError
```

设计含义：

`dataset` 层只消费已经清洗好的标准表，不负责 raw parser、不负责抗原补全、不负责 label 方向转换。

## 2. 训练记录过滤

调用入口：

```text
filter_trainable_records(records)
```

调用逻辑：

```text
检查 keep_for_training / rank_label 字段

keep_mask:
  用 _parse_bool 严格解析 keep_for_training

finite_mask:
  用 _is_finite_number 检查 rank_label 是有限数值

返回 keep_mask & finite_mask 的记录副本
```

质检标准：

1. `keep_for_training=False` 的记录不能进入训练。
2. `rank_label=NaN/inf/非数值` 的记录不能进入训练。
3. 布尔字段不能用粗暴 `astype(bool)`，因为字符串 `"False"` 会被错误转成 `True`。

## 3. Pairwise 数据构造

调用入口：

```text
build_pairs(records, max_pairs_per_group, seed, ...)
```

调用逻辑：

```text
检查 record_id / group_id / rank_label / label_kind / keep_for_training 字段
校验 pair sampler 参数
trainable = filter_trainable_records(records)

对每个 group_id:
  n_candidates = _candidate_pair_count(group)

  如果 n_candidates == 0:
    跳过

  如果 _should_enumerate_pairs(...) 为 True:
    candidates = _candidate_pairs(group)
    如果候选数超过 n_sample:
      用 seed + group_id 稳定抽样
    写入 pair rows

  否则:
    调用 _sample_large_group_pairs(...)
```

核心禁止项：

1. 禁止跨 `group_id` 构造 pair。
2. 禁止同 label pair 进入训练。
3. 禁止反向重复 pair。
4. `y_ij` 必须由排序后的 `label_i > label_j` 得到。

## 4. 大 group 采样器

调用入口：

```text
_sample_large_group_pairs(...)
```

调用逻辑：

```text
如果 _is_two_label_group(group):
  直接调用 _sample_two_label_group_pairs(...)
  返回

blocks = _build_label_blocks(group, label_block_count)
use_label_buckets = _is_discrete_label_group(...)

先跨 block 抽 inter pairs
再在 block 内补少量 intra pairs
按 pair_id 排序后返回
```

这里最容易看错的一点是：二分类或只有两个 rank label 的 group 不走 5 分位块。它们直接进入 `_sample_two_label_group_pairs`，因为这种数据天然只有两个桶，强行切分位块会制造大量空/同类 pair，既浪费采样次数，也会让代码行为很难解释。

## 5. 双标签 group 采样器

调用入口：

```text
_sample_two_label_group_pairs(...)
```

调用逻辑：

```text
label_to_ids = _label_to_record_ids(group)

如果 label 数不是 2:
  返回空

根据 pair_sample_strategy 算 target
循环随机抽：
  从低 label bucket 抽一个 record
  从高 label bucket 抽一个 record
  用 record_id 排序生成 canonical pair
  去重
  y_ij = 1.0 if label_i > label_j else 0.0
```

科学性含义：

二分类 pair 的有效信息只来自正负类之间的相对顺序；同类内部没有 ranking 信号。因此这条分支只抽跨 label pair。

## 6. 连续或多标签 group 采样器

调用链：

```text
_build_label_blocks
  -> 按 rank_label 排序
  -> 切成 label_block_count 个近似等大小分位块

_sample_from_block_pairs
  -> 在 block pair 之间加权抽样
  -> block 距离越远，权重越高

_sample_within_blocks
  -> 在同一 block 内补少量细粒度 pair
```

当前 v0.6 的边界：

1. 连续 label 还没有按经验分位难度桶做 easy/medium/hard/local 配额。
2. `min_label_diff` 还不是根据 group 分布启发式估计。
3. 这些改进已经适合放进 v0.7 sampler，而不是混进 v0.6 的稳定实现里。

## 7. Listwise 数据构造

调用入口：

```text
build_groups(records, max_group_size, seed)
```

调用逻辑：

```text
trainable = filter_trainable_records(records)

对每个 group_id:
  member_ids = _group_member_ids(group)

  如果 distinct rank_label < 2:
    跳过

  如果 max_group_size 生效且 group 太大:
    用 seed + group_id 稳定采样 member_ids

  输出 group_id / record_id / rank_label / label_kind
```

## 8. Dataset 包装器

调用入口：

```text
AffinityRecordDataset(records)
PairwiseAffinityDataset(records, pairs)
ListwiseAffinityDataset(records, groups)
```

调用逻辑：

```text
AffinityRecordDataset:
  每个 index 返回一个 AffinityExample

PairwiseAffinityDataset:
  每个 index 读取一行 pair
  根据 record_id_i / record_id_j 回查 records
  返回 AffinityPairExample

ListwiseAffinityDataset:
  每个 index 对应一个 group_id
  回查该 group 的所有 records
  返回 AffinityGroupExample
```

组长验收时要看：

1. `pairs` 里的 `record_id_i/j` 必须都能在 `records` 中查到。
2. `groups` 里的 `record_id` 必须都能在 `records` 中查到。
3. `_row_to_example` 不应重新计算 label、group 或 antigen 字段，只做类型转换。
