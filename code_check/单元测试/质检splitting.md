# splitting/
## results.py

```python
@dataclass

class WithinAntigenSplitResult:
  
"""Output of `build_within_antigen_split` (programming_spec_v1.0.md 3.2,
"known-antigen, new-antibody")."""

train: pd.DataFrame
valid: pd.DataFrame
test: pd.DataFrame
summary: pd.DataFrame
leakage_report: pd.DataFrame
pinned_groups: pd.DataFrame
```

这是存放 `WithinAntigenSplitResult` 结果的规范。在代码实现的过程中重点检查，针对某一类抗原的抗体是否会同时出现在该抗原的训练集、验证集及测试集里


```python
@dataclass(frozen=True)

class GroupFold:

"""One group-isolated cross-validation fold."""

index: int
train: pd.DataFrame
valid: pd.DataFrame
```
这是K折交叉验证中，存放每一折训练集和验证集的数据结构

```python
@dataclass

class EntityColdStartSplitResult:

"""Fixed train/valid/test artifacts for one strict entity protocol."""


protocol: str
train: pd.DataFrame
valid: pd.DataFrame
test: pd.DataFrame
summary: pd.DataFrame
leakage_report: pd.DataFrame
eligibility_report: pd.DataFrame
excluded_records: pd.DataFrame
unit_assignments: pd.DataFrame


@dataclass

class EntityColdStartFold:

"""One protocol-aware development fold; final test never rotates here."""

  

protocol: str
index: int
train: pd.DataFrame
valid: pd.DataFrame
leakage_report: pd.DataFrame
eligibility_report: pd.DataFrame
excluded_records: pd.DataFrame
unit_assignments: pd.DataFrame
```

这两种数据结构应该是给Within-antigen antibody holdout、Antigen-cluster holdout这两种协议下的基础数据分割 及 K折交叉验证做的准备

```python
@dataclass

class DualColdStartSplitResult:

"""Fixed train/valid/test artifacts for the Dual cold-start protocol.

Same fields as an entity split plus ``component_summary`` -- the preflight
per-component feasibility statistics (largest-component first).

"""

  

protocol: str
train: pd.DataFrame
valid: pd.DataFrame
test: pd.DataFrame
summary: pd.DataFrame
leakage_report: pd.DataFrame
eligibility_report: pd.DataFrame
excluded_records: pd.DataFrame
unit_assignments: pd.DataFrame
component_summary: pd.DataFrame
```
DualColdStartSplitResult的数据结构

```python
def write_splits(result: SplitResult, output_dir: Path) -> None:
"""Write split records and QC reports to `output_dir`."""

def write_within_antigen_split(result: WithinAntigenSplitResult, output_dir: Path) -> None:
"""Write within-antigen split records and QC reports to `output_dir`."""

def write_entity_cold_start_split(
result: EntityColdStartSplitResult,
output_dir: Path,
) -> None:
"""Write one strict entity-protocol split and all audit artifacts."""
```
这些函数很简单，就是把上面定义的数据结构转化为实体报表

```python
def build_antibody_cold_start_manifest(
*,
seed: int,
valid_fraction: float,
test_fraction: float,
min_eval_records: int,
require_train_group: bool,
input_records_hash: str,
entity_annotations_hash: str,
) -> dict[str, object]:
"""Build the small ``split_manifest.yaml`` payload for one split."""
```
这个函数 `build_antibody_cold_start_manifest` 的主要作用是生成一个用于描述“抗体冷启动（Antibody Cold Start）”数据拆分（Split）配置的元数据字典


```python
def _strip_entity_columns(frame: pd.DataFrame) -> pd.DataFrame:

"""Drop cold-start identity columns so base split files keep base schema."""
```
分割过程需要扩张表格，而该函数就是在分割结束后舍弃临时列


## common.py

```python
def frame_hash(frame: pd.DataFrame) -> str:

"""Deterministic content hash of a DataFrame (column-order independent)."""

canonical = frame.reindex(sorted(frame.columns), axis=1).astype(str)

# 先把表格的列按照列名排序，消除表格列顺序差异造成的影响

row_hashes = pd.util.hash_pandas_object(canonical, index=False)

#  会对每一行的数据计算一个哈希值，并忽略 DataFrame 的索引（Index

return hashlib.sha256(row_hashes.values.tobytes()).hexdigest()

最后，它将所有行的哈希值转换为字节流（`.tobytes()`），并使用 SHA-256 算法计算出一个最终的、全局的十六进制字符串。
```

逐行哈希函数

```python
def derive_link_components(

records: pd.DataFrame,

link_columns: tuple[str, ...],

) -> pd.Series:

"""Assign each record to a connected component via union-find over links.

  

Records sharing any value in any of ``link_columns`` are unioned; the full

transitive closure forms one indivisible component. The component id is a

deterministic hash of its sorted ``record_id`` set, so the labelling is

stable across runs and input orderings.


Returns:

A Series (aligned to ``records.index``) of ``component_<hash>`` ids.

"""

...
for position, record_id in enumerate(records["record_id"].astype(str)):

root_to_record_ids.setdefault(find(position), []).append(record_id)

root_to_component = {

# 对于每一棵树的树根 for root, record_ids in root_to_record_ids.items()

# 将这棵树的所有节点，排序，聚合，生成哈希字符串，截取字符串，然后生成属于这棵树的标识，用于标记当前record分属于哪个group
root: "component_" + hashlib.sha256(
"\n".join(sorted(record_ids)).encode("utf-8") 
).hexdigest()[:16]

for root, record_ids in root_to_record_ids.items()
}

return pd.Series(

[root_to_component[find(position)] for position in range(n_records)],

index=records.index,

)
```
这个函数试图在处理如下情况：

| record | antibody_cluster_id | measurement_family_id | antigen_cluster_id |
| ------ | ------------------- | --------------------- | ------------------ |
| r1     | abA                 | mf1                   | ag1                |
| r2     | abA                 | mf2                   | ag2                |
| r3     | abB                 | mf2                   | ag3                |
| r4     | abC                 | mf3                   | ag3                |
先按不同字段看：
```
antibody_cluster_id = abA  → {r1, r2}
measurement_family_id = mf2 → {r2, r3}
antigen_cluster_id = ag3    → {r3, r4}
```
这些集合两两不是完全一样，但它们链式相交：

```
{r1, r2} 和 {r2, r3} 有交集 r2
{r2, r3} 和 {r3, r4} 有交集 r3
```
所以最后必须合并成：

```
{r1, r2, r3, r4}
```

link_columns: tuple[str, ...],这个参数很重要，我相信ai写的代码实现应当不会有严重错误。最有可能犯错的地方就是column的指定了，不同的分割策略应当要指定不同的column

- 对于antibody cold-start
```python
(
    "antibody_cluster_id",
    "antibody_sequence_key",
    "measurement_family_id",
)
```
会合并:
- 同一个抗体簇；
- 同一个精确抗体序列；
这两者是显然的，而合并：
- 同一个 measurement family。
这有待商榷，因为这里最危险的合并来源其实就是 `measurement_family_id`。如果它太粗，比如把一个 dataset、一个 assay、甚至一个 study 都写成同一个 measurement family，那大量 records 会被合成一个超级 component。
它不会因为同一个抗原合并。因为 antibody cold-start 的目标就是：
```
新抗体，已知抗原
```
所以抗原可以在 train/valid/test 之间出现；它不是切分隔离对象

- 对于 antigen cold-start 的 component link 是：
```
(
    "antigen_cluster_id",
    "measurement_family_id",
)
```
所以会合并：
- 同一个抗原簇；
- 同一个 measurement family。
不会因为同一个 antibody cluster 合并。这个是故意的，因为 antigen cold-start 需要：
```
新抗原，已知抗体
```
valid/test 里的抗体簇应该在 train 见过，所以 antibody cluster 跨 split 不是泄露，反而是协议要求。

也不会因为 `effective_antigen_input_hash` 合并。这个也很重要：effective input collision 是 audit 项，不是 component link。也就是说，如果两个不同抗原截断后模型输入一样，代码不是把它们合并，而是在 leakage audit 里报错。

- 对于dual ，它的 link columns 是：

```
(
    "antibody_cluster_id",
    "antigen_cluster_id",
    "measurement_family_id",
    "interaction_key",
)
```

所以会合并：

- 同一个抗体簇；
- 同一个抗原簇；
- 同一个 measurement family；
- 同一个 interaction；
- 以及它们造成的所有链式连接。

dual 是最容易出现巨大 component 的，因为它同时从抗体侧、抗原侧、实验重复侧、interaction 侧连边。

`measurement_family_id` 的语义由 annotation 生成阶段决定；split 代码只把它当成一个已经给好的、不透明的字符串 ID 来消费。

所以如何以合适的方式指定family_id非常重要，函数留下这个接口就是为了增加灵活性

```python
def _assign_component_splits(
records: pd.DataFrame,
*,
train_units: set[str],
valid_units: set[str],
test_units: set[str],
) -> pd.DataFrame:
```
把一个组，或者说一个连通分量（上文有提到这个概念），分配给一个数据集


```python
def _assign_weighted_units_to_folds(

weights: dict[str, int],
n_splits: int,
seed: int,

) -> list[set[str]]:
```
该函数尝试将一系列不可分割的组尽可能均匀地分配给不同的折。weight是衡量组均匀度的指标，一般设置为集合大小


```python
def _partition_units(

units: list[str],
valid_fraction: float,
test_fraction: float,
seed: int,

) -> tuple[set[str], set[str], set[str]]:
```

将数据单元（比如我们前面提到的连通分量 `component_id`）按比例切分为训练集（Train）、验证集（Valid）和测试集（Test）。

这个函数为了应对极端数据量和浮点数精度问题，加入了一系列非常严谨的处理逻辑


```python
def _partition_weighted_units(

weights: dict[str, int],

valid_fraction: float,

test_fraction: float,

seed: int,

) -> tuple[set[str], set[str], set[str]]:

"""Partition group ids while keeping over-target groups in train."""
```
此函数为上一个函数的改版，在计算训练集、验证集、测试集数量后，它还会根据权重决定每个集合的去向，确保验证集、测试集尽可能均匀


## audits.py

主要是定义了几种报表的格式

一上来，文件就定义了需要用到的表头：

```python
SPLIT_COLUMNS = ("split", "strategy", "n_records", "n_trainable_records", "n_groups",
"n_trainable_groups", "n_spearman_eligible_groups",
"label_kind_counts", "antigen_source_counts")

LEAKAGE_COLUMNS = ("check_name", "status", "n_violations", "details")
  
PINNED_GROUPS_COLUMNS = ("group_id", "n_records", "n_antibody_units", "reason")

ENTITY_UNIT_COLUMNS = (
"component_id", "assigned_split", "validation_fold", "n_records",
"n_entity_clusters", "n_measurement_families",
)

ELIGIBILITY_COLUMNS = (
"split", "group_id", "n_assigned_records", "n_candidate_records",
"n_eligible_records", "n_unique_inputs", "n_unique_labels", "status", "reason",
)
```


## dispatch.py

很简单，最重要的就是一个将数据注释总表转化为train/valid/test分割表的函数


## entity_cold_start.py

这个文件实现的不是项目总文档里的 `Within-antigen` 和 `Antigen-cluster` 主协议。

更准确地说，它实现的是两套 **global entity holdout**，并额外保留一套可选 strict 模式：

1. `antibody_cold_start` 默认问：“训练集中完全没见过的抗体簇，整体表现会不会失真”，不强制要求 paired antigen 在 train 见过。
2. `antigen_cold_start` 默认问：“训练集中完全没见过的抗原簇，整体表现会不会失真”，不强制要求 paired antibody 在 train 见过。
3. 当 `strict_known_counterpart=True` 时，才切换到 Claude 原先写死的受控口径：
   - antibody strict：未见抗体 + 已见抗原/可选已见 group；
   - antigen strict：未见抗原 + 已见抗体。

这里需要特别标记 Claude 原先的思路混乱点：他把 `entity_cold_start.py` 写成了 antibody/antigen 两种协议的共用框架，又默认强制做 known-counterpart 后置筛选。这会把“全局未见实体 holdout”偷偷改成更窄的“未见实体 + 另一侧已见”评估口径，并且会直接丢弃一部分 valid/test records。现在代码已经改成：默认不启用 known-counterpart 筛选，只保留 group 可评估性检查；strict 模式才调用原来的 `_select_protocol_eligible_records`。

但是，经过研判，entity_cold_start.py 不应作为当前四协议主线实现。
它更像是 antibody/antigen global entity holdout 的实验性共用框架。
其中 `_select_protocol_eligible_records` 只应作为 strict 模式的可选筛选器，不能作为默认主流程。
当前主线应优先审 within_antigen.py、antigen_cluster.py、dual_cold_start.py。

所以审查这个文件时，不要把它和下面两个文件混在一起：

| 项目总文档协议 | 主要实现文件 |
|---|---|
| Within-antigen | `within_antigen.py` |
| Antigen-cluster | `antigen_cluster.py` |
| global antibody/antigen entity holdout + optional strict mode | `entity_cold_start.py` |

`entity_cold_start.py` 的实现思路是：

```text
records / entity annotations
→ 补齐或 join identity 字段
→ 过滤 trainable records
→ 构造不可拆 entity component
→ 按 component 分配 train / valid / test
→ 默认只检查 holdout group 是否可评估
→ strict 模式才筛出 known-counterpart records
→ 生成 leakage / eligibility / excluded / unit assignment 报告
```

这里最容易看晕的地方是：它不是“每个 group 内切抗体”的逻辑，而是“全局实体隔离”的逻辑。

### 核心入口函数

```python
def build_antibody_cold_start_split(...)
```

核心问题：

```text
默认：valid/test 的抗体簇在整个 train 中完全没出现过；
不要求 paired antigen 一定在 train 见过。

strict_known_counterpart=True：
valid/test 的抗体簇全局未见，同时要求 paired antigen 在 train 见过；
require_train_group=True 时，还要求 group_id 在 train 见过。
```

它不是 `Within-antigen`。`Within-antigen` 是在同一个 `group_id` 内切抗体；这里是全局隔离抗体簇。

异常情况：

- valid/test 的 `antibody_cluster_id` 是否真的不出现在 train；
- 默认模式下，不应因为 `antigen_sequence_key` 没在 train 出现而丢弃 records；
- strict 模式下，才应检查 `antigen_sequence_key` 是否在 train 出现；
- strict + `require_train_group=True` 时，才应要求 `group_id` 在 train 出现；
- 被排除的 records 是否进入 `excluded_records`，且原因可读。

```python
def build_antigen_cold_start_split(...)
```

核心问题：

```text
默认：valid/test 的抗原簇在 train 中完全没出现过；
不要求 paired antibody 在 train 见过。

strict_known_counterpart=True：
valid/test 的抗原簇全局未见，同时要求 paired antibody cluster 在 train 见过。
```

它不是普通 `Antigen-cluster holdout`。普通 antigen-cluster 只关心新抗原簇；这里是 global antigen entity holdout，并且只有 strict 模式才会筛掉 train 中没见过的抗体簇。

异常情况：

- valid/test 的 `antigen_cluster_id` 是否真的不出现在 train；
- 默认模式下，不应因为 `antibody_cluster_id` 没在 train 出现而丢弃 records；
- strict 模式下，才应要求 valid/test 的 `antibody_cluster_id` 都能在 train 找到；
- 如果传入 `representation_annotations`，是否会检查 `effective_antigen_input_hash`；
- 如果没传入 representation annotation，manifest 或审查记录里是否明确“未做 effective-input audit”。

### 核心流程函数

```python
def _resolve_cold_start_inputs(...)
```

这个函数解决输入来源问题：

```text
如果传 entity_annotations，就临时 join 到 records；
如果不传，就走 legacy embedded-column 路径；
如果 antigen protocol 传了 representation_annotations，就临时补 effective_antigen_input_hash。
```

质检重点不是逐行看 merge，而是确认：

- annotation 只是临时 join，不应污染最终 train/valid/test base schema；
- antibody protocol 不应强行要求 representation annotation；
- representation annotation 只用于 effective-input audit，不应变成 component link。

```python
def _prepare_cold_start_records(records, protocol)
```

这个函数是正式切分前的清洗和一致性检查。

它主要做四类事：

1. 检查协议所需 identity columns 是否存在；
2. 只保留 `keep_for_training=True` 且 `rank_label` 可用的 records；
3. 检查 identity 字段非空；
4. 检查 exact sequence 到 cluster 的映射一致性。

这里的关键检查是：

```text
antibody_sequence_key → antibody_cluster_id 必须唯一
antigen_sequence_key  → antigen_cluster_id 必须唯一（当协议需要 antigen_cluster_id 时）
```

意思是：同一个精确序列不能一会儿属于 `cluster_A`，一会儿属于 `cluster_B`。

但反方向不要求唯一：

```text
一个 cluster 可以包含多个相似但不完全相同的 sequence_key
```

质检重点：

- 不要把“sequence_key → cluster_id 唯一”误解成“cluster_id → sequence_key 唯一”；
- `measurement_family_id` 只在这里被要求存在、非空；它的真实生成规则不在本文件里；
- 如果当前数据还没有正式 entity annotation 生成脚本，这里只能消费字段，不能证明字段语义正确。

```python
def _derive_entity_components(records, protocol)
```

这个函数决定 strict antibody/antigen 协议中“哪些 records 必须绑在一起”。

当前规则：

```python
antibody_cold_start:
    antibody_cluster_id
    antibody_sequence_key
    measurement_family_id

antigen_cold_start:
    antigen_cluster_id
    measurement_family_id
```


注意事项：
无论当前策略是不是antigen holdout，都不应该存在antibody同时位于两个cluster
质检重点：

- antibody cold-start 不应因为同一个抗原而合并，否则会变成 antigen/group holdout；
- antigen cold-start 不应因为同一个抗体簇而合并，因为已知抗体跨 split 是协议要求；
- `measurement_family_id` 是最危险的连接边：如果它太粗，会制造巨大 component；
- 这里和 `common.py::derive_link_components` 底层思想一样，只是这个函数把 link columns 写死在 antibody/antigen 协议里。

```python
def _build_entity_cold_start_split(...)
```

这是 fixed train/valid/test 的核心编排函数。

它的流程是：

```text
prepare records
→ derive components
→ 用 component size 做 weighted partition
→ assign component to train/valid/test
→ 对 valid/test 做 holdout record selection
   - 默认：只做 group evaluability 检查
   - strict：调用 _select_protocol_eligible_records 做 known-counterpart 筛选
→ leakage audit
→ 汇总 eligibility/excluded/unit assignment
→ 返回 EntityColdStartSplitResult
```

这个函数里最需要审查的是：它是 **先按 component 切，再选择 holdout records**。

默认模式下，它不再强行修正为 known-counterpart 口径；只有 `strict_known_counterpart=True` 时，才会启用 Claude 原先写的后置筛选。

质检重点：

- 默认模式下，raw valid/test 中另一侧未见的 records 仍应保留；
- strict 模式下，另一侧未见的 records 不应偷偷进入 valid/test 指标，而应写入 `excluded_records`；
- 如果筛完以后 valid 或 test 为空，应 fail，不应继续导出伪 split；
- leakage audit 是否随 `strict_known_counterpart` 切换：默认不应检查 `*_seen_in_train` coverage；strict 才检查。

```python
def _select_entity_holdout_records(...)
```

这是当前新加的包装选择函数，也是审查默认/strict 语义的关键位置。

它的逻辑应该是：

```text
strict_known_counterpart=False：
    不调用 _select_protocol_eligible_records；
    只调用 build_group_eligibility 检查 holdout group 是否能用于 ranking。

strict_known_counterpart=True：
    调用 _select_protocol_eligible_records；
    同时做 known-counterpart 筛选和 group evaluability 检查。
```

质检重点：

- 默认模式不应产生 `antigen_sequence_not_seen_in_train` 或 `antibody_cluster_not_seen_in_train` 的排除原因；
- 默认模式的 leakage report 不应包含 `valid_antibody_seen_in_train`、`valid_antigen_seen_in_train` 这类 coverage 检查；
- strict 模式必须保留旧行为，便于需要受控变量时显式启用。

```python
def _select_protocol_eligible_records(...)
```

这个函数不是默认主流程，而是 **strict 模式专用的 known-counterpart 筛选器**。

它试图解决的问题是：

```text
component 已经被分到 valid/test；
strict 模式下，其中某些 records 不再是“未见实体 + 另一侧已见”的受控评估问题。
```

对于 `antibody_cold_start`：

- valid/test 抗体是新的，这由 component split 保证；
- strict 模式下，valid/test 的抗原必须在 train 见过；
- 如果 `require_train_group=True`，valid/test 的 group 也必须在 train 见过。

对于 `antigen_cold_start`：

- valid/test 抗原是新的，这由 component split 保证；
- strict 模式下，valid/test 的抗体簇必须在 train 见过；
- 否则这条 record 就变成“新抗原 + 新抗体”，更接近 dual。

然后它还检查 holdout group 是否能用于 ranking evaluation：

```text
候选 records 数 >= min_eval_records
不同 antibody input 数 >= 2
不同 label 数 >= 2
```

质检重点：

- 这个函数不应在默认模式调用；
- 如果 strict 模式 excluded 太多，说明受控口径在真实数据上可用性差，不能只看 leakage PASS；
- `eligibility_report` 和 `excluded_records` 是判断 split 是否真的可用的关键，不是附属报表。

```python
def _build_entity_cold_start_kfolds(...)
```

这是 strict antibody/antigen 的 K 折版本。

它没有 final test 轮转，只是把 component 分到不同 validation folds：

```text
每一折：
当前 fold components → raw_valid
其他 components → raw_train
再做 protocol eligibility 和 leakage audit
```

质检重点：

- fold assignment 是否仍按 component，不拆不可拆实体；
- 每折 valid 是否真的有 protocol-eligible records；
- 默认 K 折不应因为另一侧实体未见而丢掉 records；
- strict K 折才应该要求另一侧实体 train-seen；
- `unit_assignments` 是否能追溯 component 属于哪一折；
- K 折结果不要和 fixed train/valid/test 协议混报。

### 本文件的核心风险

1. **协议命名风险**

   `antibody_cold_start` 容易被误写成 `Within-antigen`，但它们不是一回事。

   ```text
   Within-antigen：每个 group 内切抗体
   antibody_cold_start：全局隔离抗体簇
   ```

2. **measurement_family_id 风险**

   本文件只消费 `measurement_family_id`，不生成它。

   如果上游把它写得太粗，例如 `study_id`、`table_id`、`assay_name`、`unknown`，就会把大量 records 合并成巨大 component。

3. **strict eligibility 之后数据缩水风险**

   component split 成功不代表协议可用。

   默认模式只做 group evaluability 检查；strict 模式才会经过 `_select_protocol_eligible_records`。如果 strict valid/test 剩很少，甚至只剩很少 group，说明受控口径的评估价值会很弱。

4. **和主协议混用风险**

   如果当前目标是跑项目总文档里的四种协议，应优先看：

   ```text
   group.py
   within_antigen.py
   antigen_cluster.py
   dual_cold_start.py
   ```

   `entity_cold_start.py` 可以作为额外 strict entity protocol 审查，不应替代 `within_antigen.py` 或 `antigen_cluster.py`。
