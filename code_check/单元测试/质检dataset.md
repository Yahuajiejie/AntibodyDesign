# dataset/
## dataset/__init__.py

这个module主要是定义了向外暴露什么模块

## dataset/schema.py

`REQUIRED_COLUMNS`、`_EXAMPLE_COLUMNS`、`PAIR_COLUMNS`、`GROUP_COLUMNS`均符合文档规范要求

```python
_BINARY_LABEL_KIND = "binary"   # 表示 label_kind == "binary" 的二分类标签。这类 group 只应该抽正负类之间的 pair，不应该在同类内部抽 pair，也不应该强行切成 5 个分位块。

_DEFAULT_LARGE_GROUP_THRESHOLD = 10_000   # 一个 group 里 trainable records 数量达到 10000 时，视为大 group。大 group 不再枚举所有 pair，而是走分块采样。否则 n*(n-1)/2 会爆炸。


_DEFAULT_PAIR_ENUMERATION_LIMIT = 100_000  

_DEFAULT_LABEL_BLOCK_COUNT = 5

_DEFAULT_INTRA_BLOCK_PAIRS_PER_LARGE_GROUP = 50


_DEFAULT_DISCRETE_LABEL_UNIQUE_THRESHOLD = 32    # 如果一个 group 的唯一 label 数量不超过 阈值，就认为它更像离散标签 group。

#这种 group的亲和力数据通常通过多分类标签来表示：不结合 / 弱结合 / 中等 / 强 / 很强

_DEFAULT_DISCRETE_LABEL_RATIO_THRESHOLD = 0.05   # 如果独特的标签类别占总数很小， 那就意味着很可能仍然是离散的数据
```

## dataset/records.py

```python
def load_records(path: Path) -> pd.DataFrame:
    """Load a standard processed table and check required columns."""
```

检查主要包括以下方面：
- 能不能检测出路径缺失
- 能不能检测出缺失必要列的表格
- 会不会修改原始表格
- 能不能检测出非parquet及非csv文件

```python
def filter_trainable_records(records: pd.DataFrame) -> pd.DataFrame:
    """Keep only records with ``keep_for_training=True`` and finite label."""
```
检查主要包括以下两个方面：
- train records只包含keep for training = True 和 rank_label 有限的数值
- 该函数同样不能修改原始表格
- 该函数同样得拒绝缺失的行
相关自定义子函数：
```python
def _is_finite_number(value: object) -> bool:
    """Return True if ``value`` can be interpreted as a finite float."""
```
- 能处理数值不存在的情况
- 能处理无法转换为float的情况（float()函数自带质检？)
- 能处理不能表示为非无穷数值的情况
```python
def _parse_bool(value: object) -> bool:
    """Parse a standard-table boolean cell strictly."""
```
- 现在这个函数已经能处理字符串“false”被翻译为True（非空字符串）的情况了

## dataset/examples.py

除了为分块取样法构造的数据结构之外，其他数据结构均符合文档定义，且被成功frozen，确保一旦生成数据结构的对象，其数值就不能被修改

需要关注的数据结构：
```python
@dataclass(frozen=True)
class _LabelBlock:
    """Rank-label quantile block used by the large-group pair sampler."""

    index: int
    items: tuple[tuple[str, float], ...]
    label_to_ids: dict[float, tuple[str, ...]]
    n_records: int  
```


## dataset/pairs.py
```python
def build_pairs(
    records: pd.DataFrame,
    max_pairs_per_group: int,
    seed: int,
    pair_sample_strategy: str = "absolute_cap",
    pair_fraction: float | None = None,
    min_pairs_per_group: int = 1,
    large_group_threshold: int = _DEFAULT_LARGE_GROUP_THRESHOLD,
    pair_enumeration_limit: int = _DEFAULT_PAIR_ENUMERATION_LIMIT,
    label_block_count: int = _DEFAULT_LABEL_BLOCK_COUNT,
    intra_block_pairs_per_large_group: int = _DEFAULT_INTRA_BLOCK_PAIRS_PER_LARGE_GROUP,
    discrete_label_unique_threshold: int = _DEFAULT_DISCRETE_LABEL_UNIQUE_THRESHOLD,
    discrete_label_ratio_threshold: float = _DEFAULT_DISCRETE_LABEL_RATIO_THRESHOLD,
) -> pd.DataFrame:

    """Build pairwise ranking examples within each group."""
```

关键函数，实现逻辑如下：

1. 首先检查有没有缺失pair必须的行，如果有，马上报错
2. 然后调用`_validate_pair_sampling`，检查调用者是否填入了合适的参数，只要参数不合适，该子函数马上就会raise Error
3. `filter_trainable_records`提取所有可以用于训练的记录
4. 针对每一个group：`_candidate_pair_count`会计算所有可能的pairs数目，并依据此做出分布决策
- 如果group不存在可用pair，就跳过这一group
- 如果当前group的数据量较小，就直接进行全局配对+采样：
	- `_cadidate_poirs`进行全局配对，返回所有可能的配对
	- `_pair_sample_count`则决定采样的数量
		- 如果策略是"absolute_cap”，直接取 min(n_candidates, max_pairs_per_group)
		- 如果策略是"capped_propotional”,target = max(min_pairs_per_group, ceil(候选对数 * pair_fraction));最终对数 = min(候选对数, max_pairs_per_group, target)
	- 使用随机采样器，采出相应数量的配对
- 如果group数据量较大，就调用_sample_large_group_pairs

质检项目：

- build_pairs不能跨越组别
- build_pairs 不能把相同数据点放在同一组里 “自己跟自己比”
- build_pairs 必须自己跟自己比
- 采用默认策略"absolute_cap”，指定max_pair_per_group时，必须不超上限
- 采用策略”capped_propotional”时，得到的结果必须能够满足相应比例
- 采用策略”capped_propotional”时，必须指定比例
- 参数检查函数必须正确工作，比如，不准乱设上限
- 没有可训练对时，不构建pairs
- 对于大数据集，不准调用全局列举函数
- 一个大数据集的采样结果必须包含跨块的和同一块以内的
- 随机采样器的结果在同一种子下必须可复现
- 对于一个二进制数据集，当处理超大的 binary group，即使样本很多、类别极不平衡，也不能走连续标签的分位块采样，而必须只抽正负类之间的 pair

相关子函数（不包含在pair_sampling）里的函数：
```python
def _candidate_pairs(group: pd.DataFrame) -> list[tuple[str, str, float, float, float]]:

    """Enumerate valid unordered candidate pairs within one group."""
```
返回所有的配对方式

## dataset/groups.py

```python
def build_groups(

    records: pd.DataFrame,

    max_group_size: int | None,

    seed: int,

) -> pd.DataFrame:

    """Build listwise ranking groups."""
```
实现思路：

1. 检查表格和相关参数合不合规

- 参数max_group_size 必须大于等于2，因为一个组至少两个数据才能训练
- 表格必须要有相关行

1. 获得所有可训练的数据
2. 遍历表格里所有的group

- 先找找该group里可以用于训练的样本有哪些 一个样本对应一个record_id = f"{STUDY_ID}/{TABLE_ID}/{source_row}"
- 如果没有的话，继续
- 如果超过了一个组所能容纳的样本数上限，就随机抽到满足为止

## dataset/pair_sampling/

| **函数名**                       | **定义所在文件**                   | **调用者**                        | **调用者所在文件**                           |
| ----------------------------- | ---------------------------- | ------------------------------ | ------------------------------------- |
| _pair_row                     | pair_sampling/common.py      | build_pairs                    | affinity_transformer/dataset/pairs.py |
| _pair_row                     | pair_sampling/common.py      | _sample_two_label_group_pairs  | pair_sampling/two_label.py            |
| _pair_row                     | pair_sampling/common.py      | _sample_until_target           | pair_sampling/blocks.py               |
| _candidate_pair_count         | pair_sampling/common.py      | build_pairs                    | affinity_transformer/dataset/pairs.py |
| _should_enumerate_pairs       | pair_sampling/common.py      | build_pairs                    | affinity_transformer/dataset/pairs.py |
| _canonical_pair               | pair_sampling/common.py      | _sample_two_label_group_pairs  | pair_sampling/two_label.py            |
| _canonical_pair               | pair_sampling/common.py      | _sample_until_target           | pair_sampling/blocks.py               |
| _weighted_choice              | pair_sampling/common.py      | _draw_between_blocks           | pair_sampling/blocks.py               |
| _weighted_choice              | pair_sampling/common.py      | _draw_within_block             | pair_sampling/blocks.py               |
| _weighted_choice              | pair_sampling/common.py      | _weighted_label_excluding      | pair_sampling/blocks.py               |
| _pair_sample_count            | pair_sampling/common.py      | build_pairs                    | affinity_transformer/dataset/pairs.py |
| _pair_sample_count            | pair_sampling/common.py      | _sample_large_group_pairs      | pair_sampling/large_group.py          |
| _pair_sample_count            | pair_sampling/common.py      | _sample_two_label_group_pairs  | pair_sampling/two_label.py            |
| _validate_pair_sampling       | pair_sampling/validation.py  | build_pairs                    | affinity_transformer/dataset/pairs.py |
| _is_discrete_label_group      | pair_sampling/labels.py      | _sample_large_group_pairs      | pair_sampling/large_group.py          |
| _is_two_label_group           | pair_sampling/labels.py      | _sample_large_group_pairs      | pair_sampling/large_group.py          |
| _label_to_record_ids          | pair_sampling/labels.py      | _sample_two_label_group_pairs  | pair_sampling/two_label.py            |
| _sample_two_label_group_pairs | pair_sampling/two_label.py   | _sample_large_group_pairs      | pair_sampling/large_group.py          |
| _sample_large_group_pairs     | pair_sampling/large_group.py | build_pairs                    | affinity_transformer/dataset/pairs.py |
| _build_label_blocks           | pair_sampling/blocks.py      | _sample_large_group_pairs      | pair_sampling/large_group.py          |
| _sample_from_block_pairs      | pair_sampling/blocks.py      | _sample_large_group_pairs      | pair_sampling/large_group.py          |
| _sample_within_blocks         | pair_sampling/blocks.py      | _sample_large_group_pairs      | pair_sampling/large_group.py          |
| _sample_until_target          | pair_sampling/blocks.py      | _sample_from_block_pairs       | pair_sampling/blocks.py               |
| _sample_until_target          | pair_sampling/blocks.py      | _sample_within_blocks          | pair_sampling/blocks.py               |
| _draw_between_blocks          | pair_sampling/blocks.py      | _sample_until_target 的 draw 回调 | pair_sampling/blocks.py               |
| _draw_within_block            | pair_sampling/blocks.py      | _sample_until_target 的 draw 回调 | pair_sampling/blocks.py               |
| _weighted_label_excluding     | pair_sampling/blocks.py      | _draw_between_blocks           | pair_sampling/blocks.py               |
| _weighted_label_excluding     | pair_sampling/blocks.py      | _draw_within_block             | pair_sampling/blocks.py               |
| _cross_block_candidate_count  | pair_sampling/blocks.py      | _sample_from_block_pairs       | pair_sampling/blocks.py               |
| _within_block_candidate_count | pair_sampling/blocks.py      | _sample_within_blocks          | pair_sampling/blocks.py               |

## dataset/pair_sampling/blocks.py

Blocks采样方法是一种适用于大数据组的近似采样算法，用于从按rank_label排序的数据组中切割成若干块，然后从"跨块"和"块内"随机抽取标签不同的样本对，直到凑够目标数量。该方法通过分块策略将时间复杂度从O(n²)降低到更可接受的水平，避免了直接枚举所有样本对在大数据量下的性能问题。
### 整体流程
#### 1. 数据准备阶段

假设一个group中有以下样本：

```
record_id    rank_label
A            0.1
B            0.2
C            0.2
D            0.6
E            0.8
F            0.9
```

#### 2. 分块策略

按照标签排序后切块：

- **block 0**: A(0.1), B(0.2)
  
- **block 1**: C(0.2), D(0.6)
  
- **block 2**: E(0.8), F(0.9)
  

#### 3. 抽样方式

**跨块抽样**：从不同块中各抽一个样本

- 示例：A(0.1)和F(0.9)
  

**块内抽样**：在同一个块内抽两个样本

- 示例：C(0.2)和D(0.6)
  

#### 4. 后处理步骤

1. 排除标签相同的pair
   
2. 排除已经抽过的pair
   
3. 统一pair的方向
   
4. 生成训练数据

### 核心函数详解

#### `_build_label_blocks` - 排序并分块

**功能**：将一个group按标签排序，然后平均切成若干块。

**实现步骤**：

1. **排序样本**

```python
items = sorted(
    zip(group["record_id"].astype(str), group["rank_label"].astype(float)),
    key=lambda item: (item[1], item[0]),
)
```

- 按(label, record_id)排序
- 优先按rank_label排序，标签相同时按record_id排序
- 二次排序保证结果稳定、可复现

2. **决定实际块数**

```python
n_blocks = min(label_block_count, len(items))
```

- 块数不能超过样本数
- 避免出现空块

3. **均匀切块**

```python
start = block_index * len(items) // n_blocks
end = (block_index + 1) * len(items) // n_blocks
```

- 典型的整数均分方式
- 块大小差距最多为1
- 前面的块标签整体较低，后面的块标签整体较高

4. **建立标签索引**

```python
label_to_ids.setdefault(label, []).append(record_id)
```

- 一个块内部可能有多个样本拥有同一个标签
- 索引用于先选标签，再从该标签对应的样本中选一个

**返回结构**：

```python
_LabelBlock(
    index=0,
    items=(("A", 0.1), ("B", 0.2), ("C", 0.2)),
    label_to_ids={
        0.1: ("A",),
        0.2: ("B", "C"),
    },
    n_records=3,
)
```

#### `_sample_from_block_pairs` - 跨块抽样

**功能**：从不同块之间抽pair。

**实现步骤**：

1. **枚举所有块组合**

```python
for left, right in itertools.combinations(blocks, 2):
```

- 产生所有不重复的两块组合
- 避免重复组合

2. **计算合法pair数量**

```python
valid_count = _cross_block_candidate_count(left, right)
```

- 合法pair：两个样本标签不同
- 标签相同的pair对排序训练没有意义

3. **设置块对权重**

```python
valid_count * abs(right.index - left.index)
```

- 权重 = 合法pair数量 × 块之间的距离
- 块距离越远，通常标签差距越大
- 更倾向于抽取合法组合数量多、标签距离较远的块对

4. **创建独立随机数生成器**

```python
rng = random.Random(seed)
```

- 不污染Python全局随机状态
- seed相同则抽样结果可复现

5. **调用统一抽样循环**

```python
_sample_until_target(
    ...
    draw=lambda: _draw_between_blocks(candidates, rng, use_label_buckets),
)
```

- 将"如何抽一个跨块pair"包装成draw函数
- 交给通用循环_sample_until_target处理

#### `_sample_within_blocks` - 块内抽样

**功能**：在同一个块内抽pair。

**实现步骤**：

1. **找出存在合法pair的块**

```python
candidates = [
    (block, _within_block_candidate_count(block))
    for block in blocks
    if _within_block_candidate_count(block) > 0
]
```

- 每个块的权重是该块内部标签不同的pair数量
- 如果块中所有标签都相同，则不参与抽样

2. **调用统一抽样控制器**

```python
draw=lambda: _draw_within_block(candidates, rng, use_label_buckets)
```

- _sample_from_block_pairs提供"跨块抽一个"的方法
- _sample_within_blocks提供"块内抽一个"的方法
- _sample_until_target负责重复抽样和去重

####  `_sample_until_target` - 通用抽样控制器

**功能**：核心循环，负责不断调用draw，直到达到目标数量。

**实现步骤**：

1. **记录起始行数**

```python
start_count = len(rows)
```

- 目标是新增target行，而不是让总长度变成target

2. **设置最大尝试次数**

```python
max_attempts = max(1000, target * 100)
```

- 防止死循环
- 至少1000次，或每个目标pair允许100次尝试

3. **调用具体抽样逻辑**

```python
drawn = draw()
```

- draw可能是跨块或块内抽样

4. **排除相同标签**

```python
if label_a == label_b:
    continue
```

- 相同标签无法构成明确高低关系的排序样本

5. **统一pair方向**

```python
record_id_i, label_i, record_id_j, label_j = _canonical_pair(...)
```

- 将(A,B)和(B,A)统一成同一种标准顺序
- 避免同一个无序pair被当成两个不同pair

6. **去重**

```python
key = (record_id_i, record_id_j)
if key in seen:
    continue
```

- seen保存已经加入过的pair

7. **生成监督标签**

```python
y_ij = 1.0 if label_i > label_j else 0.0
```

- 如果i的rank_label比j大，则y_ij = 1
- 否则y_ij = 0

8. **构造pair数据**

```python
rows.append(
    _pair_row(
        group_id,
        record_id_i,
        record_id_j,
        label_i,
        label_j,
        y_ij,
    )
)
```

#### `_draw_between_blocks` - 跨块pair抽样

**功能**：真正完成选一对块，然后从左右块各选一个样本。

**实现步骤**：

1. **按权重选块对**

```python
left, right = _weighted_choice(candidates, rng)
```

- 权重大的块对更容易被选到

2. **普通模式**

```python
if not use_label_buckets:
    record_id_left, label_left = rng.choice(left.items)
    record_id_right, label_right = rng.choice(right.items)
```

- 直接随机选样本
- 可能选到相同标签，之后会被过滤

3. **标签桶模式**

```python
label_left = _weighted_label_excluding(left, right, rng)
label_right = _weighted_choice(
    [
        (label, len(record_ids))
        for label, record_ids in right.label_to_ids.items()
        if label != label_left
    ],
    rng,
)
```

- 先选标签，再选样本
- 保证标签不同，更加高效

#### `_draw_within_block` - 块内pair抽样

**功能**：与跨块抽样相似，但两个样本都来自同一个块。

**实现模式**：

- 普通模式：直接独立抽两次
- 标签桶模式：先选标签，再选样本

#### `_weighted_label_excluding` - 加权标签选择

**功能**：按可形成的合法pair数选择标签。

**权重公式**：

```
weight = len(record_ids) * (other.n_records - other_same)
```

**实现原理**：

- 不是简单随机选择标签
- 按照这个标签能够形成多少个合法pair来加权
- 让每个具体合法pair在整体上更接近均匀采样

####  `_cross_block_candidate_count` - 跨块合法pair计算

**功能**：计算跨块合法pair数。

**计算公式**：

```
合法pair数 = |L|×|R| - ∑(n_L,l × n_R,l)
```

**实现**：

```python
return left.n_records * right.n_records - same_label
```

#### `_within_block_candidate_count` - 块内合法pair计算

**功能**：计算块内合法pair数。

**计算公式**：

```
合法pair = C(n,2) - ∑C(n_l,2)
```

**实现**：

```python
total = block.n_records * (block.n_records - 1) // 2
same_label = sum(
    len(record_ids) * (len(record_ids) - 1) // 2
    for record_ids in block.label_to_ids.values()
)
```

#### 整套实现的五层架构

**第1层**：`_build_label_blocks`

- 按标签排序并切块

**第2层**：`candidate_count`

- 计算每个块或块对中有多少合法pair

**第3层**：`_draw_...`

- 按照权重抽取一个候选pair

**第4层**：`_sample_until_target`

- 过滤、去重、统一方向并加入结果

**第5层**：`_sample_from_block_pairs`/`_sample_within_blocks`

- 组织跨块采样和块内采样