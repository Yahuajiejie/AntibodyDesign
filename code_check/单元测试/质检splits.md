# splits.py
```python
def build_group_kfolds(

    records: pd.DataFrame,
    n_splits: int,
    seed: int,
    
) -> list[GroupFold]:
```
K折交叉验证的目标是保证 每个折尽可能拥有一样多的样本数。由于我们排序问题要求组是原子的、不可分割的，因此K折划分问题在这种背景下等价于**“多机调度”问题**

而当前函数实现的，其实是多机调度问题的**LPT（Longest Processing Time First，最长处理时间优先）算法**

算法步骤如下：

- 将组按照大小降序排列。如果大小相同，则按照 tie_order 排序。将大组优先分配，有助于后续更好地平衡各折的数据量。
- 初始化 n_splits 个空的验证集（fold_groups）和对应的计数器（fold_sizes）。
- 遍历排好序的组，每次都将当前组分配给当前总记录数最少的那个折（如果数量相同，则优先分配给索引小的折）。
- 这种贪心策略能够最大程度地保证各个验证折（Validation Fold）之间的数据规模是相对均衡的。 

算法做的质检如下：

（1）在开始划分前，函数进行了多项防御性检查：

- 列校验：确保 DataFrame 中包含必需的 record_id、group_id 和 dataset_id。
- 参数校验：确保折数 n_splits >= 2，且数据不为空。
- 数据完整性校验：检查 record_id 和 group_id 是否存在空值（null），以及 record_id 是否存在重复。

（2）构建完成后，函数又做了如下检查

- 对于每一个折，将其分配到的组作为验证集（valid_groups），剩下的所有组作为训练集（train_groups = all_groups - valid_groups）。
- 根据组 ID 从原始 DataFrame 中提取对应的训练集和验证集数据。
- 防泄露双重保险：在返回结果前，函数会进行严格的断言检查：

- 检查训练集或验证集是否为空。
- 计算交集：检查训练集和验证集的 group_id 是否存在交集（set(...) & set(...)）。如果存在交集，说明同组数据跨越了训练集和验证集，函数会直接抛出 ValueError: group leakage detected 异常。

  
其他函数
1. 入口：build_splits (主指挥官)
2. 策略选择：

- 策略A（按记录）：_split_by_record -> _partition_units
- 策略B（按组）：_split_by_group -> _partition_weighted_units -> _split_holdout_by_weight

4. 善后工作（无论哪种策略）：

- _build_summary: 生成统计报表。
- _build_leakage_report: 生成防泄露报表。

6. 通用工具：

- _rows_for_values: 哪里需要数据切片哪里就有它。

-  _trainable_records: 划分前先清理无效数据。