# matrics
```python
def compute_group_spearman(predictions: pd.DataFrame) -> pd.DataFrame
```
计算的是模型输出结果中，每个组的Pearson相关系数：
- group["rank_label"].rank()计算真实标签的排名。
- group["score"].rank()计算模型预测分数的排名。
- corr(...)：计算这两个排名序列的 Pearson 相关系数。
```python
def _summarize_subset(subset: pd.DataFrame) -> dict[str, float | int]:

    """Compute the macro/weighted-average summary for one subset of groups.

    Args:
        subset: Rows of a `compute_group_spearman` result (any subset, e.g.
            all rows, or just those with `label_kind == "binary"`).
    Returns:
        Dict with keys `n_groups, n_valid_groups, n_skipped_groups,
        macro_spearman, weighted_spearman` (see `summarize_group_spearman`).
    """
```
计算的是spearman的：
- 宏观平均（macro_spearman）：对所有有效组（n_valid_groups）的 Spearman 系数求简单算术平均。每个组无论大小，权重相同。这可以防止少数包含大量样本的组主导整体指标。
- 加权平均（weighted_spearman）：以每个组内的样本数（n_records）为权重进行加权平均。这反映了模型在每一个具体样本上的平均排序表现。
```python
def summarize_group_spearman(group_metrics: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    """Aggregate per-group Spearman correlations into macro/weighted averages."""
```
统计模型输出的质量，汇总为一张表。为了全面评估模型在不同数据子集上的表现，汇总结果还包含以下统计量：
- 按标签类型拆分（label_kind）：汇总指标不仅会输出一个全局的 "overall" 结果，还会严格按照数据的标签类型（如 binary 二分类标签、experimental 实验标签等）分别计算上述的宏观和加权平均指标。
- 组数量统计：
- n_groups：该子集下的总组数。
- n_valid_groups：成功计算出 Spearman 系数的有效组数。
- n_skipped_groups：因不满足条件而被跳过的组数（即 n_groups - n_valid_groups）。