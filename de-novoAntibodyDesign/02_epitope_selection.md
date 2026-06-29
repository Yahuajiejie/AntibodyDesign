# 02 Epitope Selection

目标：针对高度易突变蛋白，优先选择保守、暴露、结构稳定、功能约束强的表位，降低突变逃逸风险。

## 表位选择原则

优先级从高到低：

1. 高保守：跨同源序列或自然变异数据中变异熵低。
2. 高暴露：在多个构象中 solvent accessible surface area 较高。
3. 功能约束强：突变后可能影响蛋白功能、折叠、配体结合或构象转换。
4. 构象稳定：不是完全无序、强柔性或结构缺失区域。
5. 可被抗体接近：不被糖基化、膜、复合物伙伴或 steric occlusion 遮挡。
6. 低交叉反应：与人源蛋白或重要非目标蛋白相似度低。

## 评分维度

建议每个候选表位打 0 到 5 分：

- `conservation_score`：越保守越高。
- `surface_exposure_score`：越暴露越高。
- `functional_constraint_score`：越功能关键越高。
- `structure_confidence_score`：实验结构覆盖或 AlphaFold pLDDT 越可靠越高。
- `mutation_escape_risk_score`：越不容易逃逸越高。
- `accessibility_score`：越不被 glycan、膜或复合物遮挡越高。
- `specificity_score`：越不容易交叉反应越高。

加权总分建议：

```text
total = 0.25 * conservation
      + 0.20 * exposure
      + 0.20 * functional_constraint
      + 0.10 * structure_confidence
      + 0.15 * escape_resistance
      + 0.05 * accessibility
      + 0.05 * specificity
```

## 快速流程

1. 在结构上标注已知功能残基、变异热点、糖基化/PTM、低 pLDDT 区域。
2. 用多序列比对计算每个位点保守性。
3. 在多个结构或预测构象上计算表面暴露。
4. 将连续 8 到 25 aa 的高分区域合并成候选表位。
5. 每个候选表位检查三维邻近残基，不只看线性序列。
6. 输出 2 到 4 个主表位，每个表位记录设计约束和不能接触的区域。

## 对易突变靶标的特别策略

- 不要只追求最凸出的 loop，因为这些区域常常也是高变区。
- 优先找“功能不能乱变”的保守沟槽、受体/配体附近、构象转换核心或寡聚界面边缘。
- 如果靶蛋白存在多个亚型或变体，首轮就纳入变体结构面板，避免只对单一序列过拟合。
- 每个表位至少设计两类 binder：一类直接压住热点，一类从侧面锁住保守结构元素。

## 输出

把结果填入：

```text
tables/epitope_scoring_template.csv
```

每个表位需要包括：

- residue range
- 关键三维邻近残基
- 使用的结构/PDB chain
- 选择理由
- 设计约束
- 备用表位

