# 04 In Silico Screening

目标：用统一标准筛选两条路线输出的候选，选出值得合成和体外验证的一小批设计。

## 一级过滤：结构可信度

保留：

- binder 自身 pLDDT 或等价置信度高。
- target-binder interface PAE 低。
- 复合物 ipTM/pTM 或等价指标合理。
- 预测界面位置与目标表位一致。
- 不依赖明显不可信的低置信度 loop。

剔除：

- binder 自身不折叠。
- 复合物靠长柔性尾巴或单点偶然接触。
- 明显 clash 或穿模。
- 设计表位偏离目标表位。
- 只在生成模型中好看、独立模型不复现的候选。

## 二级过滤：界面质量

建议记录：

- interface contact count
- buried solvent accessible surface area
- shape complementarity
- hydrogen bond / salt bridge pattern
- hydrophobic patch 是否合理
- Rosetta/PyRosetta interface energy 或 ddG
- 多构象、多变体结构中的结合保持情况

## 三级过滤：抗体可开发性

对 antibody/nanobody/scFv 候选检查：

- CDR 长度是否合理。
- framework 是否可编号，ANARCI/IMGT 是否识别。
- 是否过多疏水暴露。
- 是否存在明显聚集风险。
- 是否存在异常 cysteine、N-linked glycosylation motif、deamidation/isomerization hotspot。
- 是否接近人源抗体序列空间，或是否需要 humanization。
- 是否有多克隆/多特异风险。

## 变体鲁棒性过滤

由于靶蛋白高度易突变，必须做：

1. 选择代表性变体结构或建模变体结构。
2. 对每个候选重新预测或快速 docking/refold。
3. 统计表位残基突变后接触是否保留。
4. 优先保留跨变体结合姿态稳定的候选。

## 进入湿实验的推荐组合

首轮不要只选 top score。建议组合：

- 每个主表位 3 到 8 个候选。
- 每条技术路线至少保留若干候选。
- 包含不同 CDR 长度或 binder topology。
- 包含 1 到 2 个高风险高回报候选。
- 包含 1 到 2 个稳健保守候选。

## Go / No-Go

进入基因合成的最低门槛：

- 目标表位正确。
- 独立结构预测复现复合物。
- binder 自身可折叠。
- 序列没有明显不可开发性红旗。
- 对主要变体至少保持合理接触。

