# Rapid De Novo Antibody Design Plan

目标：针对一个高度易突变的靶蛋白，先在计算端选择相对稳健的保守/暴露/功能约束表位，再并行执行两条从头抗体或纳米抗体设计路线，最后用体外结合实验验证候选分子的结合潜力。

> 当前假设：靶标不涉及实验室风险等级生物，不做病原体培养、感染、增强功能或动物挑战实验。湿实验部分限定为重组蛋白、合成基因、体外结合与常规生化表征；具体实验条件按所在实验室 SOP 或 CRO 标准流程执行。

## 目录结构

- `00_target_intake.md`：填写靶蛋白、物种/来源、UniProt、PDB、AlphaFold DB、突变数据来源。
- `01_data_acquisition.md`：从 PDB、UniProt、AlphaFold DB、变异数据库和文献收集数据。
- `02_epitope_selection.md`：针对易突变蛋白筛选保守、暴露、功能约束表位。
- `03_dual_path_design.md`：两条设计路线并行：RFdiffusion/RFantibody 路线与 AF2-backprop/BindCraft/Germinal/mBER/BoltzGen 路线。
- `04_in_silico_screening.md`：结构复核、界面评分、去冗余、抗体可开发性过滤。
- `05_wet_lab_validation.md`：基因合成、表达纯化、结合验证、变体面板和优先级决策。
- `tables/epitope_scoring_template.csv`：候选表位评分表。
- `tables/candidate_scoring_table.csv`：候选抗体/binder 评分表。
- `scripts/download_target_data.sh`：公开数据库下载模板。

## 赶实验版时间线

### Day 0：定靶与拉数据

1. 填写 `00_target_intake.md`。
2. 用 `scripts/download_target_data.sh` 下载 UniProt FASTA、PDB/mmCIF、AlphaFold DB 结构。
3. 收集突变频率、多序列同源、已知结构构象、已知结合位点和功能位点。

### Day 1：选表位

1. 结构上计算 solvent accessibility、二级结构、柔性区、糖基化/PTM、跨膜/低复杂度区域。
2. 序列上计算保守性、变异熵、功能约束、同源蛋白交叉反应风险。
3. 选择 2 到 4 个表位窗口，每个表位保留一个主表位和一个备用表位。

### Day 2-4：两条设计路线并行

1. 路线 A：RFdiffusion/RFantibody 生成抗体或 nanobody backbone，ProteinMPNN/AbMPNN 设计序列。
2. 路线 B：BindCraft/Germinal/mBER/ColabDesign/BoltzGen 通过 AF2/Boltz 类预测模型反向优化或统一生成。
3. 每条路线至少保留结构多样的候选，不只保留同一构象附近的重复设计。

### Day 5-7：计算筛选

1. 用独立结构预测模型复核，不只相信生成阶段的模型。
2. 按界面置信度、接触、埋藏面积、clash、形状互补、序列自然性、聚集风险排序。
3. 每个表位选 5 到 20 个候选进入基因合成，急的话先选 top 8 到 24 个。

### Day 8+：湿实验验证

1. 合成候选基因，表达纯化抗体片段或 nanobody。
2. 体外验证结合：ELISA 或 BLI/SPR。
3. 用变体靶蛋白面板测保守表位鲁棒性。
4. 命中后做亲和力成熟或第二轮计算优化。

## 两条技术路线的核心差异

路线 A 更像“先生成骨架，再设计序列”：

```text
target structure + epitope constraints
        -> RFdiffusion/RFantibody backbone generation
        -> ProteinMPNN/AbMPNN sequence design
        -> refold/filter
```

路线 B 更像“冻结结构预测模型，把它当可微分打分器，直接优化 binder 输入”：

```text
binder sequence logits or generator output
        -> frozen AF2/AF-Multimer/Boltz-like predictor
        -> design loss from predicted complex
        -> gradient back to binder variables only
```

两条路线的输出应混合进入统一筛选表，而不是分开用不同标准挑选。

