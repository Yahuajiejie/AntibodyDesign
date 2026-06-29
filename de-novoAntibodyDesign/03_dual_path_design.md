# 03 Dual Path Design

目标：同时跑两条技术路径，减少单一路线偏差，并让结构多样性覆盖多个候选表位。

## 输入

每个设计任务至少需要：

- 靶蛋白结构，最好是清理后的 PDB/mmCIF。
- 表位残基列表。
- 可接触热点残基。
- 禁止接触或低优先级区域。
- 期望 binder 类型：VHH、scFv、Fab、IgG CDR、mini-binder。
- 候选数量目标。

## 路线 A：RFdiffusion/RFantibody + ProteinMPNN/AbMPNN

本路线是 backbone-first：

```text
target + epitope constraints
        -> RFdiffusion/RFantibody backbone generation
        -> ProteinMPNN/AbMPNN sequence design
        -> independent refold/filter
```

适合：

- 需要探索全新结合构象。
- 想生成 nanobody 或 antibody-like backbone。
- 表位比较清楚，有 hotspot 或 docking 约束。

关键设计点：

1. 对每个表位设置 hotspot，不要只给一个残基。
2. 控制 binder 长度和形状，避免生成过大的无关界面。
3. 生成 backbone 后用 ProteinMPNN 或 AbMPNN 做多序列设计。
4. 每个 backbone 设计多个序列，不要把 backbone 和 sequence 一对一绑定。
5. 用独立预测模型复核，而不是只看 RFdiffusion 自身输出。

首轮建议输出：

- 每个表位 100 到 1000 个 backbone。
- 每个 backbone 4 到 16 个 sequence design。
- 计算资源紧张时，每个表位先做 50 到 200 个快速探索。

## 路线 B：AF2-backprop / BindCraft / Germinal / mBER / BoltzGen

本路线更像 predictor-guided optimization：

```text
binder sequence logits or generator output
        -> frozen AF2/AF-Multimer/Boltz-like predictor
        -> design loss from predicted complex
        -> gradient back to binder variables only
```

适合：

- 想快速针对一个表位 hallucinate binder。
- 想利用 AF2/AF-Multimer 的复合物预测信号。
- 想结合抗体先验，如 VHH/scFv framework、AbMPNN、抗体语言模型或模板。

可选工具定位：

- BindCraft：通用 de novo protein binder，不是抗体专用，但适合理解和快速探索。
- Germinal：epitope-targeted de novo antibody pipeline，适合 nanobody/scFv。
- mBER：抗体 binder design，强调结构模板和 sequence conditioning。
- BoltzGen：统一 binder design 框架，可覆盖 protein、peptide、nanobody/antibody CDR 等协议。

关键设计点：

1. AF2/AF-Multimer 或 Boltz 类模型冻结，不更新预测模型参数。
2. 被优化的是 binder sequence logits、binder 坐标变量或小型 generator 输出。
3. loss 需要同时约束界面、binder 自身折叠和表位接触。
4. 最后必须离散化序列，并用独立模型复核，防止 predictor hacking。

首轮建议输出：

- 每个表位 50 到 300 个优化轨迹。
- 每条轨迹保留不同 checkpoint 的候选。
- 对同一表位用不同初始长度、随机种子和接触约束产生多样性。

## 统一候选命名

建议：

```text
TARGET_EPITOPE_ROUTE_INDEX
```

例子：

```text
TGT1_EPI2_RFA_0001
TGT1_EPI2_BC_0042
TGT1_EPI3_MBER_0018
```

## 输出进入同一评分表

所有路线候选统一填入：

```text
tables/candidate_scoring_table.csv
```

不要让 RFdiffusion 路线和 backprop 路线使用不同筛选门槛，否则会引入人为偏差。

