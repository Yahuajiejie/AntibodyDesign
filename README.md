# 第四届 Bio-OS AI 开源大赛 · 抗体设计赛道

## 项目概述

本仓库为参赛工作目录，赛题为：**利用 AI 模型针对 Nipah 病毒 G 蛋白从头设计高亲和力中和抗体**。

初赛任务是对给定抗原-抗体对进行亲和力排序预测（Spearman 相关系数评分），复赛任务是提交最多 6 条真实可验证的新抗体序列。

---

## 目录结构

```
antibody/
├── FLAb/                        # FLAb benchmark（来自 Graylab/FLAb）
│   ├── data/binding/            # 80+ 抗体亲和力数据集
│   ├── models/                  # 官方打分脚本（零样本）
│   ├── score/                   # 官方已有结果（antiberty/esmif/iglm/mpnn/pyrosetta）
│   ├── run_scoring_batch.py     # 【新增】批量打分脚本（见下）
│   └── requirements_esm2.txt   # 【新增】ESM2 环境依赖
├── docs/                        # 赛题解读与技术文档
└── 第四届Bio-OS开源大赛数据/      # 官方提供的比赛数据集
    ├── 初赛-序列数据/            # 22 组亲和力数据集 + proteinbase Nipah 数据
    ├── 初赛-结构数据/            # SAbDab 结构数据库（2026-03-05 版本）
    └── 初赛-纳米抗体数据/        # INDI2 + ANDD 纳米抗体数据集
```

---

## 已完成工作

### 1. 数据整理与分析

- 梳理了 FLAb benchmark 的完整结构：80 个 binding 数据集，样本量从 5 到 190 万不等
- 确认官方仓库缺失 `score_ft/`、`envs/`、`embedding_help.py`、`embedding_train.py` 等文件，`ft_scoring_*` 系列脚本无法直接运行
- 从 git 历史（commit `81756d90^`）恢复了 `envs/` 下的 conda 环境配置文件
- 识别出 `初赛-序列数据/22/proteinbase_all_data_28_01_2026.csv`：含 5253 条抗体数据，其中 3754 条针对 Nipah G 蛋白，242 条有 Strong binding 标注，包含 SPR 实验 Kd 值——**这是复赛靶标的直接参考数据**

### 2. 基线分析

统计了 FLAb 官方已有的零样本模型在 binding 任务上的表现：

| 模型 | 平均 Spearman | 中位 Spearman |
|------|:---:|:---:|
| IgLM | 0.174 | 0.233 |
| AntiBERTy | 0.117 | 0.097 |
| ESM-IF | 0.007 | 0.083 |
| ProteinMPNN | -0.011 | 0.013 |
| PyRosetta | -0.150 | -0.077 |

现有最强基线平均 Spearman 仅 0.17，竞争门槛较低。

### 3. ESM2 基线验证

在 `hie2023efficient_CoV2_S309_Kd`（n=20）上运行 ESM2-650M 零样本打分：

```
Spearman = 0.514，p = 0.020
```

与已有模型对比：AntiBERTy（0.558）、IgLM（0.521）、**ESM2-650M（0.514）**，三者相近，远优于结构类模型。

### 4. 新增工具脚本

`run_scoring_batch.py`：统一批量打分脚本，替代官方零散的 `scoring_*.py`。

- 支持 ESM2 全系列（8M / 35M / 150M / 650M / 3B / 15B）
- 自动处理 `.csv` 和 `.csv.zip` 格式
- 每个数据集输出逐序列 perplexity 分数（`_perseq.csv`）+ 全局 Spearman 汇总（`summary.csv`）
- 结果写入 `score_batch/`，不覆盖官方 `score/` 目录
- 模型懒加载缓存，避免官方脚本中每条序列重复加载模型的低效问题

```bash
# 安装依赖
pip install -r requirements_esm2.txt

# 快速测试（前50条）
python run_scoring_batch.py --model esm2_650M --max_rows 50

# 全量运行
nohup python run_scoring_batch.py --model esm2_650M esm2_3B > log_scoring.txt 2>&1 &
```

---

### 5. 有监督训练框架

`train_affinity_model.py`：ESM2 + MLP + Pairwise Ranking Loss 完整训练脚本。

核心设计：
- **Backbone**：ESM2-650M（冻结），提取 per-protein mean pooling embedding（1280维）
- **Head**：MLP（1280→256→1），GELU 激活 + Dropout(0.2)
- **Loss**：Pairwise Hinge Ranking Loss（直接优化排序，与 Spearman 评分对齐）
- **训练范式**：Per-benchmark fine-tuning（80/10/10 划分），每个数据集训练独立 head
- **缓存机制**：ESM2 embedding 缓存到磁盘（只算一次，后续训练直接读取）

```bash
# 步骤1：提取并缓存所有 embedding（只需跑一次，GPU 密集）
python train_affinity_model.py --mode embed

# 步骤2：训练 + 评估（从缓存读 embedding，CPU/GPU 均可）
python train_affinity_model.py --mode train

# 一键全流程
nohup python train_affinity_model.py --mode all > log_train.txt 2>&1 &
```

---

## 下一步计划

- [ ] 在所有 binding 数据集上完成 ESM2-650M 零样本基线（`run_scoring_batch.py`）
- [ ] 完成有监督训练并对比 zero-shot 基线（预期提升显著）
- [ ] 分析 proteinbase Nipah 数据，提取复赛候选序列特征
- [ ] 考虑用 ISM（抗体专用语言模型）替换 ESM2 backbone
