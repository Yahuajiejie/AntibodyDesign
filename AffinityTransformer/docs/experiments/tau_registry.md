# 各数据来源的 τ（噪声下限）取值表

日期：2026-06-22
状态：用于 `noise_aware_multiscale` 的 tau 解析（`affinity_transformer/dataset/pair_sampling/tau_registry.py`）

## 1. 为什么不能用一个全局 τ

`AbRank/dataset/records.parquet` 实际上是把 8 个完全不同的原始数据来源（不同实验方法、不同实验室、不同年代）合并成了一张表。这 8 个来源的测量噪声不是一回事——拿同一个 τ 套所有数据，要么对精度高的来源（比如下面的 RBD-escape）太保守、丢掉本来可以用的信号，要么对精度低的来源（比如下面的 CATNAP）太宽松、继续把噪声当成信号训练。所以 τ 必须按数据来源分别定。

## 2. 数据来源分布

原始 CSV（`data/binding/AbRank_dataset.csv.zip`）的 `Source` 列：

| Source | 原始行数 | 说明 |
| - | - | - |
| RBD-escape | 192,559 | 仅 20,500 行带 Kd（其余是 escape fraction，没进 `records.parquet`） |
| CATNAP | 74,540 | HIV 中和试验数据库（Los Alamos） |
| AlphaSeq | 71,834 | A-Alpha Bio 高通量酵母展示平台 |
| AbCoV | 1,392 | |
| SKEMPIv2 | 935 | 突变对蛋白-蛋白结合能影响的人工整理库 |
| AbSci | 758 | |
| SabDab | 249 | 主要是结构数据库，少量带 Kd |
| OVA-binders | 89 | |

进了 `records.parquet`（134,958 条可训练记录）以后，原始 `Source` 列没有保留下来，但可以通过 `antigen_key` 几乎完全还原对应关系（用记录数核对，误差 <0.5%）：

| `antigen_key` 匹配规则 | 记录数 | 对应来源 | group 数 |
| - | - | - | - |
| `antigen_key == "SARS_CoV_2"` | 73,411 | AlphaSeq（99.6%+ 纯，混入极少量 AbCoV） | 2 |
| `antigen_key` 形如 `HIV_*` | 73,508 | CATNAP | 2,390 |
| `antigen_key` 形如 `SARS_CoV_2_<点突变>`，或 `SARS_CoV`/`WIV1_CoV`/`SHC014_CoV`/`MERS_CoV` | 20,500+77 | RBD-escape 的 Kd 子集 / Bloom 实验室同一方法测的近缘 sarbecovirus | 2,033+2 |
| `antigen_key == "HER2"` | 758 | AbSci | 1 |
| `antigen_key` 形如 `AgSKEMPI_*` | ~935 | SKEMPIv2 | 多个小组 |
| 其余（OVA 等） | ~89 | OVA-binders / SabDab 残余 | 少量小组 |

这张表覆盖了 99%+ 的可训练记录，按这 5 类分别定 τ。

## 3. τ 取值表

| 来源 | τ（log10 单位） | 判断依据 | 置信度 |
| - | - | - | - |
| **AlphaSeq**（`SARS_CoV_2` 主组，73,411 条，原本卡 loss 的那个超大组） | **0.3** | Engelhart et al., *Sci Data* 9:653 (2022)。论文报告三次技术重复的 Pearson r = 0.66–0.93；k=1 突变体亲和力 IQR = 1.69 log10(nM)。换算出的重复测量标准差约 0.47–1.03 log10 单位，0.3 比这个区间的下沿还保守一些。 | 高（已在上一轮实验报告中用真实数据验证过） |
| **RBD-escape Kd 子集 / Bloom 实验室 Titeseq**（点突变 RBD 抗原小组，约 20,577 条，组很小） | **0.15** | Starr et al., *Cell* 182:1295 (2020) 提出的酵母展示 Titeseq 方法；后续 Omicron 跟进论文（Taylor et al., *PLoS Pathog* 2022）方法部分确认了同一套流程（每个突变体最终值由约 40+ 条 barcode 在重复实验间取平均）。搜索到的引用资料显示同一方法下独立文库的重复测量决定系数 R² > 0.99。**没有查到直接以 log10(Kd) 单位报告的标准差**，0.15 是用 R²>0.99 倒推（噪声方差 = 总方差 ×(1-R²)，假设该突变扫描的亲和力分布总标准差量级在 1–1.5 log10 单位）估出来的，比 AlphaSeq 更精确但换算环节弱一些。 | 中（方法论上确认是更精确的单实验室方法，但没有原始 SD 数字，是换算估计） |
| **CATNAP / HIV 中和试验（IC50）**（`HIV_*` 系列，73,508 条，2,390 个小组） | **0.5** | CATNAP 本身是跨实验室、跨年代汇总的文献数据库，没有统一测量协议。能查到的最接近的定量依据，是同一标准化中和试验体系（A3R5/TZM-bl 联盟验证，Sarzotti-Kelsoe et al. *J Immunol Methods* 2014；A3R5 验证论文 PMC4138262）：repeatability/intermediate precision/reproducibility 三个层级都用"**3 倍以内**"作为可接受标准——换算成 log10 是 0.48。这是单一标准化协议、同一时期内的最优情况；CATNAP 汇总的是几十年、多个实验室、多套协议的数据，真实噪声大概率比这个数更大，所以 0.5 只是在这个下界上稍微加了一点保守余量，不是足够保守的上界估计。 | 中低（有定量依据，但依据来自"理想情况"的协议内验证，CATNAP 实际异质性可能更大） |
| **SKEMPIv2**（`AgSKEMPI_*` 系列，~935 条，多个小组） | **0.35** | SKEMPI 2.0 原论文（Jankauskaitė et al., *Bioinformatics* 35:462, 2019）指出同一突变在不同来源间的重复测量会给出不同的 ΔΔG，量级在 ±0.5 kcal/mol。按 ΔΔG = −RT·ln(Kd_mut/Kd_wt) 换算，RT·ln10 ≈ 1.365 kcal/mol（298K），0.5/1.365 ≈ 0.37 log10 单位，取 0.35。 | 中（有数量级依据，但论文给的是"大多数 ΔΔG 落在 ±0.5 kcal/mol"这个分布描述，不是专门给的重复测量误差） |
| **其余未单独查证的来源**（AbSci/HER2、OVA-binders、SabDab 残余，合计 < 1% 的记录） | **0.2**（默认值，`noise_aware_default_tau`） | 没有专门查这几个来源的重现性文献——记录数太少（合计约 850 条，占比 <1%），调研投入产出比低。0.2 是介于"高精度 Titeseq(0.15)"和"中等精度多来源汇总(0.35-0.5)"之间的折中默认值，主要起"别让代码因为找不到 τ 而崩，也别太离谱"的作用，不代表真的查过这些来源的测量误差。 | 低（明确未查证，纯保守占位） |

## 4. 代码里怎么用

`tau_registry.py` 按 `antigen_key` 正则匹配上表规则；任何不匹配的组都落到 `default_tau`（默认 0.2，可在 YAML 里覆盖）。匹配规则、τ 值、依据原文都写在该文件里，不在 YAML 里配置——因为这是查文献查出来的事实，不是每次训练想调就调的超参数。

## 5. 局限

1. RBD-escape 和 CATNAP 这两条都没查到"直接以 log10(Kd) 为单位报告的重复测量标准差"这种第一手数字，是从相关但不完全等价的指标（R²、3 倍边界、ΔΔG 分布宽度）换算出来的，置信度比 AlphaSeq 那条低，后续如果能查到这两个来源更直接的复现性数据，应该回来更新这两个值。
2. `antigen_key` 到原始 `Source` 的映射是用记录数核对出来的（数字基本能对上，误差 <0.5%），不是逐条记录验证的——`SARS_CoV_2` 主组里仍可能混入个别非 AlphaSeq 记录（早先的分析认为比例 <0.4%），但不影响 τ=0.3 这个结论的方向。
3. 默认值 0.2 覆盖的几个小来源完全没有查证，只是为了不让代码遇到陌生来源时崩溃或者拍脑袋用 0——如果这几个来源（合计约850条）以后变得重要，需要单独补研究。
