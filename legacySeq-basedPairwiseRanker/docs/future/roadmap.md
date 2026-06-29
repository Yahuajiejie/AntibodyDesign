# AffinityTransformer 整改路线图

本项目当前处于大集群联调阶段。现有配置的首要任务是验证数据、缓存、训练和资源链路能否跑通，并不承担正式模型比较。因此，当前最高优先级不是整理所有实验 YAML，而是修复那些会让代码“表面跑通，实际执行了另一件事”的 Python 硬伤。

后续整改遵守三条原则：

1. 先保证执行语义真实，再谈数据协议和模型优劣。
2. 集群冒烟结果只证明链路可运行，不自动升格为科学结论。
3. 每项修复必须同时提供失败用例、代码修复和自动化验收。

## 1. 当前阶段如何解释

| 产物 | 当前用途 | 不能说明什么 |
| --- | --- | --- |
| v065 训练配置 | 检查环境、缓存、DataLoader、模型和 SLURM 链路 | 不用于得出 sampler、深度或标签来源的优劣 |
| group holdout 结果 | 工程基线和未见同质 group 诊断 | 不等于 unseen-antigen 或 dual cold-start |
| 每轮 valid/test 输出 | 联调评估代码和文件产物 | 不是已经隔离的最终测试 |
| 最后 epoch checkpoint | 检查保存和恢复链路 | 不代表验证集最优模型 |

当前可以继续跑小规模任务和资源压力测试。输出统一标记为 `smoke`、`integration` 或 `diagnostic`，不进入正式模型排行。

## 2. 第一阶段：修复 Python 执行硬伤

这一阶段立即执行。它不要求主数据协议已经定稿，也不要求当前 YAML 已经具备正式实验的可比性。

### P0-1：修正 online encoder mode

#### 现状

`frozen_online` 加载 ESM2 后没有关闭梯度，`Trainer` 又把全部参数交给 optimizer，实际会全量微调 encoder。`lora_online` 没有注入 LoRA，也会静默变成全量微调。

#### 修改

- `frozen_online` 的 encoder 参数全部 `requires_grad=False`；
- 外层 ranker 进入 `train()` 时，冻结 encoder 仍保持 `eval()`；
- optimizer 只接收可训练参数；
- LoRA 真正实现前，`lora_online` 在构建模型时明确报错；
- 如果未来需要全量微调，新增名义明确的 `full_finetune_online`。

#### 验收

- 训练一步后，冻结 encoder 的参数和 BatchNorm/Dropout 状态不变；
- optimizer 中不存在冻结参数；
- `lora_online` 不会触发模型下载或静默 fallback。

### P0-2：在模型构建前设置 seed

#### 现状

seed 在 `Trainer.fit()` 才设置，模型和打分头已经初始化。同一配置重复运行时，初始权重不一定相同；K 折的后一折还会继承前一折消耗后的 RNG 状态。

#### 修改

- `train.py` 读取配置后立即设置 seed，然后才进入 runner；
- `Trainer.fit()` 不再负责首次播种；
- K 折每折进入 runner 前使用稳定的派生 seed。

#### 验收

- 同一配置的两次独立运行获得相同的初始参数；
- 改变 seed 后初始参数或采样结果发生变化；
- K 折的某一折不受其他折是否先执行的影响。

### P0-3：打通 frozen-cached 训练到新序列推理

#### 现状

训练主线的 `frozen_cached` 构建 `EmbeddingAffinityRanker`，而 `load_predictor` 和 `load_model` 仍调用旧的 `build_model_and_tokenizers`，构建 `AffinityRanker`。两种模型的参数名称和结构不同，缓存主线训练出来的 checkpoint 无法按现有用户入口正常使用。

#### 修改

- 推理时按 checkpoint/config 判断模型类型；
- `frozen_cached` 模型对新序列在线生成与训练缓存同规格的 token embedding；
- 使用 `EmbeddingAffinityRanker` 和 `EmbeddingBatch` 打分，不将缓存模型强行转成旧的 online ranker；
- 核对 encoder name、revision、tokenizer revision、embedding layer、max length 和 long-sequence strategy；
- 无法保证等价时立即报错，不接受部分加载或 `strict=False`。

#### 验收

- 一个缓存训练 checkpoint 能对从未进入缓存的新抗体序列打分；
- 同一批序列的“预建缓存推理”与“临时在线 embedding 推理”在容差内一致；
- encoder 元数据不一致时加载失败。

### P0-4：保存真正的最优 checkpoint

#### 现状

`checkpoint_latest.pt` 每轮覆盖，最后的 `checkpoint.pt` 也是最后一轮。验证集只记分，不选模型。

#### 修改

- 每轮保存 latest，用于断点续训；
- 验证指标改善时保存 best；
- 训练结束后恢复 best 权重，最终 `checkpoint.pt` 保存被选中的模型；
- checkpoint 记录 selected epoch、metric、optimizer 和必要的 RNG 状态。

#### 验收

- 人工构造第 2 轮最优、第 3 轮下降的用例，结束后模型必须等于第 2 轮；
- latest 和 best 的用途明确，不相互覆盖。

### P0-5：对未实现的公开能力提前失败

`Config` 已声明 pointwise、listwise 和 LoRA，但当前 runner/Trainer 只完整支持 RankNet 与冻结缓存主线。在能力完成前，入口层应在加载大模型、缓存或数据之前报错。不允许运行到第一个 batch 才发现任务不支持。

## 3. 第二阶段：建立数据实体和真实输入

这一阶段与集群联调可以并行，但在正式 cold-start 实验之前必须完成。

### P1-1：数据身份和冲突处理

标准表补充：

- `measurement_family_id`；
- `antibody_sequence_key` 和 `antibody_cluster_id`；
- `antigen_sequence_key` 和 `antigen_cluster_id`；
- `interaction_key`；
- 伪标签来源和 teacher 元数据。

已知需处理：

- AbRank 有 69 组完全相同的抗体—抗原—指标输入对应不同标签，共 150 条；
- `MERS_CoV` 一个 antigen key 对应两条不同序列；
- 664 个精确抗体出现在多个抗原中，涉及 60,508 条记录。

相同模型输入且标签冲突的记录不允许直接组成硬 pair。所有合并、删除或降权必须保留 provenance。

### P1-2：审计模型真正看到的序列

当前抗原最长输入为 512 token。AbRank 中 84% 的可训练记录对应抗原超过该长度；166 个 antigen key 与其他完整序列共享相同的前 510 个残基。

修改要求：

- 保存基于实际 tokenizer 和截断策略的 `effective_input_hash`；
- split 审计同时检查完整序列和有效模型输入；
- 有 assay construct 时优先使用真实构建体；
- 对长抗原比较功能区域、chunk 聚合和更长上下文编码器；
- 报告每个 split 的截断数和截断率。

### P1-3：抗体与抗原聚类

抗体聚类不直接复用抗原的全长 Hamming 逻辑。应优先使用 ANARCI 或等价编号，同时考虑 VH/VL 全长、heavy CDR3、light CDR3 和抗体类型。

抗原聚类需要：

- 统一函数和 CLI 默认阈值；
- 修复一 key 多序列后再生成正式产物；
- 完整记录阈值、linkage、输入哈希、代码 commit 和簇大小分布；
- 明确处理少量插入、缺失和长度不同的序列。

## 4. 第三阶段：实现 README 中的五种评估协议

README 已将泛化问题分为 Pair holdout、Group holdout、Within-antigen antibody holdout、Antigen cluster holdout 和 Dual cold-start。代码、产物名称和审计报告必须使用同一套含义。

### P1-4：五种协议分开实现

- Pair holdout 只作为 transductive diagnostic，允许 record 和实体重复；
- Group holdout 只隔离 `group_id`，保留为工程基线；
- Within-antigen 在全局范围隔离抗体簇，不是只在各 group 内删重；
- Antigen cluster holdout 隔离抗原簇，允许抗体簇重复；
- Dual cold-start 同时隔离抗原簇、抗体簇、interaction 和测量家族。

Dual cold-start 优先使用连通分量作为不可拆分单元。如果出现超大连通分量，必须报告它占据的 records、groups 和抗原数，不能默默塞进 train。

### P1-5：切分产物和泄露审计

每套 split 保存协议名称、输入哈希、聚类参数、seed 和代码版本。审计至少包含：

- record overlap；
- measurement family overlap；
- exact antibody 和 antibody cluster overlap；
- exact antigen、antigen cluster 和 effective input overlap；
- interaction overlap；
- group、study、assay、metric 和 label kind 分布。

审计不只输出 `PASS/FAIL`，还要说明“本协议允许哪些重叠”。

## 5. 第四阶段：predicted 弱监督与标签体系

当前 1,982,690 条可训练记录中：

- predicted：1,099,640（55.5%）；
- binary：658,943（33.2%）；
- experimental：224,107（11.3%）。

predicted 不能一刀切删除，也不能按普通实验记录等权使用。它应当作为通过门控的 teacher 信号。

### P1-6：先统一 predicted 数据口径

当前 predicted 转换脚本来自 `engelhart2022dataset` 和 `li2023machine` 的三张表，全部是 `CoV2_RBD`。它们扩充了同一抗原下的抗体密度，不是抗原多样性。

`manifest.csv` 将两张 Li 数据标为 `experimental`，而 converter 实际写入 `predicted`。需要统一 manifest、converter、合并表和训练统计的口径。

### P1-7：predicted 质量门控

每个伪标签批次按数据来源、teacher model、抗原、assay 和 metric 单独审计。

1. **来源**：记录 teacher 版本、训练数据、生成方法和文献，检查 teacher 是否看过项目 valid/test。
2. **相关性**：在独立实验数据上以 Spearman rho 为主，Pearson R 为辅，同时报告样本数、95% 置信区间、置换检验 p 值和 BH-FDR 校正后 q 值。
3. **效应量**：不能只看显著性。百万级样本中，很小的 R 也可以获得极小的 p 值。
4. **外推**：相关性必须在抗体簇或连通分量隔离后计算，不用随机 record split 证明新抗体泛化。
5. **pair 置信度**：有 ensemble 方差时结合分数差和不确定性构造 pair；没有时优先使用跨较大分位区间的 pair。

可以从以下初始门槛开始敏感性分析：独立匹配样本不少于 30，rho 不低于 0.30，95% 置信区间下界大于 0.10，q < 0.05。这些值写入配置，不写死在代码中。

### P1-8：predicted 的训练方式

- 保留 experimental-only 对照，用它判断弱监督是否真有帮助；
- 优先比较“predicted 预训练 → experimental 微调”与低权重混合；
- 混合时对 predicted 的总 loss 占比封顶，可先比较 10%、20% 和 30%；
- 一个百万级 group 必须设置 pair 数、record degree 和总 loss 上限；
- early stopping、checkpoint 选择和最终评估只看 experimental valid；
- binary 作为单独任务或明确的多任务目标，不与 continuous Spearman 混成一个总分。

## 6. 第五阶段：正式实验前再整理配置与评估

这些问题很重要，但不应阻塞当前用于测集群链路的运行。它们在开始正式模型比较前升为 P0。

### P2-1：test 隔离

- 普通训练和消融不读取最终 test；
- 单独的 final-evaluation 入口只读取已冻结 checkpoint；
- 当前反复查看的 test 改称 `dev_test`，正式实验重建未查看 holdout；
- 最终产物记录 checkpoint 和 split 哈希。

### P2-2：正式配置与单变量消融

当前 sampler 和深度配置同时改变 filter、group weight、batch size 等字段，在联调阶段可以接受，但不能用于单因素归因。

正式实验前：

- 建立一份冻结 baseline manifest；
- 为每类消融定义允许变化的字段白名单；
- 用梯度累积维持 effective batch size；
- 固定 split、标签范围、records/pairs、optimizer steps 和模型选择规则；
- 多个 seed 报告均值、标准差和原始结果；
- 用脚本自动检查 YAML 的非白名单差异。

### P2-3：正式模型选择指标

- 主指标使用 dual cold-start 下 continuous experimental macro Spearman；
- weighted Spearman 同时报告，但不让超大 group 单独决定模型；
- binary 和 predicted 分层报告，不再用混合 label kind 的 overall 选模型；
- valid 和 final test 使用同一套分层汇总逻辑。

## 7. 后续功能

以下工作不进入当前 Python 硬伤修复的关键路径：

- 完整 pointwise 和 listwise runner；
- 按 dataset、assay、metric 和重复性数据建立 tau registry；
- 统一 `X` 残基在 prepare、编码和推理中的处理；
- MSA、AbLang-2 和其他新表征；
- 长抗原 chunk 模型的性能优化；
- sampler 统一诊断报告。

## 8. 执行顺序

### 闸门 A：代码语义可信

1. frozen/LoRA 语义；
2. seed 时机；
3. cached checkpoint 推理；
4. best/latest checkpoint；
5. 未实现能力 fail-fast。

通过标准：模型实际执行与 mode 名称一致，同配置可复现，训练出来的缓存模型能对新序列推理。

### 闸门 B：数据和协议可信

1. 数据身份与冲突；
2. 有效模型输入；
3. 抗体/抗原聚类；
4. 五种 split 及泄露审计；
5. predicted 质量门控。

### 闸门 C：正式实验可解释

1. test 隔离；
2. 冻结 baseline 和单变量配置；
3. 主指标与分层指标；
4. 多 seed 复验。

只有 A、B、C 全部通过后，才重新运行并解释正式消融实验。

## 9. 每项整改的交付物

每项任务至少包含：

1. 一个能稳定复现问题的失败用例；
2. 实现代码；
3. 自动化回归测试；
4. 小规模真实链路验证；
5. 对 README、programming spec、checklist 或运行文档的同步更新。

提交说明必须回答：修复了哪个执行契约，用什么证据证明，对现有 smoke 或后续正式实验有什么影响。
