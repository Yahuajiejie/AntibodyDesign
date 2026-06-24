# AffinityTransformer 系统级质检清单

这份清单用于查找“单个函数看起来都对，整个系统却在做错事”的问题。它不按某次会话、版本或 diff 组织，而是按项目长期稳定的执行契约组织。

质检时先看端到端闭环，再钻进函数细节。每发现一个问题，先问它会不会改变模型实际看到的数据、实际优化的参数、实际选中的 checkpoint，或最终指标的含义。

## 1. 审查前先确定本次任务性质

- [ ] 本次运行是 `smoke`、`integration`、`diagnostic` 还是 `formal experiment`？
- [ ] 如果只是测集群链路，是否避免对配置差异作科学归因？
- [ ] 如果要产生正式结论，split、标签范围、主指标、test 隔离和多 seed 是否已经冻结？
- [ ] 报告中的“能跑”、“指标更高”和“泛化更好”是否被当成三个不同结论？

## 2. 高效质检方法

### 2.1 不要从“逐函数逐行读”开始

先画出四张小表：

1. **入口表**：用户、集群和测试分别从哪个入口进入。
2. **产物表**：records、split、embedding、checkpoint 和 metrics 分别是谁生成、谁消费。
3. **语义表**：每个 config mode 实际创建什么模型、哪些参数可训练。
4. **评估表**：每个 split 允许哪些实体重叠，指标回答什么问题。

这四张表如果填不完，说明工程契约本身就不清楚，此时继续检查小函数的收益很低。

### 2.2 按“纵向切片”检查

每次选一条完整路径，从入口跟到产物：

- 一条 record 如何变成 pair，最后如何进入 loss；
- 一个 encoder mode 如何变成具体模型和 optimizer 参数；
- 一个 valid 指标如何决定 checkpoint；
- 一个训练 checkpoint 如何对全新序列输出排名；
- 一个 split 名称如何对应到真实的实体隔离约束。

纵向切片比“这个文件的函数有没有 bug”更容易抓到接缝问题。

### 2.3 每条契约都要反向构造失败用例

- [ ] mode 写着 frozen，故意检查训练一步后 encoder 是否变了。
- [ ] 声称 cold-start，故意放入跨 split 的精确实体或近重复序列，看审计是否拦截。
- [ ] 声称可复现，故意改变折的运行顺序，看同一折是否变化。
- [ ] 声称可推理，使用从未进入缓存的新序列加载正式 checkpoint。
- [ ] 声称最优模型，构造中间 epoch 最好、最后 epoch 下降的用例。

## 3. 发现任意一项即中止正式运行的 P0 问题

- [ ] 配置名称与实际执行模式不一致，例如 frozen 实际全量微调。
- [ ] 未实现能力静默 fallback，例如 LoRA 变成 full fine-tuning。
- [ ] 模型在 seed 设置前初始化。
- [ ] 训练模型类型与推理模型类型不一致，checkpoint 无法严格加载。
- [ ] 验证指标没有参与 checkpoint 选择，却将最后 epoch 称为 best。
- [ ] 同一 record、interaction、禁止重叠的实体簇或测量家族跨 split。
- [ ] 先生成派生 pair，再在不允许 record 重叠的协议中随机切 pair。
- [ ] 模型实际输入因截断而相同，数据层却将它们当作独立实体。
- [ ] 一个配置声称某 objective 可用，但运行到第一个 batch 才报未实现。
- [ ] 训练过程发生 NaN/Inf，但仍保存或汇总为有效结果。

## 4. 当前目录与职责地图

### 根目录入口

| 路径 | 职责 |
| --- | --- |
| `train.py` | 加载配置、设置 run seed、选择 cached/online runner、分派 CV |
| `predict.py` | 用户命令行排名入口 |
| `README.md` | 项目科学问题、五种评估协议和模型设计 |

### Python 包

| 路径 | 职责 |
| --- | --- |
| `affinity_transformer/config.py` | YAML 到 dataclass，配置层校验 |
| `affinity_transformer/record_filter.py` | record 筛选规则 |
| `affinity_transformer/splits.py` | split 构造与泄露报告 |
| `affinity_transformer/antigen_clustering.py` | 抗原序列聚类 |
| `affinity_transformer/dataset/` | schema、records、groups、datasets 和 pair 构造 |
| `affinity_transformer/dataset/pair_sampling/` | 各种 pair sampler 与 tau 规则 |
| `affinity_transformer/embeddings/` | embedding request、extractor、cache、store、collate 和元数据校验 |
| `affinity_transformer/model/` | online/cached ranker、投影、交互、池化、头和 loss |
| `affinity_transformer/training/` | cached/online runner、loader、CV、evaluation 和 artifacts |
| `affinity_transformer/trainer.py` | 优化循环、验证、checkpoint 和旧 online 模型构建 |
| `affinity_transformer/metrics.py` | group-level Spearman 及分层汇总 |
| `affinity_transformer/user_entry.py` | 模型注册表加载、新序列编码、打分与排名 |

### 脚本与产物

| 路径 | 职责 |
| --- | --- |
| `scripts/prepare/binding/` | 各数据集 converter、prepare 和源数据测试 |
| `scripts/data/` | 记录审计、过滤和 split 构造 |
| `scripts/embeddings/` | 生成并校验 embedding cache |
| `scripts/experiments/` | 批量运行和结果汇总 |
| `scripts/runs/` | 具名运行组织 |
| `scripts/slurm/` | 集群资源、环境、依赖和任务链 |
| `configs/` | 运行配置和 model registry |
| `tests/` | 函数、契约和小型端到端回归测试 |
| `docs/future/` | 未完成整改计划 |
| `docs/checklist/` | 质检、运行流程和集群文档 |

审查时先用 `rg --files` 生成实时目录表。上表是职责地图，不应被当成永久不变的文件清单。

## 5. 配置到执行语义

- [ ] `config.py` 允许的每个 mode/objective/strategy 是否有完整 runner？
- [ ] 未实现选项是否在读大数据、下载模型或申请 GPU 前 fail-fast？
- [ ] 是否存在配置字段被接受、记录，但从未被消费？
- [ ] 是否存在 runner 里的隐藏默认值覆盖配置值？
- [ ] 同一字段在 config、loader、trainer 和 artifact 中的单位、方向和 `None` 含义是否一致？
- [ ] 使用结构化新配置和 legacy 配置时，是否产生完全相同的运行语义？

建议搜索：

```bash
rg -n "mode|objective|strategy|NotImplemented|fallback" affinity_transformer train.py
rg -n "config\.[a-zA-Z0-9_.]+" affinity_transformer train.py predict.py
```

## 6. 数据 schema、来源和标签

- [ ] 每条 record 是否能追溯到 source file 和 source row？
- [ ] `record_id` 是否稳定、唯一，不依赖当前 DataFrame 行号？
- [ ] 同一 antigen key 是否唯一对应规范化序列？
- [ ] 抗体的 heavy/light/single-chain 规则在 prepare、cache 和 inference 中是否一致？
- [ ] 方向变换后是否始终“数值越大越好”？
- [ ] `group_id` 内的 antigen、assay、metric、direction 和 label kind 是否一致？
- [ ] 相同模型输入、相同 metric 却标签冲突的记录如何处理？
- [ ] 技术重复、派生记录和同一生物样本是否共享 `measurement_family_id`？
- [ ] 可训练 records 数、可排序 groups 数和有效 pairs 数是否分开报告？

## 7. Predicted 弱监督数据

- [ ] converter、manifest、合并表和统计中的 `label_kind` 是否一致？
- [ ] 是否记录 teacher model、版本、训练数据和生成方法？
- [ ] teacher 是否可能看过本项目的 valid/test 实体？
- [ ] 相关性是否在独立实验数据和实体隔离切分上计算？
- [ ] 排序任务是否以 Spearman rho 为主，Pearson R 为辅？
- [ ] 是否同时报告效应量、样本数、置信区间、p 值和多重校正后 q 值？
- [ ] 是否避免用“样本很大，所以很小的 R 也显著”代替实际效果？
- [ ] 没有逐条不确定性时，是否避免将微小 teacher score 差构造为硬 pair？
- [ ] predicted 的 pair 数、record degree 和总 loss 占比是否封顶？
- [ ] experimental-only 对照是否始终保留？
- [ ] 模型选择和最终评估是否只看 experimental valid？

## 8. Split 与泄露审计

先写清楚当前协议允许的重叠，再检查。

| 协议 | 抗原簇可重叠 | 抗体簇可重叠 | record 可重叠 | 主要含义 |
| --- | --- | --- | --- | --- |
| Pair holdout | 是 | 是 | 是 | 已见实体关系补全 |
| Group holdout | 可能 | 可能 | 否 | 未见同质实验组 |
| Within-antigen | 是 | 否 | 否 | 已知抗原的新抗体 |
| Antigen cluster holdout | 否 | 是 | 否 | 新抗原、可已见抗体 |
| Dual cold-start | 否 | 否 | 否 | 新抗原与新抗体 |

当前如果审查目标是“四种协议从数据处理到 split 导出”，本轮只看：

| 本轮协议 | 主要入口 | 必须隔离/检查 | 备注 |
| --- | --- | --- | --- |
| Group holdout | `scripts/data/build_splits.py --strategy group_holdout_split` | `record_id`、`group_id` 不跨 split | 用作数据与导出管道 smoke test，不代表实体 cold-start |
| Antibody cold-start | `scripts/data/build_antibody_cold_start_split.py` | `record_id`、`measurement_family_id`、`interaction_key`、`antibody_sequence_key`、`antibody_cluster_id` 不跨 split | 不强制要求 `antigen_cluster_id` 或 effective-input audit |
| Antigen cold-start | `scripts/data/build_antigen_cold_start_split.py` | `record_id`、`measurement_family_id`、`interaction_key`、`antigen_sequence_key`、`antigen_cluster_id` 不跨 split；valid/test 抗体簇必须在 train 出现 | representation annotation 存在时必须审 effective-input overlap |
| Dual cold-start | `scripts/data/build_dual_cold_start_split.py` | 抗体、抗原、measurement family、interaction、group、record 全部不跨 split | 先看 `component_summary.csv`，防止超大 component 使协议不可行 |

本轮暂不审 pair holdout；within-antigen 和 antigen-cluster holdout 只作为历史/辅助路径检查，不作为四协议导出目标。

### 8.1 四协议 split 导出流水线

目标不是训练，而是确认下面这条链能在真实数据上重复运行：

```text
all_records.parquet
→ entity_annotations.parquet
→ 可选 representation_annotations.parquet
→ group / antibody / antigen / dual 四个 split 目录
→ train.parquet / valid.parquet / test.parquet / audit artifacts
```

- [ ] `all_records.parquet` 是否能由 `scripts/prepare/binding/merge_records.py` 稳定重建？
- [ ] `record_id` 是否唯一、稳定，且不会因为 DataFrame 行号变化而变化？
- [ ] `group_id`、`rank_label`、`keep_for_training` 是否足够支撑四种协议的过滤和评估？
- [ ] `entity_annotations.parquet` 是否是窄表，而不是把所有审计字段塞回 base records？
- [ ] annotation 是否覆盖所有输入 records，且 `record_id` 无缺失、无重复？
- [ ] `antibody_sequence_key → antibody_cluster_id` 是否一对一映射到唯一 cluster？
- [ ] `antigen_sequence_key → antigen_cluster_id` 是否一对一映射到唯一 cluster？
- [ ] `interaction_key` 是否由实体键稳定派生，而不是依赖手填名称？
- [ ] `measurement_family_id` 是否不会过粗到把大量无关 records 串成一个 component？
- [ ] 专用 cold-start 脚本是否是 canonical 导出入口，旧 `build_splits.py` 的兼容分支是否不会被误当成正式 artifact 入口？
- [ ] 四个协议目录是否全部写出 `train.parquet`、`valid.parquet` 和 `test.parquet`？
- [ ] entity cold-start 协议目录是否写出 `split_manifest.yaml`、`component_assignments.parquet`、`eligibility_report.csv`、`excluded_records.parquet`、`leakage_report.csv` 和 `summary.csv`？
- [ ] dual 协议是否额外写出 `component_summary.csv`？
- [ ] 写出的 train/valid/test parquet 是否默认剥离 entity annotation 字段，保持 base records schema？
- [ ] 每个 `leakage_report.csv` 是否全部 PASS；若 FAIL，是否中止而不是继续进入训练？
- [ ] 每个 `excluded_records.parquet` 是否带有可解释的 `protocol_exclusion_reason`？
- [ ] 每个协议的 valid/test 是否非空，且有足够可计算 Spearman 的 group？
- [ ] 如果提供 `representation_annotations.parquet`，manifest 是否记录 `effective_input_audited: true`？
- [ ] 如果未提供 representation annotation，manifest 是否明确记录未做 effective-input audit？
- [ ] dual 的最大 component 占比是否被记录；如果过大，是否回查 cluster / measurement family / interaction 定义，而不是直接改 split 算法？

- [ ] 除明确的 Pair holdout 外，是否先切 records 再在 split 内构造 pairs？
- [ ] split unit 是 `group_id`、antigen cluster、antibody cluster 还是 connected component？
- [ ] Within-antigen 是否在全局范围隔离抗体簇，而不只在当前 group 内？
- [ ] Dual cold-start 是否将抗原簇、抗体簇、interaction 和测量家族形成不可拆分分量？
- [ ] 超大分量是否被报告，而不是默认塞入 train？
- [ ] 是否检查 exact 序列、近重复簇和 effective input hash 三个层次？
- [ ] split artifact 是否记录输入哈希、聚类参数、seed、协议名称和代码版本？
- [ ] 训练启动时是否核对 split manifest，防止文件被替换？

## 9. Embedding、截断和缓存

- [ ] cache key 是否包含实际编码输入所需的链、类型和序列信息？
- [ ] 不同抗体类型或 heavy/light 组合是否可能冲突到同一 hash？
- [ ] encoder name、model revision、tokenizer revision、layer、dtype、max length 和长序列策略是否全部进入元数据契约？
- [ ] 特殊 token 和 padding 是在哪一层删除，mask 是否与最终 token 对齐？
- [ ] 长序列是报错、截断还是 chunk，配置和 extractor 实际行为是否一致？
- [ ] 是否报告每个 split 的截断数、比例和有效输入冲突？
- [ ] 缓存覆盖率是按真正要训练/评估的 records 计算，还是按一张未过滤表计算？
- [ ] 缓存不完整或 metadata 不匹配时是否硬失败？
- [ ] 预建缓存与新序列在线 embedding 是否有数值一致性测试？

## 10. 模型、任务头和 optimizer

- [ ] 每个 encoder mode 实际创建的模型类是否正确？
- [ ] frozen encoder 的 `requires_grad`、`training` 状态和 forward graph 是否都被冻结？
- [ ] optimizer 是否只包含 `requires_grad=True` 的参数？
- [ ] run artifact 是否记录总参数和各模块可训练参数？
- [ ] 是否只有一个共享 scalar head，还是按 dataset/group/label kind 创建多个头？
- [ ] 如果有多头，训练、valid、test 和新用户输入的路由键是什么？推理时不知道 dataset 时如何选头？
- [ ] 多头是不是不小心把数据集 ID 变成了答案捷径？
- [ ] pointwise/pairwise/listwise 是共享同一个可推理分数，还是只在训练阶段临时适配？
- [ ] 缺失抗原或某表征路径时，mask 是否真正阻止了占位向量参与注意力？
- [ ] model train/eval 切换是否会意外改变冻结编码器的 dropout？

建议搜索：

```bash
rg -n "ModuleDict|ParameterDict|head|dataset_id|group_id|label_kind" affinity_transformer/model affinity_transformer/trainer.py
rg -n "requires_grad|optimizer|\.train\(|\.eval\(" affinity_transformer
```

## 11. Pair sampler、loss 和实际训练权重

- [ ] pair 只在同一 `group_id` 内构造吗？
- [ ] tie 如何处理，是跳过、soft target 还是当成硬顺序？
- [ ] binary group 是否只使用跨类 pair？
- [ ] sampler 能否保证 record coverage，还是有大量 records 永远不进 loss？
- [ ] 近邻 pair、远距离 pair 和噪声阈值的实际比例是多少？
- [ ] 每个 group 的 pair 数、degree 和 loss 权重是多少？
- [ ] `build_pairs` 被多处调用时，strategy、seed 和全部参数是否完全一致？
- [ ] group reweight 修正的是目标分布，还是反而让伪标签超大 group 主导 loss？
- [ ] 报告的 record 数是否与真正进入 loss 的有效 records 数区分开？
- [ ] 用多 worker 时，sampler 和 worker seed 是否稳定？

## 12. 可复现性与运行状态

- [ ] seed 是否在模型、DataLoader、sampler 和 optimizer 构建前设置？
- [ ] Python、NumPy、PyTorch CPU 和 CUDA 是否全部播种？
- [ ] K 折每折是否在模型构建前独立重置 RNG？
- [ ] 同一折是否不依赖折的运行顺序？
- [ ] DataLoader worker 是否有明确的 worker init seed 和 generator？
- [ ] 如果 CUDA 不启用完全确定性，run log 是否说明可能的非确定性？
- [ ] 两个独立进程的初始参数哈希、首批 pair 和首轮 loss 是否一致？

## 13. Checkpoint 与推理闭环

- [ ] latest 和 best 是否分开保存？
- [ ] best 是否真的由 valid 指标选出，而不是最后 epoch？
- [ ] 训练结束后内存中的模型是否恢复为 best？
- [ ] latest 是否保存 optimizer、epoch、global step 和必要的运行状态？
- [ ] checkpoint 中的 config 与外部 config 不同时是否拒绝加载？
- [ ] embedding metadata hash 不同时是否拒绝加载？
- [ ] 加载是否使用 `strict=True`，禁止靠 `strict=False` 掩盖架构不一致？
- [ ] cached 训练 checkpoint 是否构建 `EmbeddingAffinityRanker`，而不是旧 `AffinityRanker`？
- [ ] 用户输入新序列时，是否用与训练缓存相同的 extractor 契约生成 embedding？
- [ ] 抗体排序是否在各 `query_id` 内独立进行？

## 14. 评估和 test 生命周期

- [ ] Spearman 是否在 group 内计算，没有将不可比较 records 混成全局相关？
- [ ] macro 和 weighted 是否分开报告？
- [ ] weighted 是按 records 还是 pairs 加权，文档与实现是否一致？
- [ ] binary、experimental 和 predicted 是否分层报告？
- [ ] 主模型选择指标是否与主任务一致？
- [ ] 小 group 因标签不变或样本过少无法计算 Spearman 时，是否报告 skipped 数？
- [ ] valid 和 test 是否使用同一套指标和分层逻辑？
- [ ] 当前 test 是工程联调集还是一次性最终 holdout？
- [ ] 正式实验时，普通 runner 是否完全不读最终 test？
- [ ] 结果汇总是否会将 test 列和 valid 列一起用于排序配置？

## 15. 正式实验归因

本节对 smoke/integration 不作硬性要求，对 formal experiment 是必查项。

- [ ] 是否有一份冻结 baseline manifest？
- [ ] 每组消融是否有可变字段白名单？
- [ ] sampler 消融是否同时改了 filter、group weight、batch size 或 split？
- [ ] 深度消融是否保持 effective batch size 和 optimizer steps 一致？
- [ ] 不同配置实际看到的 records、pairs 和标签权重是否一致？
- [ ] 计算预算差异是否被误解为架构差异？
- [ ] 每个结论是否至少经过多 seed 复验？
- [ ] 联合改动是否被如实命名，而不是冒充单变量消融？

## 16. 集群和资源链路

- [ ] SLURM 请求的 GPU、CPU、内存和时间是否与真实模式匹配？
- [ ] frozen 模式是否意外变成全量反传，使原本的 batch size 直接爆显存？
- [ ] 计算节点无网络时，所有模型和 tokenizer revision 是否已锁定且预下载？
- [ ] cache 是否完整生成后再被训练使用，部分目录是否会被当成成功产物？
- [ ] 中断后重跑是覆盖、续训还是新建目录，行为是否明确？
- [ ] 多 worker 读取 mmap/shard 时是否有真实节点压力测试？
- [ ] run log 是否记录主机、GPU、代码 commit、config、split/cache hash 和参数量？

## 17. 测试覆盖的正确层次

每类测试回答的问题不同，不要用大量单元测试代替全局契约测试。

| 层次 | 应当证明什么 |
| --- | --- |
| 单元测试 | 一个函数在小输入上的精确行为 |
| 契约测试 | config 与 runner、sampler 与 dataset、checkpoint 与 loader 等接口一致 |
| 集成测试 | 小数据从入口走到 checkpoint/metrics |
| 反例测试 | 错误 mode、泄露 split、错元数据和架构不匹配会被拒绝 |
| 真实环境测试 | CUDA、混合精度、多 worker、分片缓存和 SLURM 资源真正可用 |
| 科学验收 | 在冻结协议上，结论经多 seed 且可归因 |

至少保留以下端到端回归：

- [ ] frozen online 训练一步，encoder 不变；
- [ ] 同 seed 两次运行，初始状态一致；
- [ ] 中间 epoch 最好时，训练结束恢复 best；
- [ ] frozen-cached checkpoint 对新序列完成排名；
- [ ] 每种 split 用故意泄露的合成数据验证拦截规则；
- [ ] 完整小型 cached 链路：records → cache → train → best checkpoint → new-sequence inference。

## 18. 审查记录模板

每个问题使用同一格式记录：

| 字段 | 内容 |
| --- | --- |
| 主张 | 项目原本声称会做什么 |
| 实际路径 | 入口 → 关键函数 → 产物 |
| 反例 | 如何稳定复现问题 |
| 影响 | 影响运行、训练语义、数据泄露还是结论解释 |
| 严重度 | P0 停止运行 / P1 正式实验前修 / P2 质量改进 |
| 修复 | 需改哪些契约，不只写某一行 patch |
| 验收 | 哪个自动化测试会在回归时失败 |
| 历史结果 | 哪些运行需要重命名、降级或重跑 |

## 19. 建议的审查顺序

如果当前任务是“四种协议 split 导出”，先按下面这个短顺序审，不要先跳到训练、checkpoint 或模型：

1. 从 `scripts/prepare/binding/merge_records.py` 确认 `all_records.parquet` 如何生成。
2. 审 `record_id`、`group_id`、`rank_label`、`keep_for_training` 的稳定性和可用性。
3. 审 `affinity_transformer/annotations/`：annotation 窄表、覆盖率、唯一性、临时 join 和 base schema 隔离。
4. 审 cluster / entity key 的生成假设：`antibody_sequence_key`、`antibody_cluster_id`、`antigen_sequence_key`、`antigen_cluster_id`、`measurement_family_id`、`interaction_key`。
5. 先跑/审 group holdout，确认基础导出管道没坏。
6. 再审 antibody cold-start 专用脚本和产物。
7. 再审 antigen cold-start 专用脚本和产物。
8. 最后审 dual cold-start，重点看 component 可切性和 `component_summary.csv`。
9. 汇总四个协议目录的 train/valid/test 行数、leakage 状态、excluded 原因和 manifest。

只有这条链稳定后，再回到训练相关审查。

长期全系统审查仍可使用下面的完整顺序：

1. 读 README 的问题定义和评估协议。
2. 用 `rg --files` 重建当前目录地图。
3. 从 `train.py` 跟一次 cached 和 online 路径。
4. 从 `predict.py` 跟到正式 checkpoint 的新序列排名。
5. 从一条 raw record 跟到 split、pair 和 loss weight。
6. 核对 encoder mode、seed、optimizer 和 checkpoint。
7. 核对五种 split 的允许/禁止重叠。
8. 核对 predicted 来源、效应量和训练权重。
9. 最后才逐函数检查算法细节、异常路径和代码风格。

这个顺序的核心是：先确认项目在做正确的事，再确认每个函数把这件事做得够不够漂亮。
