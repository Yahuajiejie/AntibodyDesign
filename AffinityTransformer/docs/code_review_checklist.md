# 质检顺序指南

本文回答两个问题：自上次质检(`dataset/`、`model/`、`embeddings/`)以来改动这么多，**要不要重新质检**——要；**按什么顺序看，才比一行一行翻代码更高效**——按依赖层次自底向上看，每层只读"契约 + 这次的diff"，不要重读整个文件。

## 0. 方法论：为什么这么排序

一个训练系统的代码，天然分层：下层是不依赖任何其他模块的纯函数（一个`group_id`对应多少条pair、一个loss怎么算），上层是把下层拼起来的编排逻辑（`Trainer.fit()`、`train.py`）。**下层的bug是"地基级"的**——一个`_pair_sample_count`算错，所有上层调用它的代码表现都会跟着错，但你在上层看不出问题出在这里，只会看到"训练效果不对"这种模糊症状。所以质检顺序应该是：

1. **先看契约（config.py / schema.py）**：这一层定义了"别的模块可以假设什么"。如果契约本身有歧义或漏洞（比如某个yaml字段没校验、默认值跟文档描述不一致），后面每一层都可能在错误的假设上建房子。
2. **再看纯函数（pair_sampling/common.py、losses.py、metrics.py）**：这些函数输入输出明确，不依赖外部状态，最容易独立验证对错，也是性价比最高的质检对象——花小成本验证一个纯函数，能排除掉一大片潜在bug源。
3. **再看构造层（dataset/、embeddings/）**：这一层把纯函数组装成可训练的数据结构，依赖第1、2层。
4. **再看模型结构（model/）**：通常改动最少（这次diff几乎没碰这一层），可以快速过一遍确认真的没改。
5. **最后看编排层（trainer.py、training/、train.py）**：这一层依赖前面所有层，也是这次改动最集中、最容易出"接线错误"（哪个变量没传到位、哪个分支条件写反）的地方，留到最后看，因为只有确认了下层没问题，才能放心说"如果训练结果不对，问题在这一层"。
6. **配置文件（yaml）和脚本（slurm）单独放一层**——它们不是Python逻辑，但决定了上面那些代码实际怎么被参数化调用，一个yaml字段名拼错可能完全不报错（被`section.get(key, default)`默默吃掉），必须跟`config.py`的解析逻辑对照着看，不能单独看。
7. **测试放最后**：读完实现再读测试，去判断"这个测试是真的验证了新逻辑的关键分支，还是只是没让代码跑崩"——测试覆盖率不等于测试质量，需要带着"这个测试如果改实现里的某一行会不会失败"的心态去读。

每一层给出：变化范围（行数）、为什么要重新看、看的时候具体盯哪里、用什么`git diff`命令把质检范围缩小到真正变化的部分。

**质检范围**：本文档覆盖 `debug000` ~ `debug009`（12个commit，52个文件，+1376/-328行）相对于你上次质检基线的全部改动。本轮会话里还有一批**尚未提交**的改动（树形/堆/随机BST采样结构），单独列在第8层，不要跟前面混在一起看。

---

## Layer 0：配置契约

- [ ] `affinity_transformer/config.py` 🔧 +96行（这次最大改动之一）
  - 看什么：`DataConfig`新增的字段（`cross_validation`整块、`weight_pairs_by_group_size`、`tree_extra_random_pairs_per_group`等）是否在**三处**保持一致——dataclass默认值、`load_config`里的`section.get(key, default)`解析、`_validate_pair_sampling`里的校验范围。三处不一致是最容易藏雷的地方，因为yaml拼错字段名时`section.get`会静默回退到默认值，不报错。
  - 命令：`git diff a17dd18 d765a17 -- affinity_transformer/config.py`
- [ ] `affinity_transformer/dataset/schema.py` ⏭ 未变
  - 确认一下：`git diff a17dd18 d765a17 -- affinity_transformer/dataset/schema.py` 应为空，跳过。

## Layer 1：纯函数 / 底层工具（性价比最高，优先看）

- [ ] `affinity_transformer/model/losses.py` 🔧 +9行
  - 看什么：`ranknet_loss`新增的`weight`参数有没有改变`weight=None`时的行为（应该完全等价于改动前）；`F.binary_cross_entropy_with_logits(..., weight=weight)`的weight语义是"每个样本的loss乘这个系数"还是"梯度也按比例缩放"——确认跟`training/loaders.py`里`compute_group_pair_weights`算出来的权重含义对得上。
  - 命令：`git diff a17dd18 d765a17 -- affinity_transformer/model/losses.py`
- [ ] `affinity_transformer/metrics.py` ⏭ 未变（`git diff`确认即可）
- [ ] `affinity_transformer/dataset/pair_sampling/common.py` ⏭ 这次debug提交里未变；**本轮会话有未提交改动，见第8层**

## Layer 2：数据集构造

- [ ] `affinity_transformer/splits.py` 🆕 整个文件新增，67行
  - 看什么：`build_group_kfolds`——这是`cross_validation.py`唯一依赖的核心函数，重点查"按`group_id`切分"有没有可能让同一个`group_id`同时出现在train和valid里（一旦发生就是数据泄露，而且不会报错，只会让CV分数虚高）。
  - 命令：`git diff a17dd18 d765a17 -- affinity_transformer/splits.py`
- [ ] `affinity_transformer/dataset/groups.py`、`dataset/datasets.py`、`dataset/examples.py`、`dataset/pairs.py`（debug提交范围内）⏭ 未变
- [ ] `affinity_transformer/dataset/pair_sampling/{blocks,large_group,two_label,labels,validation}.py` ⏭ 未变（本轮会话对`validation.py`有未提交改动，见第8层）

## Layer 3：Embedding / 特征层

- [ ] `affinity_transformer/embeddings/store.py` 🔧 **128行改动，本次debug提交里改动最大的单个文件**
  - 看什么：debug007提到"allow using mmap"——重点查mmap模式下的并发读取是否安全（多个DataLoader worker进程同时mmap同一个shard文件，要确认没有写时复制或缓存一致性问题）；mmap失败时是否有清晰的报错而不是静默返回错误数据；之前`debug006`提到"remove LRU which decreases GPU efficiency"，确认LRU移除后没有引入内存无限增长的风险（之前靠LRU做的淘汰逻辑现在靠什么替代）。
  - 命令：`git diff a17dd18 d765a17 -- affinity_transformer/embeddings/store.py`
  - 这是本轮质检里**风险最高**的一项，建议单独留出时间，不要跟其他文件一起扫。
- [ ] `embeddings/{schema,collate,extractors,huggingface,pipeline,validation}.py` ⏭ 未变

## Layer 4：模型结构

- [ ] `model/{ranker,embedding_ranker,factory,attention,blocks,interaction,heads,pooling,projections}.py` ⏭ 全部未变
  - 命令：`git diff a17dd18 d765a17 -- affinity_transformer/model/` 应该只剩`losses.py`的改动，其余为空——这一层可以快速确认完直接跳过，不用展开看。

## Layer 5：训练循环与编排（本次改动最密集的一层，留在最后看）

- [ ] `affinity_transformer/trainer.py` 🔧 **+129行，本次第二大改动**
  - 看什么：
    1. bf16 `torch.autocast`的作用范围——只包in前向（forward+loss），还是不小心也包住了`optimizer.step()`（包住step会导致优化器状态精度异常）。
    2. `group_weights`/`_batch_group_weights`——`group_weights=None`时必须完全退化成原来的无权重行为，确认这条退化路径有没有被测试覆盖。
    3. `checkpoint_latest.pt`按epoch保存的逻辑跟"NaN loss报错保存error_context"这两条路径有没有冲突（比如NaN发生在保存checkpoint之后还是之前，会不会留下一个对应checkpoint的脏状态）。
    4. `valid_group_metrics_epoch{N}.csv`的导出会不会在`output_dir`很大、组数很多时显著拖慢每个epoch的eval阶段。
  - 命令：`git diff a17dd18 d765a17 -- affinity_transformer/trainer.py`
- [ ] `affinity_transformer/training/samplers.py` 🆕 整个文件新增，44行（`GroupShuffleSampler`）
  - 看什么：这是控制"每个组内的pair会不会被打散到不同batch"的关键逻辑——确认`__len__`跟`__iter__`实际产出的数量一致（`DataLoader`依赖`__len__`算总step数，算错会导致进度条/早停逻辑基于错误的总数判断）。
- [ ] `affinity_transformer/training/cross_validation.py` 🆕 整个文件新增，131行
  - 看什么：这个模块现在没有任何v065配置启用它（`cross_validation.enabled: false`），但必须确认`run_training`里的dispatch分支万无一失——如果有人手滑把某个yaml的`enabled`改成`true`，会不会因为`all_records_path`缺失或其他前提没满足而在跑了很久之后才报错（理想情况应该在`load_config`阶段就快速失败，而不是等数据加载完才发现配置不对）。
  - 命令：`git diff a17dd18 d765a17 -- affinity_transformer/training/cross_validation.py`
- [ ] `affinity_transformer/training/loaders.py` 🔧 +60行（含`compute_group_pair_weights`）
  - 看什么：`compute_group_pair_weights`内部又跑了一次`_build_pairs`——确认这跟`build_cached_train_loader`/`build_online_train_loader`内部那次`_build_pairs`用的是完全一致的参数（尤其`seed`），否则两次算出的pairs集合不一致，权重就对不上真正喂给模型的那批pair。
  - 命令：`git diff a17dd18 d765a17 -- affinity_transformer/training/loaders.py`
- [ ] `training/{cached,online}.py` 🔧 各+7行（`group_weights=compute_group_pair_weights(...)`接入点）⏭ 改动很小，跟着上面一起看
- [ ] `training/__init__.py` 🔧 +8行（导出`run_group_kfold_cross_validation`）⏭
- [ ] `train.py` 🔧 +11行（`cross_validation.enabled`的dispatch）
  - 命令：`git diff a17dd18 d765a17 -- train.py`

## Layer 6：配置文件与脚本（必须对照Layer 0一起看，不能单独看）

- [ ] `configs/v065/*.yaml` 🆕 4个新文件，每个63行
  - 看什么：四个文件（concat/deep4/deep8/deep16）除了`train.batch_size`和`model.interaction`不同，其余字段应该完全一致——用`diff configs/v065/v065_concat_ranknet.yaml configs/v065/v065_deep4_ranknet.yaml`之类的两两对比，确认没有手改时漏改某一个文件的情况。
- [ ] `configs/{ablation_*,baseline_*,experiments/*}.yaml` 🔧 每个文件4~6行改动
  - 看什么：这批改动大概率是统一加`PYTORCH_CUDA_ALLOC_CONF`相关或`batch_size`调整——抽查2~3个确认改动模式一致即可，不需要每个都精读。
  - 命令：`git diff a17dd18 d765a17 -- configs/`
- [ ] `scripts/slurm/run_config.sbatch`、`submit_v065_training_chain.sh`、`build_v065_embedding_cache.sbatch` 🔧
  - 看什么：`SKIP_G00`/`SKIP_CONCAT`之类的环境变量开关有没有互相冲突的组合（比如`SKIP_CONCAT=1`但`SKIP_G00`没设，会不会引用到不存在的上游产物路径）。
- [ ] `scripts/analysis/group_size_stats.py` 🆕（本轮会话产物，纯诊断脚本，不在训练主链路上，可以最后看或跳过）

## Layer 7：测试（实现看完之后再看，带着"这个测试能不能抓出我刚才担心的那个问题"去读）

- [ ] `tests/test_config.py` 🔧 +54行 —— 对应Layer 0新字段的校验
- [ ] `tests/test_splits.py` 🔧 +49行 —— 对应Layer 2的`build_group_kfolds`，重点确认有没有专门测"同一个group_id不会跨fold泄露"
- [ ] `tests/test_cross_validation.py` 🆕 89行 —— 对应Layer 5的`cross_validation.py`
- [ ] `tests/test_training_loaders.py` 🆕 95行 —— 对应Layer 5的`compute_group_pair_weights`/`GroupShuffleSampler`
- [ ] `tests/test_trainer.py` 🔧 仅+2行 —— 改动很小，但要重点确认：`group_weights`这个新增的大功能为什么测试只加了2行？大概率说明这部分**没有被测试覆盖**，建议质检时手动补一个"group_weights=None时输出跟改动前一致"的回归测试，而不是假设它没问题。
- [ ] `tests/embeddings/test_store.py` 🔧 仅+3行 —— 对应Layer 3那个128行的mmap改动，**测试只加了3行，覆盖严重不足**，这是本次质检里第二个"测试行数跟实现改动行数明显不成比例"的危险信号，值得重点手动验证。
- [ ] `tests/test_v065_training_scripts.py` 🔧 98行改动（净减少，可能是重构）

---

## Layer 8：本轮会话尚未提交的改动（单独验收，不要跟上面混在一起commit）

这部分是树形/堆/随机BST pair采样结构的实现，还没有进入任何debug commit：

- [ ] `affinity_transformer/dataset/pair_sampling/tree.py` 🆕（中位数平衡树 + 冗余边）
- [ ] `affinity_transformer/dataset/pair_sampling/randomized_tree.py` 🆕（随机BST + 树高保险 + 兜底）
- [ ] `affinity_transformer/dataset/pair_sampling/heap_tree.py` 🆕但已废弃（堆数组方案，已确认有结构性偏置问题，文件内容已清空只留说明，不参与`pair_sample_strategy`）
- [ ] `affinity_transformer/dataset/pair_sampling/common.py` 🔧（新增共享的`_emit_pair`）
- [ ] `affinity_transformer/dataset/pair_sampling/validation.py` 🔧（`balanced_tree`/`randomized_bst`两个新策略名）
- [ ] `affinity_transformer/dataset/pairs.py` 🔧（`build_pairs`里新增两个分支）
- [ ] `affinity_transformer/training/loaders.py` 🔧（`tree_extra_random_pairs_per_group`参数透传）
- [ ] `affinity_transformer/config.py` 🔧（同上参数的dataclass字段+解析）

命令：`git diff` （工作区相对于`d765a17`的全部未提交改动）

这一批目前**没有接入任何v065 yaml配置**（四个配置文件仍是`pair_sample_strategy: capped_proportional`），纯属新增、未启用，质检通过后才建议考虑要不要在某个配置上切换试跑，不要在还没验证reweight效果之前就一起提交进同一个commit。
