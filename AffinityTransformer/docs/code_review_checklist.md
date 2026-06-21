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

**质检范围**：本文档覆盖 `debug000` ~ `debug009`（12个commit，52个文件，+1376/-328行）相对于你上次质检基线的全部改动。本轮会话里还有两批**尚未提交**的改动：树形/堆/随机BST采样结构在第8层，`noise_aware_multiscale`（多尺度噪声感知采样+tau注册表）在第9层——两批是先后两次会话做的，互相独立，不要混在一起看。

---

## Layer 0：配置契约

- [x] `affinity_transformer/config.py` 🔧 +96行（这次最大改动之一）
  - 看什么：`DataConfig`新增的字段（`cross_validation`整块、`weight_pairs_by_group_size`、`tree_extra_random_pairs_per_group`等）是否在**三处**保持一致——dataclass默认值、`load_config`里的`section.get(key, default)`解析、`_validate_pair_sampling`里的校验范围。三处不一致是最容易藏雷的地方，因为yaml拼错字段名时`section.get`会静默回退到默认值，不报错。
  - 命令：`git diff a17dd18 d765a17 -- affinity_transformer/config.py`
- [x] `affinity_transformer/dataset/schema.py` ⏭ 未变
  - 确认一下：`git diff a17dd18 d765a17 -- affinity_transformer/dataset/schema.py` 应为空，跳过。

## Layer 1：纯函数 / 底层工具（性价比最高，优先看）

- [x] `affinity_transformer/model/losses.py` 🔧 +9行
  - 看什么：`ranknet_loss`新增的`weight`参数有没有改变`weight=None`时的行为（应该完全等价于改动前）；`F.binary_cross_entropy_with_logits(..., weight=weight)`的weight语义是"每个样本的loss乘这个系数"还是"梯度也按比例缩放"——确认跟`training/loaders.py`里`compute_group_pair_weights`算出来的权重含义对得上。
  - 命令：`git diff a17dd18 d765a17 -- affinity_transformer/model/losses.py`
- [x] `affinity_transformer/metrics.py` ⏭ 未变（`git diff`确认即可）
- [ ] `affinity_transformer/dataset/pair_sampling/common.py` ⏭ 这次debug提交里未变；**本轮会话有未提交改动，见第8层**

## Layer 2：数据集构造

- [x] `affinity_transformer/splits.py` 🆕 整个文件新增，67行
  - 看什么：`build_group_kfolds`——这是`cross_validation.py`唯一依赖的核心函数，重点查"按`group_id`切分"有没有可能让同一个`group_id`同时出现在train和valid里（一旦发生就是数据泄露，而且不会报错，只会让CV分数虚高）。
  - 命令：`git diff a17dd18 d765a17 -- affinity_transformer/splits.py`
- [x] `affinity_transformer/dataset/groups.py`、`dataset/datasets.py`、`dataset/examples.py`、`dataset/pairs.py`（debug提交范围内）⏭ 未变
- [x] `affinity_transformer/dataset/pair_sampling/{blocks,large_group,two_label,labels,validation}.py` ⏭ 未变（本轮会话对`validation.py`有未提交改动，见第8层）

## Layer 3：Embedding / 特征层

- [x] `affinity_transformer/embeddings/store.py` 🔧 **128行改动，本次debug提交里改动最大的单个文件**
  - 看什么：debug007提到"allow using mmap"——重点查mmap模式下的并发读取是否安全（多个DataLoader worker进程同时mmap同一个shard文件，要确认没有写时复制或缓存一致性问题）；mmap失败时是否有清晰的报错而不是静默返回错误数据；之前`debug006`提到"remove LRU which decreases GPU efficiency"，确认LRU移除后没有引入内存无限增长的风险（之前靠LRU做的淘汰逻辑现在靠什么替代）。
  - 命令：`git diff a17dd18 d765a17 -- affinity_transformer/embeddings/store.py`
  - 这是本轮质检里**风险最高**的一项，建议单独留出时间，不要跟其他文件一起扫。
- [x] `embeddings/{schema,collate,extractors,huggingface,pipeline,validation}.py` ⏭ 未变

## Layer 4：模型结构

- [x] `model/{ranker,embedding_ranker,factory,attention,blocks,interaction,heads,pooling,projections}.py` ⏭ 全部未变
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

---

## Layer 9：本轮会话尚未提交的改动（noise_aware_multiscale + tau registry，单独验收）

这一层接在Layer 8之后——如果Layer 8那批`tree.py`/`randomized_tree.py`已经提交，这里的`git diff`基线请换成那次提交后的hash；这个沙箱看不到git，下面的命令里基线hash都留了占位符`<BASELINE>`，自己核对替换。

背景一句话：`randomized_bst`/`balanced_tree`在密集大组里会选出标签差小到跟测量噪声分不清的比较对（`docs/experiments/noise_floor_tree_analysis.md`记录了诊断过程），`noise_aware_multiscale`是这轮的修法；中途还经历了一个失败的原型（`noise_floor_tree.py`，单链聚类会把密集组串成一个簇），看的时候要分清"现在生效的是哪一版"。

- [ ] `affinity_transformer/dataset/pair_sampling/noise_aware_multiscale.py` 🆕 整个文件新增，约300行（核心算法）
  - 看什么，按风险从高到低：
    1. `_build_tau_separated_anchors`——确认比较的是"当前点 vs 当前锚点"，不是"当前点 vs 上一个点"。后者（单链聚类）正是`noise_floor_tree.py`那个失败原型的bug：密集区会被一路串成一个簇。这是这个文件存在的全部理由，如果这一行写错，整个模块的设计目的就落空了。
    2. `_choose_degree_balanced_partner`的`max_degree`参数——coverage阶段必须传`None`（永不因度数拒绝一个本来能连上的partner，否则违反"存在可辨认partner就必须连上"的目标），enrichment阶段必须传一个真实数值（硬上限，超了就放弃这条边，否则又会退化成度数集中）。这两个调用点的`max_degree`传参容易看混。
    3. `_stable_second_band`——必须用`hashlib`而不是Python内置`hash()`。内置`hash()`对字符串的结果默认按进程随机化（`PYTHONHASHSEED`），同样的seed换一个进程跑会得到不同结果，悄悄破坏"相同seed必须复现"这条契约，而且不会报错，只会在"重新跑一次结果不一样"的时候才被发现。
    4. `_has_any_resolvable_partner`——确认它是用排序后的首尾两个极值做O(1)判断，不是真的去遍历找一遍；如果改成了遍历，单条记录的判断会变成O(n)，整组就退化回O(n²)。
  - 命令：`git diff <BASELINE> -- affinity_transformer/dataset/pair_sampling/noise_aware_multiscale.py`
- [ ] `affinity_transformer/dataset/pair_sampling/tau_registry.py` 🆕 整个文件新增，约150行
  - 看什么：`_RULES`元组里的正则规则有没有互相重叠——比如`SARS_CoV_2`精确匹配和`SARS_CoV_2_.+`前缀匹配理论上不会同时命中同一个`antigen_key`，但如果以后有人往`_RULES`里加新规则、又没注意顺序，`resolve_tau_for_group`是按列表顺序`pattern.match`后第一个命中就返回，规则顺序变了结果就可能变。另外确认`resolve_tau_for_group`对"一个group里出现多个不同`antigen_key`"这种异常输入是真的抛错，不是默默取第一个——这是质检脚本里最容易漏测的边界情况。
  - 命令：`git diff <BASELINE> -- affinity_transformer/dataset/pair_sampling/tau_registry.py`
- [ ] `affinity_transformer/dataset/pair_sampling/noise_floor_tree.py` 🗑 内容清空，只留说明（之前的失败原型）
  - 看什么：确认`__init__.py`/`pairs.py`/`validation.py`/`config.py`里没有任何地方还在`import`这个文件——`grep -rn "noise_floor_tree" affinity_transformer/`应该只剩这个文件自己。沙箱权限没法真删这个文件，只清空了内容，正式仓库里如果能删就直接删掉，不用保留这个空文件。
- [ ] `affinity_transformer/dataset/pair_sampling/__init__.py` 🔧 +4行（新增两个导出）
  - 命令：`git diff <BASELINE> -- affinity_transformer/dataset/pair_sampling/__init__.py`
- [ ] `affinity_transformer/dataset/pairs.py` 🔧 +约40行（`build_pairs`新增`noise_aware_multiscale`分支）
  - 看什么：`antigen_key`列缺失时的报错检查是在`_validate_pair_sampling`之前还是之后——必须在之前，否则一个同时缺`antigen_key`又传了别的非法参数的调用，会先报出无关的校验错误，把真正缺列这个更根本的问题盖住。
  - 命令：`git diff <BASELINE> -- affinity_transformer/dataset/pairs.py`
- [ ] `affinity_transformer/dataset/pair_sampling/validation.py` 🔧 +约25行（5个`noise_aware_*`参数校验+1个新策略名）
  - 命令：`git diff <BASELINE> -- affinity_transformer/dataset/pair_sampling/validation.py`
- [ ] `affinity_transformer/config.py` 🔧 +约45行（6个`noise_aware_*`字段）
  - 看什么：跟Layer 0说的一样的"三处必须一致"检查——dataclass默认值、`_build_data_config`里的`section.get(key, default)`、`_validate_pair_sampling`的校验范围，这次新增的6个字段也要过一遍这三处。
  - 命令：`git diff <BASELINE> -- affinity_transformer/config.py`
- [ ] `affinity_transformer/training/loaders.py` 🔧 +6行（`_build_pairs`透传6个新参数）
  - 命令：`git diff <BASELINE> -- affinity_transformer/training/loaders.py`
- [ ] `configs/v065/v065_{concat,deep4,deep8,deep16}_noise_aware_multiscale.yaml` 🆕 4个新文件
  - 看什么：跟Layer 6一样，四个文件之间`diff`一下，确认只有`interaction.kind`/`num_layers`不同，`data:`整段（尤其6个`noise_aware_*`字段）四个文件必须完全一致。
- [ ] `scripts/slurm/submit_noise_aware_multiscale.sh` 🆕 + `submit_v065_training_chain.sh`、`submit_randomized_bst_no_redundancy.sh` 🔧（这两个把串行改成了并行）
  - 看什么：三个脚本里`concat`/`deep4`/`deep8`/`deep16`四个`submit_training`调用的`dependency`参数是不是都改成了`${cache}`，不要有漏改成`${concat}`/`${deep4}`/`${deep8}`的（漏改一处就会悄悄退回串行，不报错，只是跑得比预期慢）。
  - 命令：`git diff <BASELINE> -- scripts/slurm/submit_noise_aware_multiscale.sh scripts/slurm/submit_v065_training_chain.sh scripts/slurm/submit_randomized_bst_no_redundancy.sh`
- [ ] `tests/test_noise_aware_multiscale.py` 🆕（规格§13的9个测试） + `tests/test_dataset.py` 🔧（`build_pairs`接入相关的3个测试）
  - 看什么：带着"这个测试如果把`_build_tau_separated_anchors`改回单链聚类规则，会不会失败"去读密集链不坍缩那个测试——如果改回去还能通过，说明这个测试没真的锁住这个文件存在的核心理由。
  - 命令：`git diff <BASELINE> -- tests/test_noise_aware_multiscale.py tests/test_dataset.py`
- [ ] `docs/experiments/tau_registry.md` 🆕、`docs/qa_progress.md` 🔧（这两份是文档，不影响代码行为，质检优先级最低，确认内容跟代码对得上即可）

命令（这一整层一次性看）：`git diff <BASELINE> -- affinity_transformer/dataset/pair_sampling/ affinity_transformer/dataset/pairs.py affinity_transformer/config.py affinity_transformer/training/loaders.py configs/v065/*noise_aware_multiscale.yaml scripts/slurm/submit_noise_aware_multiscale.sh tests/test_noise_aware_multiscale.py`

这一批目前**也没有接入任何已经在跑的v065默认配置**（只有新增的`*_noise_aware_multiscale.yaml`四个文件用它，已有的`*_ranknet.yaml`/`*_randomized_bst_no_redundancy.yaml`都没改`pair_sample_strategy`），跟Layer 8一样是新增、未替换已有路径，质检通过后再考虑要不要提交+上机跑。

---

## 附录A：`scripts/`目录里每个脚本是干什么的，属于哪一环

`scripts/prepare/binding/<study>/<table>/`下有四十多个数据集目录，每个目录都是`convert.py`+`test.py`+`prepare.sh`这三个文件的固定模式，逐一展开没有意义——下面只解释这个模式一次，不重复列每个数据集。

### 质检（数据准备与质量校验，发生在任何训练之前）

- **`scripts/prepare/binding/<study>/<table>/convert.py`**（×40+，固定模式）：把某一个原始数据集自己的raw CSV转换成标准表格式，只在这一层处理"这个数据集特有的怪字段"——单位换算、列名映射、抗体链拆分等。
- **`scripts/prepare/binding/<study>/<table>/test.py`**（×40+，固定模式）：对应`convert.py`的单元测试，检查转换出来的标准表是否符合schema、数值是否合理。
- **`scripts/prepare/binding/<study>/<table>/prepare.sh`**（×40+，固定模式）：跑某一个数据集`convert.py`的shell入口，可能还包含下载原始数据的步骤。
- **`scripts/prepare/binding/prepare_all.sh`**：批量调用上面所有数据集的`prepare.sh`，一次性把全部原始数据集转换完。
- **`scripts/prepare/binding/merge_records.py`**：把全部数据集各自的`records.parquet`合并成一张`processed/binding/all_records.parquet`。
- **`scripts/prepare/binding/gen_manifest.py`**：生成`manifest.csv`，记录"有哪些数据集、各自的状态"，相当于一份数据清单索引。
- **`scripts/prepare/binding/patch_seq_nullcheck.sh`**：事后补丁脚本，专门检查序列字段是不是有空值漏网。
- **`scripts/prepare/validate_processed_table.py`**：质检工具本身——检查任意一张标准表是否符合`affinity_transformer.dataset.schema.REQUIRED_COLUMNS`，是`docs/training_flow.md`里提到的"质检"命令对应的脚本。
- **`scripts/data/filter_records.py`**：对合并后的全量表应用`keep_for_training`过滤规则（调用`affinity_transformer.record_filter`）。
- **`scripts/data/inspect_records.py`**：生成数据质量摘要报告（JSON/统计量），人工质检时用来快速看一眼数据分布是否符合预期。
- **`scripts/data/build_splits.py`**：从过滤后的全量表切出固定的train/valid/test（调用`affinity_transformer.dataset.load_records`+`record_filter`）。
- **`scripts/analysis/group_size_stats.py`**（本轮新增）：纯诊断脚本，统计某个split里每个`group_id`的记录数/候选对数分布，给pair-sampling参数选择提供依据——严格说是"训练前的参数质检"，不是数据本身的质检。
- **`scripts/slurm/g00_qc_and_splits.sbatch`** + **`scripts/runs/g00_qc_and_splits.sh`**：把"质检+切分"这一段在SLURM上跑起来的入口（`submit_v065_training_chain.sh`链条里的`g00`那一步）。

### 训练

- **`train.py`**（仓库根目录）：训练的唯一入口，读取一个yaml配置，分发到对应的训练流程，详见附录B。
- **`scripts/slurm/setup_affitest_env.sh`/`.sbatch`**：创建/初始化`affitest`这个conda环境本身（装torch等依赖），是所有其他训练脚本的前提。
- **`scripts/slurm/download_v065_models_login.sh`、`download_esm2_login.sh`**：在登录节点（有外网）预先下载预训练模型权重——计算节点没有外网，必须提前下好。
- **`scripts/slurm/check_v065_models.sbatch`**：训练前确认预训练模型版本/权重文件齐全且匹配配置里写的revision。
- **`scripts/slurm/warmup_esm2_cache.sbatch`**：预热ESM2相关缓存（大概率是HuggingFace缓存），减少正式训练时第一次访问的延迟。
- **`scripts/embeddings/build_v065_cache.py`** + **`scripts/slurm/build_v065_embedding_cache.sbatch`**：把抗体/抗原序列跑一遍冻结的预训练编码器，写成`processed/embeddings/v065/...`下的分片embedding缓存——这一步跑完之后，`frozen_cached`模式的训练才有缓存可读。
- **`scripts/slurm/run_config.sbatch`**：单个配置的训练入口，本质是给`train.py --config ... --output-dir ...`套一层SLURM资源声明（GPU、显存、时间上限）。
- **`scripts/slurm/run_group.sbatch`**：通用的"跑一组训练"入口，实际执行的脚本由`GROUP_SCRIPT`环境变量指定（默认指向`g01_core_ablation.sh`）。
- **`scripts/runs/g01_core_ablation.sh`、`g02_label_source_ablation.sh`、`g03_pair_sampling_ablation.sh`、`g04_antigen_subset_ablation.sh`**：四组具名的消融实验编排脚本，各自调用`run_many.py`跑一批固定的`configs/experiments/g0X_*.yaml`，再调用`collect_results.py`汇总。
- **`scripts/experiments/run_many.py`**：依次对一串配置文件执行`subprocess.run([..., "train.py", "--config", cfg, ...])`，是"跑一组配置"实际发起子进程的地方。
- **`scripts/slurm/submit_g00_g01_chain.sh`**：把质检(g00)和g01消融实验首尾接起来提交。
- **`scripts/slurm/submit_v065_training_chain.sh`**：本轮一直在用的v065专属链条（smoke→models→g00→cache→concat→deep4→deep8→deep16）。

### 验证 / 测试

这里的"验证"（验证集上的指标）和"测试"（测试集上的指标）在这个项目里**不是独立脚本**，是`train.py`训练流程内部自动做的事（`Trainer.evaluate()`跑valid，`training/evaluation.py`的`write_split_evaluation`跑test，见附录B）。`scripts/`下能称为"测试"的，是软件工程意义上的单元测试：

- **`scripts/slurm/smoke_test.sbatch`**：跑`python -m pytest -q`，是代码层面的"冒烟测试"——在花真正的GPU时间训练之前，先确认代码本身没有明显错误。本轮新加的`balanced_tree`/`randomized_bst`测试就是靠这个入口在集群上被执行到的。
- **`tests/`目录**（不在`scripts/`下，但与之配套）：实际的pytest测试代码，`conftest.py`提供共享fixture（`toy_records`、假tokenizer/encoder），各`test_*.py`针对`affinity_transformer`各模块写单元测试。

### 使用（训练完成后，拿模型做推理）

- **`predict.py`**（仓库根目录）：命令行推理入口。
- **`affinity_transformer/user_entry.py`**：库形式的推理入口（`AffinityPredictor`），给定抗原序列+候选抗体序列，直接返回打分排序——不要求调用方自己拼`AffinityRanker`/tokenizer/checkpoint，这些都被封装在`AffinityPredictor`内部。

### 结果汇总（跨训练与使用之间）

- **`scripts/experiments/collect_results.py`**：扫描一组run目录下的`metrics.json`，汇总成一张CSV报告——是消融实验"训练完了之后看结果"这一步的工具，介于训练和人工分析之间。

---

## 附录B：`affinity_transformer`在质检-训练-验证-测试-使用中的调用链

这里只画"谁调用谁"，不重复贴代码——配合附录A，看到一个脚本名，就能在这里查到它具体往下调了哪些模块/函数，反过来也可以用这个表去对照"接口契约"：上一层传给下一层的参数，类型和含义是否真的一致。

### 质检阶段

```text
raw csv（每个数据集自己的格式）
  -> scripts/prepare/binding/<study>/<table>/convert.py
  -> processed/binding/<study>/<table>/records.parquet
  -> scripts/prepare/validate_processed_table.py
       -> affinity_transformer.dataset.schema.REQUIRED_COLUMNS（校验列是否齐全）
  -> scripts/prepare/binding/merge_records.py（纯pandas拼接，不导入affinity_transformer）
  -> processed/binding/all_records.parquet
  -> scripts/data/filter_records.py
       -> affinity_transformer.record_filter.build_record_filter_config / 应用过滤规则
  -> scripts/data/build_splits.py
       -> affinity_transformer.dataset.load_records
       -> affinity_transformer.record_filter（同上）
  -> processed/binding/splits/<split_name>/{train,valid,test}.parquet
```

embedding缓存这一段是独立的一条支线，依赖上面切分好的parquet，但不依赖`train.py`：

```text
processed/binding/splits/.../{train,valid,test}.parquet
  -> scripts/embeddings/build_v065_cache.py
       -> affinity_transformer.embeddings.{extractors, huggingface, pipeline}（跑预训练编码器前向）
       -> affinity_transformer.embeddings.store（写分片embedding + manifest）
  -> processed/embeddings/v065/<encoder>/<revision>/
```

### 训练阶段（核心调用链，本轮改动最密集的地方都在这条线上）

```text
train.py: main()
  -> affinity_transformer.config.load_config(yaml路径)
       -> 返回 Config（DataConfig / ModelConfig / TrainConfig / CrossValidationConfig）
  -> train.py: run_training(config_path, config, output_dir)
       根据 config.model.antibody_encoder.mode 选 runner：
         frozen_cached -> affinity_transformer.training.cached.run_cached_ranknet
         其他          -> affinity_transformer.training.online.run_online_training
       根据 config.cross_validation.enabled 决定走哪条：
         True  -> affinity_transformer.training.cross_validation.run_group_kfold_cross_validation
                    -> affinity_transformer.splits.build_group_kfolds
                    -> 对每一折分别调用上面选出的 runner（test集不参与）
         False -> affinity_transformer.training.data.resolve_data_paths(config)
                    -> 直接调用 runner 一次（单一固定切分）

  以 run_cached_ranknet 为例，runner内部依次调用：
    -> affinity_transformer.training.data.load_trainable_records          （读train/valid/test三张表）
    -> affinity_transformer.training.data.collect_required_embedding_hashes
    -> affinity_transformer.embeddings.validation.validate_embedding_cache  （antibody、antigen各一次，确认cache跟config里的encoder/revision匹配）
    -> affinity_transformer.model.factory.build_ranker(config.model, ...)
         -> affinity_transformer.model.{embedding_ranker, attention, blocks, interaction, heads, pooling, projections}
         -> 返回 AffinityRanker(nn.Module)
    -> affinity_transformer.embeddings.store.ShardedEmbeddingStore         （antibody、antigen各一个）
    -> affinity_transformer.training.loaders.build_cached_train_loader
         -> affinity_transformer.training.loaders._build_pairs
              -> affinity_transformer.dataset.pairs.build_pairs
                   -> affinity_transformer.dataset.pair_sampling.{common, blocks, large_group, two_label, labels, tree, randomized_tree, validation}
         -> affinity_transformer.dataset.datasets.PairwiseAffinityDataset
         -> affinity_transformer.training.samplers.GroupShuffleSampler
         -> affinity_transformer.embeddings.collate.collate_pair_embedding_batch（DataLoader的collate_fn）
    -> affinity_transformer.training.loaders.build_cached_rank_loader      （valid用，AffinityRecordDataset + collate_embedding_batch）
    -> affinity_transformer.training.loaders.compute_group_pair_weights    （本轮加的，算group_weights）
    -> affinity_transformer.trainer.Trainer(model, config, train_loader, valid_loader, ..., group_weights=...)
    -> trainer.fit()
         每个epoch:
           -> trainer._run_train_epoch()
                -> model.forward(batch.left) / model.forward(batch.right)
                -> affinity_transformer.model.losses.ranknet_loss(..., weight=...)
                -> optimizer.step()
           -> trainer.evaluate(valid_loader)            ←“验证”实际发生在这里，不是独立脚本
                -> affinity_transformer.metrics.compute_group_spearman
                -> affinity_transformer.metrics.summarize_group_spearman
           -> 写 checkpoint_latest.pt / valid_group_metrics_epoch{N}.csv
    -> trainer.save_checkpoint(...)
    -> affinity_transformer.training.evaluation.predict_cached_records      ←“测试”（test split）实际发生在这里
    -> affinity_transformer.training.evaluation.write_split_evaluation
    -> affinity_transformer.training.artifacts.{write_history, write_metrics, write_resource_metrics, write_run_log, copy_config, write_embedding_metadata_refs}
```

**这条链上最该核对契约的几个接口**（出问题大概率出在这几处，而不是模块内部逻辑）：

- `config.py`的`DataConfig`/`ModelConfig` ↔ `training/loaders.py`/`training/cached.py`怎么读这些字段——字段名、默认值、是否允许`None`，三处必须一致（附录前面Layer 0已经提过）。
- `dataset.pairs.build_pairs`返回的`PAIR_COLUMNS`列 ↔ `dataset.datasets.PairwiseAffinityDataset.__getitem__`怎么按列名取值——新加一种`pair_sample_strategy`，只要返回的列名、`y_ij`的取值约定（1.0/0.0，相等时跳过）保持不变，下游就不需要改。
- `training/loaders.py`里`_build_pairs`和`compute_group_pair_weights`各自调用一次`build_pairs`——两次调用的参数（尤其`seed`）必须完全一致，否则权重和真实喂给模型的pairs集合就对不上（前面质检清单里点过这个）。
- `Trainer.__init__`的`group_weights`参数 ↔ `_run_train_epoch`里`_batch_group_weights`怎么用它——`None`时必须完全退化成无权重行为，这条退化路径目前没有专门的测试。

### 软件测试阶段（独立于上面"训练内部的验证/测试"，是代码层面的）

```text
scripts/slurm/smoke_test.sbatch
  -> pytest -q
       -> tests/conftest.py（toy_records、FakeTokenizer、FakeEncoder等共享fixture）
       -> tests/test_*.py（每个文件直接 import 对应的 affinity_transformer 模块，绕过 train.py / CLI，
                            直接对公开函数喂合成数据断言行为——跟上面训练阶段的调用链是平行的两条路，
                            不会互相调用）
```

### 使用阶段

```text
已训练好的 checkpoint + 对应的训练用 config
  -> predict.py  或  affinity_transformer.user_entry.AffinityPredictor
       -> affinity_transformer.config.load_config        （复用训练时的同一份config，保证编码器/结构一致）
       -> affinity_transformer.trainer.build_model_and_tokenizers
       -> affinity_transformer.model.AffinityRanker（加载权重）
       -> affinity_transformer.dataloader.collate_rank_batch
       -> model.forward -> 打分 -> 排序 -> 输出 OUTPUT_COLUMNS（query_id, antibody_id, score, rank, model_name）
```

注意这条链走的是`affinity_transformer.trainer.build_model_and_tokenizers`（在线/显式token路径），跟训练阶段`frozen_cached`模式那条线**不是同一个模型构造入口**——这是故意的：推理时要能对任意没见过的候选序列直接打分，不能要求它们提前进过embedding缓存流水线。这一点本身也是个值得在质检里专门确认的契约：训练用`frozen_cached`、推理用`build_model_and_tokenizers`，两条路径构造出来的`AffinityRanker`权重结构必须完全对得上，否则load checkpoint时形状不匹配。
