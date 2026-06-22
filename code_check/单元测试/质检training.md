# training/

## training/samplers.py


训练数据是按 group_id（一个抗原对应一批候选抗体）组织的 pair。如果直接用 DataLoader(shuffle=True) 做全局随机打乱，相邻两条 pair 很可能来自完全不同的 group_id。此外，每条记录的 antibody/antigen embedding 是从分片缓存（ShardedEmbeddingStore）里读的，全局乱序意味着每一步都要在不同的缓存分片之间跳着读，命中率差、I/O开销大这也是为什么之前 embeddings/store.py 那次改动要去掉LRU、改用mmap——本质是同一个问题的另一侧


`class GroupShuffleSampler(Sampler[int]):`的做法（`__iter__`）：
1. 先用带种子的 torch.Generator 把所有 group_id 的顺序整体打乱（group_order）。
2. 对每个 group 内部，再把该组里的 pair 行顺序单独打乱（within_group）。
3. 按"组打乱后的顺序"逐组吐出索引，组内部也是打乱的，但**同一组的所有** **pair** **一定连续吐出**。
```python
class GroupShuffleSampler(Sampler[int]):

    """Shuffle group blocks and their pair rows on every iteration.
    A fully global pair shuffle destroys locality in the sharded embedding cache.  Keeping each group contiguous retains repeated antigen/antibody reads while randomizing both group order and within-group pair order on every epoch.
    """
```

## training/artifact.py


包含训练记录的写入函数

1. write_history：保存训练过程的历史记录

- 保存内容：每个 Epoch（训练轮次）的各项指标（如 train_loss, valid_macro_spearman 等）。
- 文件格式：CSV 文件。
- 作用：记录模型在训练过程中的动态变化。代码中特别处理了列对齐的问题，确保即使某些 Epoch 缺少某些指标（比如验证集指标），CSV 的表头也能包含所有出现过的键，缺失的地方用空字符串填充。

2. write_metrics：保存最终的评估指标

- 保存内容：一个包含浮点数指标的字典（通常是训练结束后的最终测试集指标或最佳验证指标）。
- 文件格式：JSON 文件。
- 作用：保存实验的最终结果。代码中包含了一个 clean 函数，专门用来将 NaN（非数字，通常表示无效或未计算的指标）转换为 null，以保证 JSON 格式的合法性。

3. copy_config：保存超参数配置

- 保存内容：训练使用的配置文件（统一重命名为 config.yaml）。
- 文件格式：YAML 文件。
- 作用：保存实验的“配方”。确保未来任何时候回看这次实验，都能确切知道当时使用了哪些模型结构和训练超参数，保证实验的可复现性。

4. write_embedding_metadata_refs：保存嵌入缓存的元数据引用

- 保存内容：抗体（Antibody）和抗原（Antigen）的嵌入缓存描述符（CacheDescriptor）。包括缓存目录路径、编码器名称、嵌入维度、数据类型、覆盖率、序列长度统计等。
- 文件格式：YAML 文件。
- 作用：保存数据预处理和特征工程的状态。由于模型使用了预计算的嵌入缓存，保存这些元数据可以确保后续推理或继续训练时，能准确找到并验证对应的嵌入数据，防止数据版本不一致导致的错误。

5. write_resource_metrics：保存系统资源与训练效率指标

- 保存内容：训练过程中的硬件和性能数据。包括：融合方式、可训练参数量、优化器步数、总耗时、吞吐量（每秒处理的样本数）、峰值 GPU 显存占用、注意力机制的内存上限等。
- 文件格式：JSON 文件。
- 作用：用于性能调优和成本核算。帮助研究人员了解当前模型配置对硬件的压力，评估训练效率，以及排查是否存在显存溢出（OOM）的风险。

6. write_run_log：保存运行日志摘要

- 保存内容：本次运行的基本上下文信息。包括运行模式、融合方式、配置文件路径、训练/验证/测试集的数据路径、以及训练样本总数。
- 文件格式：纯文本（TXT）文件。
- 作用：提供一个人类可读的快速概览。当在服务器上跑了几十个实验时，不需要打开复杂的 JSON 或 CSV，只需看一眼这个文本文件，就能知道这次运行用了什么数据、在什么模式下跑的。

7.` _length_payload`：辅助函数（不直接保存文件）

- 保存内容：提取 CacheDescriptor 中与序列长度相关的统计信息（如序列长度、嵌入长度、截断数量、截断率）。
- 作用：这是一个内部辅助函数，被 write_resource_metrics 调用，用于将抗体和抗原的长度统计信息格式化后，嵌入到资源指标 JSON 文件中。



## training/data.py

```python
def resolve_data_paths(config: Config) -> tuple[Path, Path | None, Path | None]:

  """返回显式的拆分路径，如果配置选择自动拆分，则顺带完成拆分。"""

```

流程：

- 如果分割策略是none
  - 直接检查train valid test path是否存在，并返回这些path
- 否则同时检查config.data.all_records_path 与 config.data.split_dir是否同时存在
- 然后执行自动分割
  - 先加载所有records `load_records`
  - 然后执行records筛选
  - 然后将筛选后的filtered_records.parquet写入某个文件夹
  - 最后建立分割

```python
def load_trainable_records(path: Path) -> pd.DataFrame:
  """加载一个已处理的分割文件，并在可训练数据为空时拒绝数据"""
```
加载一个已处理的分割文件，并在可训练数据为空时拒绝数据
```python
def collect_required_embedding_hashes(
  record_tables: Iterable[pd.DataFrame],
) 
  """Collect unique antibody/antigen cache keys across configured splits."""
```

该模块旨在从全部数据表中筛选并整合一组唯一且经遴选的序列集合，以支持后续的嵌入向量（embedding）预提取任务。

注意，该模块调用了

## training/cached.py

该文件定义了高度工程化的基于提前计算的特征缓存的 RankNet 模型训练与评估入口函数`run_cached_ranknet`。它的核心作用是：在冻结抗体和抗原编码器的前提下，构建并训练一个用于成对排序（Pairwise Ranking）的交互模型（如 Concat 或 Deep-Cross-Attention），并在训练结束后自动执行验证集/测试集的评估，最终将模型、指标和运行日志完整地持久化到磁盘。

具体训练流程如下：
#### 1. 配置校验与前置检查

- **架构与目标校验**：强制检查模型交互类型（`interaction.kind`）必须是 `concat` 或 `deep_cross_attention`，且目标函数（`objective.name`）必须是 `pairwise_ranknet`。
- **编码器配置校验**：验证抗原编码器（`antigen_encoder`）配置是否存在（抗体编码器因上游配置约束，此处仅做缓存路径的断言检查）。

#### 2. 数据加载与缓存一致性校验

- **加载数据记录**：`load_trainable_records(train_path, config)`分别读取训练集、验证集和测试集的数据记录（Records）。
- **收集特征哈希**：`collect_required_embedding_hashes(records)`汇总所有数据集中需要使用的抗体和抗原序列的哈希值。
- **验证缓存完整性**：`validate_embedding_cache`通过哈希值校验抗体和抗原的预计算特征缓存（Embedding Cache）是否完整且合法，防止训练时因缺失特征而报错。

#### 3. 模型构建与数据管道（Data Pipeline）组装

- **构建排序模型**：`build_ranker()`基于配置和缓存描述符，初始化 RankNet 排序模型。
- **初始化特征存储**：创建分片特征存储（`ShardedEmbeddingStore`），用于在训练时高效读取缓存特征。
- **构建数据加载器**：`build_cached_train_loader()`,`build_cached_rank_loader()`将数据记录与特征存储结合，构建用于训练的 `train_loader` 和用于验证的 `valid_loader`。

#### 4. 初始化 Trainer 与准备训练

- **组装 Trainer**：将模型、配置、数据加载器、验证集元数据、输出目录、特征元数据哈希以及计算好的组权重（`group_weights`）打包进 `Trainer` 对象。
- **GPU 状态重置**：如果检测到可用 CUDA 设备，重置峰值显存统计并同步设备，确保资源监控准确。

#### 5. 执行核心训练循环（`trainer.fit()`）

- **计时与执行**：`time.perf_counter()`记录开始时间，调用 `fit()` 方法。
- **内部机制**：`fit()` 会遍历设定的 Epochs，每个 Epoch 包含前向传播、RankNet 损失计算、反向传播和优化器更新。如果配置了早停（Early Stopping），会在验证指标连续不提升时提前终止。
- **自动择优**：训练结束后，`fit()` 会自动将模型参数回滚到验证集表现最好的 Epoch。
- **计时结束**：`time.perf_counter()-start`记录训练总耗时，并再次同步 CUDA 设备。
#### 6. 模型与训练历史的持久化

- **保存最终模型**：`trainer.save_checkpoint()`将训练结束后的最优模型权重保存为 `checkpoint.pt`。
- **保存训练历史**：将每个 Epoch 的损失、验证指标和耗时写入 `history.csv`。

#### 7. 评估与指标计算

- **提取基础指标**：从训练历史中提取最后一个 Epoch 的指标、选中的 Epoch 编号以及最佳指标值。
- **验证集评估**：如果提供了验证集，使用最优模型进行预测，计算评估指标并写入文件（`write_split_evaluation`）。
- **测试集评估**：如果提供了测试集，同样进行预测、计算指标并写入文件。
- **保存指标**：将所有汇总的指标保存为 `metrics.json`。

#### 8. 环境快照与日志归档

- **复制配置文件**：将当前运行的配置文件复制到输出目录，确保实验可复现。
- **记录特征引用**：保存抗体和抗原特征缓存的元数据引用（`embedding_metadata_refs.yaml`）。
- **记录资源消耗**：保存显存占用、训练时长等资源指标（`resource_metrics.json`）。
- **生成运行日志**：记录本次运行的核心参数（如模式、融合方式、数据量等）到 `run.log`。

#### 9. 返回结果

- 最终返回一个包含所有关键评估指标的字典（`dict[str, float]`），供上游调用者使用。

```plaintext
# NOTE: 此处的验证集评估（训练后）与 Trainer.fit() 内部的验证评估有本质区别： # 1. Trainer.fit() 内部验证： # - 目的：监控训练过程、触发早停、选择最佳模型（epoch 级粒度） # - 特点：仅计算分组级指标（如 ndcg@5），不保存原始预测结果 # - 频率：每 epoch 执行一次（受 config 控制，但无法跳过） # # 2. 训练后显式验证评估： # - 目的：使用最终选定的模型（可能早停回滚到非末轮 epoch）生成**完整预测结果** # - 特点：保存每个样本的预测分数（用于错误分析/可视化），计算全量指标（可能包含更多 metric） # - 必要性：fit() 内部的验证结果仅用于训练调度，**不保证与最终模型预测一致** # # 因此，即使 Trainer 已在训练中验证多次，此处仍需重新预测验证集以确保： # (a) 指标基于最终部署模型（而非训练中途的最佳模型快照） # (b) 生成可复现的预测文件供人工审查 # # 若未来需优化高频验证成本，应在 Trainer 内部通过 `validation_interval` 配置控制训练中验证频率， # 而非删除训练后的最终评估（其目标场景不同）。

一些疑问，后面解决
```
## training/online.py

相较于上一版本，这一版则主要是为LoRA微调基座大模型而生，我们不再缓存embeddings，而是调整个数据链路的参数

#### 1. 模型与分词器构建

- **初始化组件**：直接根据模型配置（`config.model`）构建排序模型（`model`），以及抗体和抗原的分词器（`antibody_tokenizer`, `antigen_tokenizer`）。这里跳过了缓存校验，直接进入在线处理流程。

#### 2. 数据加载与在线管道组装

- **构建训练集管道**：读取训练集路径，结合分词器构建用于训练的 `train_loader`，同时返回训练集记录（`train_records`）。
- **构建验证集管道**：读取验证集路径，结合分词器构建用于验证的 `valid_loader`，同时返回验证集记录（`valid_records`）。

#### 3. 初始化 Trainer 与准备训练

- **组装 Trainer**：将模型、配置、数据加载器、验证集元数据、输出目录以及计算好的组权重（`group_weights`）打包进 `Trainer` 对象。（_注：这里没有传入 `embedding_metadata_hashes`，因为不涉及缓存_）。

#### 4. 执行核心训练循环（`trainer.fit()`）

- **执行训练**：调用 `fit()` 方法。与之前一样，内部会执行 Epoch 循环、RankNet 损失计算、参数更新，并支持早停机制。训练结束后，模型会自动回滚到验证集表现最好的 Epoch。

#### 5. 模型与训练历史的持久化

- **保存最终模型**：将训练结束后的最优模型权重保存为 `checkpoint.pt`。
- **保存训练历史**：将每个 Epoch 的损失、验证指标和耗时写入 `history.csv`。

#### 6. 评估与指标计算

- **提取基础指标**：从训练历史中提取最后一个 Epoch 的指标、选中的 Epoch 编号以及最佳指标值。
- **验证集评估**：如果存在验证集记录，使用最优模型和分词器进行在线预测，计算评估指标并写入文件。
- **测试集评估**：如果提供了测试集路径，动态构建测试集的数据加载器并获取记录。如果记录不为空，同样进行在线预测、计算指标并写入文件。
- **保存指标**：将所有汇总的指标保存为 `metrics.json`。

#### 7. 环境快照与日志归档

- **复制配置文件**：将当前运行的配置文件复制到输出目录，确保实验可复现。
- **生成运行日志**：记录本次运行的核心参数到 `run.log`。（_注：这里的 `mode` 参数被显式设置为 `None`，以区别于缓存模式的标识_）。

#### 8. 返回结果

- 最终返回一个包含所有关键评估指标的字典（`dict[str, float]`），供上游调用者使用。


## training/cross_validation.py

`run_group_kfold_cross_validation` 函数实现了基于组的 K 折交叉验证（Group K-Fold Cross-Validation），其核心目标是避免数据泄露并评估模型在未见组上的泛化能力。以下是关键点的深度解析：

#### 必要性
- 普通 K-Fold 的缺陷：若数据中存在**自然分组**（如同一患者的多次测量、同一家公司的股票数据），随机划分会导致同一组的样本分散在训练集和验证集，造成**数据泄露**（模型在验证时"见过"部分组内数据）。
- **Group K-Fold 的解决方案**：**确保同一组的所有样本始终在同一折中**，训练集和验证集**完全隔离组别**。

#### 实现步骤
1. 首先检查配置，明确是否需要K折交叉验证，如果不需要K折交叉验证、或者需要K折交叉验证但没提供train path 则Raise ValueError
2. 然后加载所有需要的记录，并调用`filter_trainable_records([pd.co](https://pd.co/)ncat(pool_tables, ignore_index=True))`筛选可用于训练的数据，同时检查是否有重复(`pool["record_id"].duplicated().any()`)
3. 调用`build_group_kfolds(pool, n_splits=cv.n_splits, seed=cv.seed)`，构建K折划分，返回的是一个数组，数组的元素是某一折的（编号，训练集、验证集）
4. 然后就是对每一折调用经典的训练流程，最后返回本折的训练评估指标。在交叉验证过程中，各折训练均采用独立的随机种子，以确保模型参数初始化的随机性与独立性，避免因种子重复导致模型初始化状态一致。

注释：目前的build k fold函数位于split.py文件，无法确认是否为四种不同的提问方式做适配