# affinity_model 代码质检文档

本文档用于人工质检 `FLAb/affinity_model` 中的通用亲和力排序模型代码。当前版本已经从旧的 `per-benchmark head` 重构为 **跨数据集共享参数的 global model**。

## 一、总体设计

### 1.1 模型目标

初赛评分关注预测排序与真实亲和力排序的一致性，即 Spearman correlation。因此模型输出不是绝对 Kd，而是一个标量 score：

```text
score 越大 => 模型认为抗体亲和力越强
```

### 1.2 为什么不再每个数据集一个 head

旧实现中，每个 benchmark 单独切分 `80/10/10`，并从零训练一个独立 MLP head。这会导致：

- 同一 benchmark 的标签同时参与训练和测试，结果偏乐观；
- 模型依赖数据集 ID，无法自然处理官方盲测；
- Kd、-logKd、IC50、EC50 等指标容易被混用；
- 很小的数据集切出 2-3 条测试样本，Spearman 统计意义不足。

当前实现改为：

```text
所有合格 Kd 数据
  -> 每个 CSV 保留为一个 compatible_group
  -> 只在组内构造 ranking pair
  -> 按 compatible_group 整组划分 train/val/test
  -> 一个共享 AffinityMLP head
  -> 按组计算 Spearman，再汇总
```

### 1.3 当前默认纳入的数据

默认只使用真实 Kd 类数据：

- 保留：`Kd`、`-logKd`、`SPR Kd`、`BLI Kd`、`Octet Kd` 等；
- 跳过：`predicted Kd`、`bind/no bind`、`binary`、`IC50`、`EC50`、`ADCC`、其它无法确认的 binding/enrichment 指标。

这是保守选择：先把“亲和力预测”这个任务做干净。

## 二、文件结构

```text
affinity_model/
  config.py       全局配置
  data_loader.py  数据读取、metadata 判断、标签方向统一
  embeddings.py   ESM2 embedding 提取与缓存
  dataset.py      PyTorch Dataset：pairwise / pointwise / scoring
  losses.py       MSE、Ranking Hinge、RankNet
  model.py        共享 MLP head
  trainer.py      全局训练、按组划分、按组 Spearman 评估
```

入口脚本：

```text
FLAb/train.py
```

## 三、config.py

### `Config`

集中保存超参数和路径。

关键字段：

```python
ESM_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
ESM_EMBEDDING_DIM = 1280
HIDDEN_DIM = 256
DROPOUT = 0.2
MAX_PAIRS_PER_GROUP = 10000
MIN_GROUP_SIZE = 5
ALLOWED_ASSAY_FAMILIES = {"kd"}
GROUP_COL = "compatible_group"
RANK_LABEL_COL = "label"
MSE_LABEL_COL = "label_z"
```

质检点：

- `ALLOWED_ASSAY_FAMILIES = {"kd"}` 保证默认不混入 IC50/EC50。
- `MIN_GROUP_SIZE = 5` 对应“至少 5 条数据才做可靠排序评估”的要求。
- `GROUP_COL = "compatible_group"` 是禁止跨 assay 构造 pair 的关键。
- `MSE_LABEL_COL = "label_z"` 保证 MSE 不直接回归原始 Kd。

外部库：

```python
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

`torch.cuda.is_available()` 用来自动判断是否有 GPU，有则训练和 embedding 提取走 CUDA。

## 四、data_loader.py

该模块负责把 FLAb 的 CSV 文件变成模型可用的数据表。

输出 DataFrame 的关键列：

```text
sequence            拼接后的输入序列
label_raw           原始测量值
label               方向统一后的标签，越大越强
label_z             组内 z-score，MSE 使用
label_rank          组内百分位 rank，可用于后续消融
compatible_group    可比较组，当前等于单个 CSV 数据集名
assay_family        当前默认只接受 kd
```

### `canonical_filename(path_or_name)`

逻辑：

1. 使用 `os.path.basename()` 取文件名；
2. 如果本地文件是 `.csv.zip`，去掉 `.zip`；
3. 返回 metadata 中能匹配的 `.csv` 文件名。

外部库用法：

```python
os.path.basename(path_or_name)
```

`os.path.basename` 是 Python 标准库函数，用于从路径中取最后一级文件名。

### `load_metadata(metadata_path)`

逻辑：

1. 用 `pd.read_csv()` 读取 `data/flab_metadata.csv`；
2. 遍历每行，把 `filename` 映射到整行 metadata；
3. 后续用 metadata 判断 assay 类型。

外部库用法：

```python
pd.read_csv(metadata_path, low_memory=False)
```

`low_memory=False` 让 pandas 在读取大 CSV 时减少分块类型推断带来的 dtype 混乱。

### `_read_csv_or_zip(filepath)`

逻辑：

1. 如果是 `.csv.zip`，用 `zipfile.ZipFile` 打开；
2. 在压缩包内找第一个 `.csv`；
3. 用 `pd.read_csv()` 读取；
4. 如果是普通 `.csv`，直接读取。

外部库用法：

```python
with zipfile.ZipFile(filepath) as z:
    csv_names = [n for n in z.namelist() if n.endswith(".csv")]
    with z.open(csv_names[0]) as f:
        df = pd.read_csv(f, low_memory=False)
```

`zipfile.ZipFile` 是标准库压缩包读取器；`z.open()` 返回类文件对象，可以直接传给 pandas。

### `classify_assay(name, metadata_row)`

逻辑：

1. 拼接文件名、`assay/units_raw`、`assay/units`、`key_words`；
2. 转小写；
3. 按关键字分类：
   - `predicted` -> `predicted_kd`
   - `bind/no bind` / `binary` -> `binary`
   - `adcc` -> `adcc`
   - `ic50` -> `ic50`
   - `ec50` -> `ec50`
   - 同时包含 `Kd` 与 `IC50/EC50` -> `mixed_affinity_functional`
   - `kd` -> `kd`
   - 其它 -> `other`

质检点：

- 这个函数是防止混用 Kd、IC50、EC50 的第一道门。
- 当前只有 `kd` 会通过 `cfg.ALLOWED_ASSAY_FAMILIES`。

### `choose_label_column(df, assay_family)`

逻辑：

1. 如果是 Kd 数据，先扫描列名包含 `kd` 的数值列；
2. 显式 `-log` / `neg log` / `neg_log` 列优先级最高；
3. `fitness` 是泛名列，优先级低于显式 log Kd 列，高于普通 Kd 列；
4. 排除 `kdis`、`ka`、`counts` 等非亲和力列；
5. 如果没有 Kd 候选列，再回退到数值化比例足够高的 `fitness`。

外部库用法：

```python
pd.to_numeric(series, errors="coerce")
```

`errors="coerce"` 会把不能转成数字的值变成 `NaN`，便于判断某列是不是可靠数值列。

### `normalize_label(df, label_col, assay_family, metadata_row)`

逻辑：

1. 把标签列转成数值，保存为 `label_raw`；
2. 先根据实际列名判断是否已经是 `-logKd` / `neg_log_kd`；
3. 只有当标签列是 `fitness` 这类泛名列时，才使用 metadata 辅助判断是否已经是 log 标签；
4. 如果是原始 Kd 而不是 `-logKd`，执行：

```python
label = -np.log10(raw_kd)
```

5. 如果已经是 `-logKd` / `neg log Kd`，直接使用；
6. 删除无穷值和缺失值；
7. 生成组内 `label_z`；
8. 生成组内百分位 `label_rank`。

外部库用法：

```python
np.log10(raw.astype(float))
out.replace([np.inf, -np.inf], np.nan)
out["label"].rank(method="average", pct=True)
```

- `np.log10` 用于把“越小越好”的 Kd 转成“越大越好”的分数；
- `replace([np.inf, -np.inf], np.nan)` 清理非法数值；
- `rank(pct=True)` 生成百分位秩，ties 用平均名次。

质检点：

- `label` 只保证组内排序方向正确；
- `label_z` 是 MSE 的训练目标；
- 不用 `label_raw` 直接做 MSE。

### `_standardize_sequences(df)`

逻辑：

1. 把别名列重命名成 `heavy` / `light`；
2. 必须存在 `heavy`；
3. 双链抗体拼接为：

```text
heavy + GGGGSGGGGSGGGGS + light
```

4. 单链 VHH 直接使用 `heavy`。

质检点：

- linker 来自 `cfg.LINKER`；
- 该拼接模拟 scFv 输入形式。

### `load_one_dataset(filepath, metadata)`

单文件主入口。

流程：

1. 匹配 metadata；
2. `classify_assay()` 判断 assay；
3. 如果不是允许的 assay family，跳过；
4. 读取 CSV；
5. 标准化序列；
6. 选择标签列；
7. 标签方向统一；
8. 过滤过大、过小、无排序差异的数据集；
9. 添加追踪列：
   - `dataset`
   - `source_file`
   - `assay_family`
   - `assay_units`
   - `compatible_group`

质检点：

- 当前 `compatible_group = name`，即每个 CSV 一个可比较组。
- 没有跨 CSV 自动合并。

### `load_all_datasets(data_dir, metadata_path)`

逻辑：

1. 读取 metadata；
2. 扫描 `data/binding` 下所有 `.csv` / `.csv.zip`；
3. 逐个调用 `load_one_dataset()`；
4. 返回 `dict[str, DataFrame]`。

## 五、embeddings.py

该模块基本沿用旧实现，用 ESM2 提取冻结 embedding。

### `get_esm_model()`

逻辑：

1. 第一次调用时加载 tokenizer 和模型；
2. 存到模块级变量 `_tokenizer` / `_model`；
3. 后续调用直接复用。

外部库用法：

```python
EsmTokenizer.from_pretrained(cfg.ESM_MODEL_NAME)
EsmModel.from_pretrained(cfg.ESM_MODEL_NAME).to(cfg.DEVICE)
_model.eval()
```

- `from_pretrained` 从 HuggingFace 模型名加载预训练权重；
- `.to(cfg.DEVICE)` 把模型放到 GPU 或 CPU；
- `.eval()` 关闭 dropout，保证 embedding 稳定。

### `embed_sequence(seq)`

逻辑：

1. tokenizer 把氨基酸序列转为 token ids；
2. ESM2 forward 得到 `last_hidden_state`；
3. 去掉首尾特殊 token；
4. 对氨基酸位置做 mean pooling；
5. 返回 1280 维 numpy array。

外部库用法：

```python
inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=cfg.MAX_SEQ_LEN)
with torch.no_grad():
    outputs = model(**inputs)
embedding = hidden[0, 1:-1, :].mean(dim=0)
```

- `return_tensors="pt"` 表示返回 PyTorch Tensor；
- `truncation=True` 防止过长序列导致显存爆；
- `torch.no_grad()` 表示不记录梯度，节省显存；
- `mean(dim=0)` 是对序列位置求平均。

### `get_or_compute_embedding(seq, cache_dir)`

逻辑：

1. 用 MD5 hash 生成缓存文件名；
2. 如果 `.npy` 存在，用 `np.load()` 读取；
3. 否则调用 ESM2 计算并 `np.save()`。

### `embed_all_datasets(datasets, cache_dir)`

逻辑：

1. 收集所有唯一 `sequence`；
2. 逐条读取或计算 embedding；
3. 把 embedding 映射回每个 DataFrame；
4. 把整个 `embedded_datasets` 用 pickle 存到磁盘。

外部库用法：

```python
pickle.dump(embedded, f)
pickle.load(f)
```

pickle 用于保存 Python dict/DataFrame 结构，方便 `--mode train` 直接加载。

## 六、dataset.py

### `PairwiseRankingDataset`

用于 `ranknet` 和 `hinge`。

核心逻辑：

```python
for group_name in sorted(set(self.groups)):
    group_indices = np.where(self.groups == group_name)[0]
    local_pairs = [
        (pos_idx, neg_idx)
        for i in group
        for j in group
        if label_i - label_j > min_label_diff
    ]
```

质检点：

- pair 只在同一个 `compatible_group` 内构造；
- 不存在跨 Kd/IC50、跨抗原、跨实验体系比较；
- 每组最多采样 `cfg.MAX_PAIRS_PER_GROUP` 个 pair。

外部库用法：

```python
rng = np.random.default_rng(seed)
rng.choice(len(local_pairs), size=max_pairs_per_group, replace=False)
```

`np.random.default_rng` 是 numpy 推荐的新随机数生成器；`choice(..., replace=False)` 表示无放回采样 pair。

### `PointwiseRegressionDataset`

用于 `mse`。

逻辑：

```python
self.embeddings = np.stack(df["embedding"].values).astype(np.float32)
self.targets = df[target_col].values.astype(np.float32)
```

质检点：

- `target_col` 默认是 `label_z`；
- MSE 不直接看原始 `label_raw`。

### `ScoringDataset`

用于验证/测试推理。

逻辑：

每次返回：

```text
(embedding, label)
```

真实的 `compatible_group` 保留在 trainer 的 DataFrame 中，用于按组计算 Spearman。

## 七、losses.py

### `MSELoss`

公式：

```text
L = mean((score - label_z)^2)
```

外部库用法：

```python
nn.MSELoss(reduction="mean")
```

PyTorch 内置均方误差，`reduction="mean"` 表示对 batch 平均。

### `PairwiseHingeLoss`

公式：

```text
L = max(0, margin - (score_pos - score_neg))
```

质检点：

- 输入 pair 已经保证 `label_pos > label_neg`；
- loss 只关心相对顺序。

外部库用法：

```python
torch.clamp(self.margin - diff, min=0.0)
```

`torch.clamp` 把负损失截断为 0，实现 hinge。

### `RankNetLoss`

公式：

```text
L = -log sigmoid(score_pos - score_neg)
  = softplus(score_neg - score_pos)
```

外部库用法：

```python
F.softplus(score_neg - score_pos)
```

`softplus(x) = log(1 + exp(x))`，比手写 `log(sigmoid())` 数值更稳定。

### `LOSS_REGISTRY`

```python
LOSS_REGISTRY = {
    "mse": MSELoss,
    "hinge": PairwiseHingeLoss,
    "ranknet": RankNetLoss,
}
```

入口脚本通过字符串选择损失函数。

## 八、model.py

### `AffinityMLP`

结构：

```text
1280 -> Linear -> GELU -> Dropout -> Linear -> 1
```

关键代码：

```python
self.net = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, 1),
)
```

外部库用法：

- `nn.Linear`：全连接层；
- `nn.GELU`：Transformer 常用平滑激活函数；
- `nn.Dropout`：训练时随机置零部分神经元，降低过拟合。

质检点：

- 这是一个共享 head；
- `trainer.py` 中每个 loss 只实例化一个模型；
- 不再按 dataset 创建多个 head。

### `_init_weights()`

逻辑：

对所有 Linear 层做 Xavier 初始化：

```python
nn.init.xavier_uniform_(m.weight)
nn.init.zeros_(m.bias)
```

Xavier 初始化用于让训练初期各层输出方差更稳定。

## 九、trainer.py

这是本次重构的核心模块。

### `flatten_datasets(embedded_datasets)`

逻辑：

1. 把 `dict[str, DataFrame]` 纵向拼接；
2. 检查必须列：
   - `embedding`
   - `label`
   - `label_z`
   - `compatible_group`
3. 返回总表。

质检点：

- 这里只拼接，不改标签、不合并 group。

### `split_by_group(df)`

逻辑：

1. 取所有 `compatible_group`；
2. 用 numpy 随机打乱组名；
3. 按组数切出 train/val/test；
4. 返回三张 DataFrame。

关键点：

```python
test_groups = set(group_names[:n_test])
val_groups = set(group_names[n_test:n_test + n_val])
train_groups = set(group_names[n_test + n_val:])
```

质检点：

- 同一个 compatible_group 不会同时出现在 train 和 test；
- 这是比随机行划分更严格的泛化测试。

外部库用法：

```python
rng = np.random.default_rng(cfg.SEED)
rng.shuffle(group_names)
```

固定 seed 保证划分可复现。

### `_predict_scores(model, df)`

逻辑：

1. 构造 `ScoringDataset`；
2. 用 `DataLoader` 分 batch 推理；
3. `torch.no_grad()` 关闭梯度；
4. 返回与 df 行顺序一致的预测分数。

外部库用法：

```python
DataLoader(dataset, batch_size=cfg.EVAL_BATCH_SIZE, shuffle=False)
with torch.no_grad():
    pred = model(emb.to(cfg.DEVICE)).squeeze(-1)
```

`shuffle=False` 很关键，否则预测分数无法按原行顺序写回。

### `evaluate_by_group(model, df, split_name)`

逻辑：

1. 对所有样本推理；
2. 按 `compatible_group` 分组；
3. 每组计算 Spearman；
4. 汇总 mean / median / weighted mean。

外部库用法：

```python
corr, p_value = spearmanr(prediction, label)
np.average(detail_df["spearman"], weights=detail_df["n"])
```

- `spearmanr` 来自 scipy，用于计算排序相关；
- `np.average(..., weights=n)` 用样本数做加权均值。

质检点：

- 小于 `cfg.MIN_GROUP_SIZE` 的组不参与 Spearman；
- label 无变化的组不参与 Spearman；
- 评估仍然在组内完成，不跨实验体系算一个总 Spearman。

### `_build_train_loader(train_df, loss_name)`

逻辑：

- `mse`：
  - 使用 `PointwiseRegressionDataset`；
  - loss 是 `MSELoss`。

- `hinge` / `ranknet`：
  - 使用 `PairwiseRankingDataset`；
  - loss 是对应 pairwise loss。

质检点：

- 这是 MSE 和 ranking loss 分流的地方；
- MSE 不会错误地走 pairwise dataset。

### `train_global_model(embedded_datasets, output_dir, loss_name)`

单个 loss 的完整训练流程。

流程：

```text
flatten_datasets
  -> split_by_group
  -> build train loader
  -> model = AffinityMLP()
  -> AdamW optimizer
  -> CosineAnnealingLR scheduler
  -> epoch loop
  -> val group Spearman 选 best checkpoint
  -> test group Spearman
  -> 保存 global_{loss}.pt 和 by_group.csv
```

外部库用法：

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
torch.save({...}, model_path)
```

- `AdamW`：带 decoupled weight decay 的 Adam 优化器；
- `CosineAnnealingLR`：余弦退火学习率；
- `clip_grad_norm_`：限制梯度范数，防止训练不稳定；
- `torch.save`：保存模型参数、超参和数据划分。

质检点：

- 每个 loss 只创建一个 `AffinityMLP`；
- 最优 checkpoint 由验证集组内 Spearman 均值决定；
- 输出模型名是 `global_ranknet.pt` / `global_hinge.pt` / `global_mse.pt`。

### `run_global_training(embedded_datasets, output_dir, loss_names)`

逻辑：

依次调用 `train_global_model()`，保存：

```text
summary_global_losses.csv
global_{loss}_val_by_group.csv
global_{loss}_test_by_group.csv
global_{loss}.pt
```

### `run_all_benchmarks = run_global_training`

这是向后兼容旧入口名。语义已经变成 global training，不再是旧的 per-benchmark training。

## 十、FLAb/train.py

入口脚本支持：

```bash
python train.py --mode embed
python train.py --mode train
python train.py --mode all
```

### `set_seed(seed)`

逻辑：

固定 PyTorch、CUDA、numpy 的随机种子。

外部库用法：

```python
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

这样做可提升实验可复现性，但可能略微降低 cuDNN 速度。

### `main()`

逻辑：

1. 解析命令行参数；
2. `embed/all` 模式加载数据并提取 embedding；
3. `train/all` 模式加载缓存并训练全局模型；
4. 打印每个 loss 的验证和测试 Spearman。

## 十一、建议质检清单

### 数据合规性

- 检查日志中是否有 `assay_family=ic50/ec50/binary/predicted_kd` 被跳过。
- 检查每个保留数据集是否都输出 `assay=kd`。
- 检查 `compatible_group` 是否等于单个 CSV 名称。
- 检查没有调用旧的 `merge_small_datasets`。

### 标签方向

- 原始 Kd 文件应显示 `transform=neg_log10_raw_kd`。
- 已经是 `-logKd` 的文件应显示 `transform=as_is_higher_is_better`。
- MSE 训练目标应来自 `label_z`。

### 模型结构

- 每个 loss 只有一个 `global_{loss}.pt`。
- `trainer.py` 中没有对每个 dataset 创建模型的循环。
- `AffinityMLP` 不接收 dataset id，也没有 per-dataset 参数。

### 训练/评估

- train/val/test 是按 group 切分，而不是随机行切分。
- Spearman 是按组计算，再求均值/中位数/加权均值。
- 少于 5 条或标签无变化的组不会进入 Spearman 汇总。

## 十二、当前实现的边界

- 现在只训练抗体序列 embedding，没有显式加入抗原序列 embedding；
- IC50/EC50 等功能性读数被跳过，不作为亲和力监督；
- 组划分是随机 group split，还可以进一步扩展成 antigen-held-out split；
- 大数据集超过 `MAX_DATASET_SIZE` 会跳过，避免 embedding 计算失控。

这些边界是有意保守处理，优先保证初赛亲和力排序模型的训练语义干净。
