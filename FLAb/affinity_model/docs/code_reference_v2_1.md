# affinity_model v2.1

## 一、v2.1 当前实现目标

v2.1 是一个保守版本，不追求复杂架构，先把 v1 的输入表示和实验记录做清楚。

v1 已经保留的正确方向：

- 一个共享 `AffinityMLP`，不再为每个 dataset 单独训练任务头；
- `RankNet` / `Ranking Hinge Loss` 仍然学习相对排序；
- `MSE` 仍然作为 baseline，目标仍是组内 `label_z`；
- ranking pair 只在同一个 `compatible_group` 内构造；
- train/val/test 仍按 `compatible_group` 整组划分；
- Spearman 仍按组计算，再做汇总。

v2.1 改动：

- 默认输入从 v1 的单个 `embedding = ESM2(heavy + LINKER + light)` 改为 `concat(heavy_embedding, light_embedding)`；
- 缺少 light chain 的数据写入零向量 `light_embedding`，并记录 `has_light=False`；
- checkpoint 主指标从 `val_mean_spearman` 改为 `val_weighted_spearman`；
- 新增 split report，记录每个 `compatible_group` 被分到哪个 split；
- 新增 CLI 参数，方便做 split/checkpoint/min_label_diff 对照实验。

v2.1 明确不做：

- 不做 `(feature_a, feature_b, target)` 方向平衡；
- 不改变 `RankNetLoss` 和 `PairwiseHingeLoss` 的输入形式；
- 不加入 `|heavy-light|`、`heavy*light`、attention 或抗原模块；
- 不默认打碎大数据集；
- 不修改 MSE 的 `label_z` 定义。

因此，RankNet/Hinge 的训练 pair 仍然是：

```text
(feature_pos, feature_neg, label_pos, label_neg)
其中 label_pos > label_neg
```

## 二、v1 结果与 v2.1 要解决的问题

v1 已观察到的全局模型结果：

```text
loss      val_mean_spearman   test_mean_spearman   test_n_groups
mse              0.384793             0.204253               7
hinge           -0.049455             0.243372               7
ranknet          0.170512             0.275120               7
```

这个结果说明：

- RankNet 在这次 split 下 test mean 暂时最好；
- 但 `mean_spearman` 对每个 group 等权，小 group 的随机波动会被放大；
- 仅凭这一张表很难判断 val/test 是否被少数 group 主导；
- v1 的单个 scFv mean embedding 可能把 heavy/light 的链级信息混在一起。

v2.1 对应处理：

- 用 `weighted_spearman` 做默认 checkpoint；
- 输出 `global_{loss}_split_report.csv` 供人工检查；
- 用 heavy/light 朴素拼接做低风险输入改进；
- 保留 `scfv_mean` 作为 ablation，而不是删除 v1 输入。

## 三、config.py

### `Config`

自变量：

```text
无。Config 是配置类，不是普通函数。
```

输出：

```text
cfg: Config
```

含义：其它模块通过 `from .config import cfg` 读取同一份配置。

v2.1 新增或修改字段：

```python
MODEL_VERSION = "v2.1"
MODEL_FEATURE_MODE = "chain_concat"
MODEL_INPUT_DIM = ESM_EMBEDDING_DIM * 2
SPLIT_OBJECTIVE = "rows_balanced"
CHECKPOINT_METRIC = "val_weighted_spearman"
OUTPUT_DIR = "results/affinity_model_v2"
```

字段解释：

- `MODEL_FEATURE_MODE="chain_concat"`：默认输入是 `[heavy_embedding, light_embedding]`；
- `MODEL_INPUT_DIM=2560`：ESM2-650M 单链 embedding 是 1280，heavy/light 拼接后是 2560；
- `SPLIT_OBJECTIVE="rows_balanced"`：默认仍优先让 val/test 样本行数接近 8:1:1；
- `CHECKPOINT_METRIC="val_weighted_spearman"`：保存最佳模型时看验证集加权 Spearman；
- `OUTPUT_DIR` 改到 v2 目录，避免覆盖 v1 结果。

质检点：

- `MAX_PAIRS_PER_GROUP` 没有被 v2.1 网格搜索；
- `MIN_LABEL_DIFF` 仍默认 0.0，但可以从 CLI 改；
- `MSE_LABEL_COL` 仍是 `label_z`。

## 四、embeddings.py

### `_normalize_sequence_value(value)`

自变量：

```text
value: 任意 DataFrame 单元格值
```

可能是字符串、空值、`NaN`。

输出：

```text
str
```

功能：把序列值整理成稳定字符串。

实现：

```python
return "".join(str(value).split()).upper()
```

解释：

- `str(value)` 把单元格转成字符串；
- `.split()` 会按空格、换行、制表符切开；
- `"".join(...)` 把切开的片段重新拼起来，相当于去掉所有空白；
- `.upper()` 转成大写，避免同一序列因为大小写不同重复缓存。

### `_is_valid_sequence_value(seq)`

自变量：

```text
seq: str
```

输出：

```text
bool
```

功能：判断整理后的序列是不是有效存在。

实现：只过滤空字符串和明显缺失值：

```text
"nan", "none", "null", "na"
```

注意：这里没有严格过滤非标准氨基酸，因为 FLAb 中可能有 `X` 等未知残基。

### `_collect_unique_sequences(datasets, columns)`

自变量：

```text
datasets: dict[str, pd.DataFrame]
columns: list[str]
```

例子：

```python
columns = ["heavy", "light"]
```

输出：

```text
set[str]
```

功能：从所有数据集中收集唯一序列，避免 ESM2 重复计算。

实现逻辑：

1. 遍历每个 DataFrame；
2. 对指定列逐个取值；
3. 调用 `_normalize_sequence_value()` 清洗；
4. 调用 `_is_valid_sequence_value()` 过滤缺失；
5. 加入 `set` 去重。

### `embed_all_datasets(datasets, cache_dir)`

自变量：

```text
datasets: dict[str, pd.DataFrame]
cache_dir: str
```

`datasets` 是 `load_all_datasets()` 输出的标准化数据表集合。

`cache_dir` 是 ESM2 `.npy` 文件和 `embedded_datasets.pkl` 的保存目录。

输出：

```text
dict[str, pd.DataFrame]
```

功能：为训练数据生成 embedding cache。

v2.1 逻辑：

```text
如果 MODEL_FEATURE_MODE == "chain_concat":
  收集 heavy 和 light 的唯一序列
  分别计算 ESM2 embedding
  写回 heavy_embedding / light_embedding / has_light

如果 MODEL_FEATURE_MODE == "scfv_mean":
  沿用 v1
  收集 sequence 的唯一序列
  写回 embedding
```

关键实现：

```python
zero_embedding = np.zeros(cfg.ESM_EMBEDDING_DIM, dtype=np.float32)
df["heavy_embedding"] = heavy_seqs.map(seq_to_emb)
df["light_embedding"] = [
    seq_to_emb[seq] if has else zero_embedding
    for seq, has in zip(light_seqs, has_light)
]
```

外部库用法：

- `np.zeros(...)`：创建 VHH/缺 light 时使用的零向量；
- `Series.map(...)`：把序列字符串映射成对应 embedding；
- `pickle.dump(...)`：把整个 embedded datasets 存到磁盘，供 `--mode train` 读取。

质检点：

- 默认 v2.1 cache 必须有 `heavy_embedding` 和 `light_embedding`；
- 缺 light 不应报错，应写零向量；
- `has_light_fraction` 后续会进入 split report；
- 旧 v1 cache 只有 `embedding`，用默认 `chain_concat` 训练时应明确报错。

## 五、dataset.py

### `_stack_embedding_column(df, column)`

自变量：

```text
df: pd.DataFrame
column: str
```

输出：

```text
np.ndarray
```

功能：把 DataFrame 中一列 `np.ndarray` embedding 堆成二维矩阵。

关键实现：

```python
np.stack(df[column].values).astype(np.float32)
```

外部库用法：

- `np.stack(...)`：把很多个形状相同的一维向量堆成 `[N, D]` 矩阵；
- `astype(np.float32)`：保证 PyTorch 输入是 float32。

### `build_model_feature_matrix(df)`

自变量：

```text
df: pd.DataFrame
```

输出：

```text
np.ndarray
```

功能：根据当前 `cfg.MODEL_FEATURE_MODE` 生成 MLP 输入矩阵。

`chain_concat` 模式：

```python
heavy = _stack_embedding_column(df, "heavy_embedding")
light = _stack_embedding_column(df, "light_embedding")
return np.concatenate([heavy, light], axis=1)
```

输出形状：

```text
[n_rows, 2560]
```

`scfv_mean` 模式：

```python
return _stack_embedding_column(df, "embedding")
```

输出形状：

```text
[n_rows, 1280]
```

质检点：

- `chain_concat` 只做拼接；
- 没有 `absdiff`、`product`、attention；
- 缺少当前 feature mode 所需缓存列时会报明确错误。

### `PairwiseRankingDataset.__init__(...)`

自变量：

```text
df
label_col: str
group_col: str
max_pairs_per_group: int
min_label_diff: float
seed: int
```

输出：

```text
PairwiseRankingDataset 实例
```

功能：为 RankNet/Hinge 构造 pairwise 训练样本。

v2.1 实现变化：

```python
self.features = build_model_feature_matrix(df)
self.feature_dim = int(self.features.shape[1])
```

pair 构造仍然是 v1 原始逻辑：

```python
if group_labels[i] - group_labels[j] > min_label_diff:
    local_pairs.append((pos_idx, neg_idx))
```

含义：

```text
pos_idx 对应高亲和力样本
neg_idx 对应低亲和力样本
```

质检点：

- 只在同一个 `compatible_group` 内构造 pair；
- 不构造 `(neg, pos)` 反向 pair；
- 不返回 `target=0/1`；
- `losses.py` 的 RankNet/Hinge 接口不需要变化。

### `PairwiseRankingDataset.__getitem__(idx)`

自变量：

```text
idx: int
```

输出：

```text
(feature_pos, feature_neg, label_pos, label_neg)
```

功能：根据 pair 索引取出一对训练样本。

关键实现：

```python
pos_idx, neg_idx = self.pairs[idx]
return (
    torch.tensor(self.features[pos_idx]),
    torch.tensor(self.features[neg_idx]),
    torch.tensor(self.labels[pos_idx]),
    torch.tensor(self.labels[neg_idx]),
)
```

外部库用法：

- `torch.tensor(...)`：把 numpy 数组转成 PyTorch Tensor，供模型训练。

### `PointwiseRegressionDataset` / `ScoringDataset`

v2.1 改动很小：

- 原来直接读取 `df["embedding"]`；
- 现在统一调用 `build_model_feature_matrix(df)`；
- 其它逻辑不变。

## 六、model.py

### `AffinityMLP.__init__(input_dim=None, hidden_dim=cfg.HIDDEN_DIM, dropout=cfg.DROPOUT)`

自变量：

```text
input_dim: int | None
hidden_dim: int
dropout: float
```

输出：

```text
AffinityMLP 实例
```

功能：创建两层共享 MLP 亲和力打分头。

v2.1 改动：

```python
if input_dim is None:
    input_dim = cfg.MODEL_INPUT_DIM
```

原因：命令行参数会在模块 import 之后覆盖 `cfg.MODEL_FEATURE_MODE`，所以不能把输入维度写死在函数默认参数里。

模型结构仍然是：

```text
Linear(input_dim, hidden_dim)
GELU
Dropout
Linear(hidden_dim, 1)
```

质检点：

- 没有 per-dataset head；
- 没有 attention；
- 输出仍是单个标量 score。

## 七、trainer.py

### `flatten_datasets(embedded_datasets)`

自变量：

```text
embedded_datasets: dict[str, pd.DataFrame]
```

输出：

```text
pd.DataFrame
```

功能：把多个 embedded DataFrame 纵向拼成训练总表。

v2.1 实现：

- `chain_concat` 时要求存在 `heavy_embedding` 和 `light_embedding`；
- `scfv_mean` 时要求存在 `embedding`；
- 如果缓存列缺失，会提示重新运行 `--mode embed`。

质检点：

- 不合并 `compatible_group`；
- 不改变标签；
- 只做总表拼接和必要列检查。

### `_select_groups_near_target(group_sizes, target_rows, forbidden_groups, min_groups, target_groups=None)`

自变量：

```text
group_sizes: pd.Series
target_rows: int
forbidden_groups: set[str]
min_groups: int
target_groups: int | None
```

输出：

```text
set[str]
```

功能：从候选 group 中选出一批作为 val 或 test。

默认 `rows_balanced`：

```text
先让选中 group 的总样本行数接近 target_rows
```

可选 `groups_then_rows`：

```text
先让选中 group 数接近 target_groups
再让样本行数接近 target_rows
```

关键实现：

```python
states: dict[tuple[int, int], tuple[str, ...]] = {(0, 0): ()}
```

这里的 key 是：

```text
(当前总行数, 当前 group 数)
```

value 是：

```text
目前选择了哪些 group
```

质检点：

- `forbidden_groups` 不会被重复选择；
- group 仍然是不可拆分单位；
- 默认策略仍是 `rows_balanced`。

### `split_by_group(df)`

自变量：

```text
df: pd.DataFrame
```

输出：

```text
(train_df, val_df, test_df)
```

功能：按 `compatible_group` 整组划分训练、验证、测试。

v2.1 改动：

- 增加 `target_val_groups` / `target_test_groups`；
- 把 `cfg.SPLIT_OBJECTIVE` 传给 `_select_groups_near_target()` 使用；
- 打印每个 split 的行数、group 数和比例。

质检点：

- 同一个 group 不会同时进入 train/val/test；
- 如果训练集为空，直接报错；
- v2.1 没有把大 group 自动打碎。

### `_write_split_report(train_df, val_df, test_df, output_dir, loss_name)`

自变量：

```text
train_df: pd.DataFrame
val_df: pd.DataFrame
test_df: pd.DataFrame
output_dir: str
loss_name: str
```

输出：

```text
str
```

含义：写出的 CSV 文件路径。

文件路径：

```text
{output_dir}/global_{loss_name}_split_report.csv
```

CSV 字段：

```text
compatible_group
split
dataset
n_rows
n_unique_label
label_min
label_max
assay_family
assay_units
has_light_fraction
```

功能：给每个 loss 保存一份 split 质检表。

关键实现：

```python
for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    for group_name, group_df in split_df.groupby(cfg.GROUP_COL):
        ...
report.to_csv(path, index=False)
```

外部库用法：

- `DataFrame.groupby(...)`：按 compatible_group 聚合；
- `DataFrame.to_csv(...)`：保存质检表。

质检点：

- 每个 `compatible_group` 在 split report 中只应出现一次；
- `n_rows` 总和应等于训练总表行数；
- `has_light_fraction` 可以检查 VHH/缺 light 数据是否集中在某个 split。

### `_build_train_loader(train_df, loss_name)`

自变量：

```text
train_df: pd.DataFrame
loss_name: str
```

输出：

```text
(loader, loss_fn, feature_dim)
```

功能：根据 loss 类型创建 Dataset、DataLoader 和损失函数。

v2.1 逻辑：

```text
loss_name == "mse":
  PointwiseRegressionDataset
  MSELoss

loss_name in {"hinge", "ranknet"}:
  PairwiseRankingDataset
  PairwiseHingeLoss 或 RankNetLoss
```

关键点：

- pairwise 路径仍然使用 `(feature_pos, feature_neg, label_pos, label_neg)`；
- 没有 target；
- 没有正反方向平衡；
- `feature_dim` 从 Dataset 读出，用来初始化 MLP。

### `train_global_model(embedded_datasets, output_dir, loss_name)`

自变量：

```text
embedded_datasets: dict[str, pd.DataFrame]
output_dir: str
loss_name: str
```

输出：

```text
dict
```

功能：训练一个共享 `AffinityMLP` 并保存结果。

v2.1 训练逻辑：

1. `flatten_datasets()` 拼总表；
2. `split_by_group()` 整组划分；
3. `_write_split_report()` 保存 split 质检表；
4. `_build_train_loader()` 构造训练 loader；
5. `AffinityMLP(input_dim=feature_dim)` 初始化共享 head；
6. 每隔 `cfg.EVAL_EVERY` 在 val 上评估；
7. 用 `cfg.CHECKPOINT_METRIC` 选择最佳 epoch；
8. 保存模型、by-group 结果和 summary 字段。

pairwise 训练代码仍然是：

```python
emb_pos, emb_neg, fit_pos, fit_neg = batch
score_pos = model(emb_pos)
score_neg = model(emb_neg)
loss = loss_fn(score_pos, score_neg, fit_pos, fit_neg)
```

质检点：

- `losses.py` 不需要跟着改；
- checkpoint 中记录 `model_version`、`feature_mode`、`input_dim`、`split_objective`、`checkpoint_metric`；
- 打印主指标为 `val_weighted` / `test_weighted` / `test_median`。

### `run_global_training(embedded_datasets, output_dir, loss_names)`

自变量：

```text
embedded_datasets: dict[str, pd.DataFrame]
output_dir: str
loss_names: list[str] | None
```

输出：

```text
pd.DataFrame
```

功能：按 loss 列表依次训练多个全局模型，并保存汇总表。

v2.1 汇总主列：

```text
loss
feature_mode
val_weighted_spearman
test_weighted_spearman
test_median_spearman
test_n_groups
```

## 八、train.py

### `refresh_model_input_dim()`

自变量：

```text
无
```

输出：

```text
无直接返回值；修改 cfg.MODEL_INPUT_DIM
```

功能：根据命令行指定的 `MODEL_FEATURE_MODE` 刷新输入维度。

实现：

```python
if cfg.MODEL_FEATURE_MODE == "chain_concat":
    cfg.MODEL_INPUT_DIM = cfg.ESM_EMBEDDING_DIM * 2
elif cfg.MODEL_FEATURE_MODE == "scfv_mean":
    cfg.MODEL_INPUT_DIM = cfg.ESM_EMBEDDING_DIM
```

### `main()`

新增命令行参数：

```text
--model_feature_mode chain_concat/scfv_mean
--split_objective rows_balanced/groups_then_rows
--checkpoint_metric val_weighted_spearman/val_median_spearman
--min_label_diff
```

默认命令：

```bash
python train.py --mode embed
python train.py --mode train --loss mse hinge ranknet
```

对照实验：

```bash
# v2.1 默认：heavy/light 拼接，不做 pair 方向平衡
python train.py --mode train --loss mse hinge ranknet

# 对照 v1 输入表示
python train.py --mode embed --model_feature_mode scfv_mean
python train.py --mode train --model_feature_mode scfv_mean --loss ranknet

# 只改变 split 搜索目标
python train.py --mode train --loss ranknet --split_objective groups_then_rows

# 只改变 pair 构造难度
python train.py --mode train --loss ranknet --min_label_diff 0.05
```

注意：

- `chain_concat` 和 `scfv_mean` 需要各自对应的 cache；
- 默认 v2.1 cache 是 `heavy_embedding/light_embedding`；
- 用旧 v1 cache 直接跑默认 v2.1 训练会报缺列错误，需要重新 `--mode embed`。

## 九、未进入 v2.1 的方向

以下想法有潜力，但没有进入当前实现：

- 把大数据集切成 pseudo-rank-group；
- 使用 antigen embedding 或 antigen attention；
- 使用 heavy/light 双塔结构；
- 使用 `[heavy, light, |heavy-light|, heavy*light]`；
- 改 MSE 的 z-score 标签方式；
- 改 RankNet/Hinge 为 `(feature_a, feature_b, target)` 接口。

后续如果要做，需要单独开 ablation，写清楚依据、输入输出、失败风险和对照实验。

## 十、v2.1 函数质检索引

这一节按“代码是否改变”重新整理，方便人工质检。

规则：

```text
没有改变的函数：只列自变量和返回值。
改变过的函数：列自变量、返回值、功能、实现逻辑和质检点。
```

### 10.1 没有改变的函数

这些函数沿用 v1 逻辑。v2.1 文档只列接口，不再重复解释完整实现。

#### `canonical_filename(path_or_name)`

自变量：

```text
path_or_name: str
```

返回值：

```text
str
```

#### `dataset_name_from_file(path_or_name)`

自变量：

```text
path_or_name: str
```

返回值：

```text
str
```

#### `load_metadata(metadata_path)`

自变量：

```text
metadata_path: str
```

返回值：

```text
dict[str, dict]
```

#### `_read_csv_or_zip(filepath)`

自变量：

```text
filepath: str
```

返回值：

```text
pd.DataFrame | None
```

#### `classify_assay(name, metadata_row)`

自变量：

```text
name: str
metadata_row: dict | None
```

返回值：

```text
tuple[str, str]
```

#### `_numeric_fraction(series)`

自变量：

```text
series: pd.Series
```

返回值：

```text
float
```

#### `choose_label_column(df, assay_family)`

自变量：

```text
df: pd.DataFrame
assay_family: str
```

返回值：

```text
str | None
```

#### `_is_log_label(label_col, metadata_row)`

自变量：

```text
label_col: str
metadata_row: dict | None
```

返回值：

```text
bool
```

#### `normalize_label(df, label_col, assay_family, metadata_row)`

自变量：

```text
df: pd.DataFrame
label_col: str
assay_family: str
metadata_row: dict | None
```

返回值：

```text
pd.DataFrame
```

#### `load_one_dataset(filepath, metadata)`

自变量：

```text
filepath: str
metadata: dict[str, dict] | None
```

返回值：

```text
pd.DataFrame | None
```

#### `load_all_datasets(data_dir, metadata_path)`

自变量：

```text
data_dir: str
metadata_path: str
```

返回值：

```text
dict[str, pd.DataFrame]
```

#### `get_esm_model()`

自变量：

```text
无
```

返回值：

```text
tuple[EsmTokenizer, EsmModel]
```

#### `_seq_hash(seq)`

自变量：

```text
seq: str
```

返回值：

```text
str
```

#### `embed_sequence(seq)`

自变量：

```text
seq: str
```

返回值：

```text
np.ndarray
```

#### `get_or_compute_embedding(seq, cache_dir)`

自变量：

```text
seq: str
cache_dir: str
```

返回值：

```text
np.ndarray
```

#### `load_cached_datasets(cache_dir)`

自变量：

```text
cache_dir: str
```

返回值：

```text
dict[str, pd.DataFrame]
```

#### `MSELoss.forward(score, target)`

自变量：

```text
score: torch.Tensor
target: torch.Tensor
```

返回值：

```text
torch.Tensor
```

#### `PairwiseHingeLoss.forward(score_pos, score_neg, fitness_pos=None, fitness_neg=None)`

自变量：

```text
score_pos: torch.Tensor
score_neg: torch.Tensor
fitness_pos: torch.Tensor | None
fitness_neg: torch.Tensor | None
```

返回值：

```text
torch.Tensor
```

#### `RankNetLoss.forward(score_pos, score_neg, fitness_pos=None, fitness_neg=None)`

自变量：

```text
score_pos: torch.Tensor
score_neg: torch.Tensor
fitness_pos: torch.Tensor | None
fitness_neg: torch.Tensor | None
```

返回值：

```text
torch.Tensor
```

#### `AffinityMLP._init_weights()`

自变量：

```text
无
```

返回值：

```text
无
```

#### `AffinityMLP.forward(x)`

自变量：

```text
x: torch.Tensor
```

返回值：

```text
torch.Tensor
```

#### `_predict_scores(model, df)`

自变量：

```text
model: AffinityMLP
df: pd.DataFrame
```

返回值：

```text
np.ndarray
```

#### `evaluate_by_group(model, df, split_name)`

自变量：

```text
model: AffinityMLP
df: pd.DataFrame
split_name: str
```

返回值：

```text
tuple[dict, pd.DataFrame]
```

#### `set_seed(seed)`

自变量：

```text
seed: int
```

返回值：

```text
无
```

### 10.2 改变过的函数

#### `_standardize_sequences(df)`

自变量：

```text
df: pd.DataFrame
```

返回值：

```text
pd.DataFrame | None
```

功能：

```text
统一 heavy/light 列，并生成 v1/scfv_mean 模式需要的 sequence 列。
```

v2.1 改动：

- v1 只要存在 `light` 列，就会要求每一行 light 非空；
- v2.1 只强制 `heavy` 存在；
- 如果某行 `light` 缺失，`sequence` 直接等于 heavy；
- 如果某行 `light` 有效，`sequence = heavy + cfg.LINKER + light`。

实现逻辑：

```text
1. 根据 COLUMN_ALIASES 重命名列；
2. 如果没有 heavy，返回 None；
3. 删除 heavy 缺失或明显空值的行；
4. 如果 light 有效，拼接 heavy-linker-light；
5. 如果 light 无效，使用 heavy-only sequence。
```

质检点：

- VHH 或缺 light 的行不能在 data_loader 阶段被误删；
- v2.1 `chain_concat` 会在 embedding 阶段给缺 light 的行写零向量；
- `sequence` 只服务于 `scfv_mean` 消融和向后兼容。

#### `Config`

自变量：

```text
无。Config 是全局配置类。
```

返回值：

```text
cfg: Config
```

功能：

```text
集中保存 v2.1 的特征模式、split 策略、checkpoint 指标和输出路径。
```

v2.1 改动：

```text
MODEL_VERSION = "v2.1"
MODEL_FEATURE_MODE = "chain_concat"
MODEL_INPUT_DIM = 2 * ESM_EMBEDDING_DIM
SPLIT_OBJECTIVE = "rows_balanced"
CHECKPOINT_METRIC = "val_weighted_spearman"
OUTPUT_DIR = "results/affinity_model_v2"
```

质检点：

- 默认输入维度应为 2560；
- 默认输出目录不覆盖 v1；
- 默认 checkpoint 不再使用 `val_mean_spearman`。

#### `_normalize_sequence_value(value)`

自变量：

```text
value: Any
```

返回值：

```text
str
```

功能：

```text
把 DataFrame 单元格中的序列值变成稳定的大写字符串。
```

实现逻辑：

```text
1. None 或 NaN 返回空字符串；
2. 其它值转成 str；
3. 删除空格、换行、制表符；
4. 转成大写。
```

质检点：

- 同一条序列不能因为大小写或换行不同而重复 embedding；
- 不能把 NaN 字符串当作有效序列。

#### `_is_valid_sequence_value(seq)`

自变量：

```text
seq: str
```

返回值：

```text
bool
```

功能：

```text
判断清洗后的序列是否不是明显缺失值。
```

实现逻辑：

```text
过滤 "", "nan", "none", "null", "na"。
```

质检点：

- 不严格过滤 X 等未知残基；
- 只解决缺失值问题，不承担氨基酸合法性校验。

#### `_collect_unique_sequences(datasets, columns)`

自变量：

```text
datasets: dict[str, pd.DataFrame]
columns: list[str]
```

返回值：

```text
set[str]
```

功能：

```text
从多个 DataFrame 的指定序列列中收集唯一序列。
```

实现逻辑：

```text
1. 遍历每个 DataFrame；
2. 遍历 columns 中存在的列；
3. 清洗序列；
4. 过滤缺失；
5. 加入 set 去重。
```

质检点：

- `chain_concat` 时应扫描 heavy/light；
- `scfv_mean` 时应扫描 sequence；
- 同一序列只计算一次 ESM2 embedding。

#### `embed_all_datasets(datasets, cache_dir)`

自变量：

```text
datasets: dict[str, pd.DataFrame]
cache_dir: str
```

返回值：

```text
dict[str, pd.DataFrame]
```

功能：

```text
生成当前 feature mode 对应的 embedding cache，并写回各数据集 DataFrame。
```

v2.1 实现逻辑：

```text
chain_concat:
  1. 收集 heavy/light 唯一序列；
  2. 逐条调用 get_or_compute_embedding；
  3. 写回 heavy_embedding；
  4. light 有效时写回对应 light_embedding；
  5. light 缺失时写零向量；
  6. 写 has_light。

scfv_mean:
  1. 收集 sequence 唯一序列；
  2. 逐条调用 get_or_compute_embedding；
  3. 写回 embedding。
```

质检点：

- `chain_concat` cache 必须有 `heavy_embedding`、`light_embedding`、`has_light`；
- 缺 light 不能报错；
- 旧 v1 cache 不应被默认 v2.1 静默使用。

#### `_stack_embedding_column(df, column)`

自变量：

```text
df: pd.DataFrame
column: str
```

返回值：

```text
np.ndarray
```

功能：

```text
把 DataFrame 中保存 np.ndarray 的一列堆成二维特征矩阵。
```

实现逻辑：

```text
1. 检查 column 是否存在；
2. np.stack(df[column].values)；
3. 转成 np.float32。
```

质检点：

- 每行 embedding 维度必须一致；
- 缺列时必须报明确错误。

#### `build_model_feature_matrix(df)`

自变量：

```text
df: pd.DataFrame
```

返回值：

```text
np.ndarray
```

功能：

```text
根据 cfg.MODEL_FEATURE_MODE 生成 MLP 输入特征。
```

实现逻辑：

```text
chain_concat:
  feature = concat(heavy_embedding, light_embedding)
  输出维度 [N, 2560]

scfv_mean:
  feature = embedding
  输出维度 [N, 1280]
```

质检点：

- v2.1 不拼接 diff/product；
- v2.1 不加入 antigen embedding；
- 缺少 feature mode 所需列时必须报错。

#### `PairwiseRankingDataset.__init__(df, label_col, group_col, max_pairs_per_group, min_label_diff, seed)`

自变量：

```text
df: pd.DataFrame
label_col: str
group_col: str
max_pairs_per_group: int
min_label_diff: float
seed: int
```

返回值：

```text
PairwiseRankingDataset 实例
```

功能：

```text
构造 RankNet/Hinge 用的组内排序 pair。
```

v2.1 实现逻辑：

```text
1. 调用 build_model_feature_matrix(df)；
2. 保存 feature_dim；
3. 遍历每个 compatible_group；
4. 只构造 label_i - label_j > min_label_diff 的 pair；
5. 如果某组 pair 太多，随机采样到 max_pairs_per_group。
```

质检点：

- pair 仍然是 `(pos, neg)`；
- 不生成 `(neg, pos)`；
- 不生成 target；
- loss 接口不变。

#### `PairwiseRankingDataset.__getitem__(idx)`

自变量：

```text
idx: int
```

返回值：

```text
tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```

具体含义：

```text
(feature_pos, feature_neg, label_pos, label_neg)
```

实现逻辑：

```text
1. 根据 idx 取出 pos_idx, neg_idx；
2. 从 self.features 中取两条特征；
3. 从 self.labels 中取两个标签；
4. 全部转成 torch.tensor。
```

质检点：

- 第一个特征对应更高亲和力；
- 第二个特征对应更低亲和力。

#### `PointwiseRegressionDataset.__init__(df, target_col)`

自变量：

```text
df: pd.DataFrame
target_col: str
```

返回值：

```text
PointwiseRegressionDataset 实例
```

功能：

```text
构造 MSE 训练数据。
```

v2.1 实现逻辑：

```text
1. 调用 build_model_feature_matrix(df)；
2. 保存 feature_dim；
3. 读取 target_col，默认是 label_z。
```

质检点：

- MSE 路径不构造 pair；
- target 仍是 label_z。

#### `ScoringDataset.__init__(df, label_col)`

自变量：

```text
df: pd.DataFrame
label_col: str
```

返回值：

```text
ScoringDataset 实例
```

功能：

```text
构造验证/测试推理数据。
```

v2.1 实现逻辑：

```text
1. 调用 build_model_feature_matrix(df)；
2. 保存 feature_dim；
3. 读取真实排序标签 label。
```

质检点：

- 推理顺序必须和 DataFrame 行顺序一致；
- 评估时再按 compatible_group 计算 Spearman。

#### `AffinityMLP.__init__(input_dim=None, hidden_dim=cfg.HIDDEN_DIM, dropout=cfg.DROPOUT)`

自变量：

```text
input_dim: int | None
hidden_dim: int
dropout: float
```

返回值：

```text
AffinityMLP 实例
```

功能：

```text
创建共享 MLP 打分头。
```

v2.1 实现逻辑：

```text
1. 如果 input_dim is None，读取 cfg.MODEL_INPUT_DIM；
2. Linear(input_dim, hidden_dim)；
3. GELU；
4. Dropout；
5. Linear(hidden_dim, 1)。
```

质检点：

- 不创建 per-dataset head；
- 不引入 attention；
- 输出仍是一个标量 score。

#### `flatten_datasets(embedded_datasets)`

自变量：

```text
embedded_datasets: dict[str, pd.DataFrame]
```

返回值：

```text
pd.DataFrame
```

功能：

```text
把多个 embedded DataFrame 拼成训练总表，并检查当前 feature mode 所需列。
```

v2.1 实现逻辑：

```text
1. 每个 DataFrame 拷贝一份；
2. 补齐 dataset / compatible_group；
3. concat 成 all_df；
4. 根据 MODEL_FEATURE_MODE 检查必要特征列；
5. 删除 label / label_z 缺失行。
```

质检点：

- 不合并 compatible_group；
- 不改变 label；
- 旧 cache 缺列时必须报错。

#### `_select_groups_near_target(group_sizes, target_rows, forbidden_groups, min_groups, target_groups=None)`

自变量：

```text
group_sizes: pd.Series
target_rows: int
forbidden_groups: set[str]
min_groups: int
target_groups: int | None
```

返回值：

```text
set[str]
```

功能：

```text
从候选 compatible_group 中选择 val/test group。
```

v2.1 实现逻辑：

```text
1. 去掉 forbidden_groups；
2. 优先过滤过大的候选 group；
3. 动态规划枚举 group 组合；
4. rows_balanced：优先样本数接近 target_rows；
5. groups_then_rows：优先 group 数接近 target_groups，再看样本数。
```

质检点：

- 同一 group 不会被 val/test 同时选中；
- 默认仍是 rows_balanced；
- group 不能被拆开。

#### `split_by_group(df)`

自变量：

```text
df: pd.DataFrame
```

返回值：

```text
tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
```

具体含义：

```text
(train_df, val_df, test_df)
```

功能：

```text
按 compatible_group 整组划分 train/val/test。
```

v2.1 实现逻辑：

```text
1. 统计每个 group 的行数；
2. 计算 val/test 的目标行数；
3. 计算 val/test 的目标 group 数；
4. 调用 _select_groups_near_target 选择 test；
5. 再排除 test 后选择 val；
6. 剩余 group 进入 train。
```

质检点：

- train/val/test group 两两不相交；
- train 不能为空；
- 打印行数、group 数、比例。

#### `_first_value(group_df, col, default="")`

自变量：

```text
group_df: pd.DataFrame
col: str
default: Any
```

返回值：

```text
Any
```

功能：

```text
取 group 内某列第一个非空值，用于 split report。
```

实现逻辑：

```text
1. 如果列不存在，返回 default；
2. 删除空值；
3. 转字符串；
4. 返回第一个唯一值。
```

#### `_write_split_report(train_df, val_df, test_df, output_dir, loss_name)`

自变量：

```text
train_df: pd.DataFrame
val_df: pd.DataFrame
test_df: pd.DataFrame
output_dir: str
loss_name: str
```

返回值：

```text
str
```

功能：

```text
保存 group 级 split 质检表。
```

实现逻辑：

```text
1. 遍历 train/val/test；
2. 每个 split 内 groupby compatible_group；
3. 统计 n_rows、n_unique_label、label_min、label_max；
4. 记录 dataset、assay_family、assay_units；
5. 如果存在 has_light，计算 has_light_fraction；
6. 保存到 global_{loss_name}_split_report.csv。
```

质检点：

- 每个 compatible_group 只出现一次；
- n_rows 总和等于训练总表行数；
- 能检查 VHH/缺 light 是否集中在某个 split。

#### `_build_train_loader(train_df, loss_name)`

自变量：

```text
train_df: pd.DataFrame
loss_name: str
```

返回值：

```text
tuple[DataLoader, nn.Module, int]
```

具体含义：

```text
(loader, loss_fn, feature_dim)
```

功能：

```text
根据 loss_name 构造训练 DataLoader、损失函数和输入维度。
```

实现逻辑：

```text
1. mse 使用 PointwiseRegressionDataset；
2. hinge/ranknet 使用 PairwiseRankingDataset；
3. 从 dataset.feature_dim 读出输入维度；
4. 创建 DataLoader。
```

质检点：

- pairwise 路径仍然是 pos/neg；
- MSE 路径不构造 pair；
- `feature_dim` 要传给 AffinityMLP。

#### `train_global_model(embedded_datasets, output_dir, loss_name)`

自变量：

```text
embedded_datasets: dict[str, pd.DataFrame]
output_dir: str
loss_name: str
```

返回值：

```text
dict
```

功能：

```text
训练一个共享 AffinityMLP，并保存模型、明细结果和 split report。
```

实现逻辑：

```text
1. flatten_datasets；
2. split_by_group；
3. _write_split_report；
4. _build_train_loader；
5. AffinityMLP(input_dim=feature_dim)；
6. 训练每个 epoch；
7. 每隔 EVAL_EVERY 计算 val 指标；
8. 使用 cfg.CHECKPOINT_METRIC 保存最佳模型；
9. 最后评估 val/test；
10. 保存 checkpoint、by-group CSV、summary 字段。
```

质检点：

- checkpoint 主指标默认是 val_weighted_spearman；
- checkpoint 记录 feature_mode/input_dim/split_objective；
- RankNet/Hinge loss 接口未变。

#### `run_global_training(embedded_datasets, output_dir, loss_names)`

自变量：

```text
embedded_datasets: dict[str, pd.DataFrame]
output_dir: str
loss_names: list[str] | None
```

返回值：

```text
pd.DataFrame
```

功能：

```text
依次训练多个 loss 的全局模型，并保存 summary_global_losses.csv。
```

实现逻辑：

```text
1. 如果 loss_names is None，使用 LOSS_REGISTRY 全部 loss；
2. 逐个调用 train_global_model；
3. 合并结果为 DataFrame；
4. 写 summary_global_losses.csv；
5. 打印 weighted/median 主结果。
```

#### `refresh_model_input_dim()`

自变量：

```text
无
```

返回值：

```text
无。该函数直接修改 cfg.MODEL_INPUT_DIM。
```

功能：

```text
命令行参数覆盖 MODEL_FEATURE_MODE 后，重新计算模型输入维度。
```

实现逻辑：

```text
chain_concat -> 2560
scfv_mean -> 1280
```

质检点：

- CLI 改 feature mode 后，input_dim 不能仍停留在 import 时的旧值。

#### `main()`

自变量：

```text
无直接函数自变量；通过 argparse 读取命令行参数。
```

返回值：

```text
无
```

功能：

```text
训练脚本主入口，负责解析参数、刷新 cfg、调用 embed/train 流程。
```

v2.1 新增参数：

```text
--model_feature_mode
--split_objective
--checkpoint_metric
--min_label_diff
```

实现逻辑：

```text
1. 解析 CLI；
2. 写回 cfg；
3. refresh_model_input_dim；
4. set_seed；
5. mode=embed/all 时加载数据并提 embedding；
6. mode=train/all 时加载 cache 并训练；
7. 打印 weighted/median 主结果。
```
