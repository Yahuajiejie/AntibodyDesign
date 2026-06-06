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

阅读这一节时，先区分两个东西：

```text
CSV 列名：当前训练数据文件里真实存在的列名，例如 KD (nM)、fitness、heavy。
metadata：另一个说明书文件 data/flab_metadata.csv，用来说明每个 CSV 大概是什么实验。
```

函数解释格式：

```text
自变量：调用函数时传进去的东西。
输出：函数返回给下一步的东西。
```

### `canonical_filename(path_or_name)`

自变量：

```text
path_or_name: str
```

含义：一个文件路径或文件名。

例子：

```text
"data/binding/AbRank_dataset.csv.zip"
"AbRank_dataset.csv.zip"
```

输出：

```text
str
```

含义：去掉路径、并把 `.csv.zip` 转成 `.csv` 后的标准文件名。

例子：

```text
"data/binding/AbRank_dataset.csv.zip" -> "AbRank_dataset.csv"
```

逻辑：

1. 使用 `os.path.basename()` 取文件名；
2. 如果本地文件是 `.csv.zip`，去掉 `.zip`；
3. 返回 metadata 中能匹配的 `.csv` 文件名。

外部库用法：

```python
os.path.basename(path_or_name)
```

`os.path.basename` 是 Python 标准库函数，用于从路径中取最后一级文件名。

### `dataset_name_from_file(path_or_name)`

自变量：

```text
path_or_name: str
```

含义：一个 CSV 或 ZIP 文件路径/文件名。

输出：

```text
str
```

含义：去掉 `.csv` / `.csv.zip` 后的数据集名，后面会用作 `dataset` 和 `compatible_group`。

例子：

```text
"AbRank_dataset.csv.zip" -> "AbRank_dataset"
"garbinski2023_kd.csv" -> "garbinski2023_kd"
```

逻辑：

1. 先调用 `canonical_filename()`；
2. 再去掉 `.csv`。

### `load_metadata(metadata_path)`

自变量：

```text
metadata_path: str
```

含义：metadata 说明书文件路径，默认是：

```text
data/flab_metadata.csv
```

输出：

```text
dict[str, dict]
```

含义：一个字典。key 是文件名，value 是该文件在 metadata 表里的整行信息。

例子：

```python
{
    "garbinski2023_kd.csv": {
        "filename": "garbinski2023_kd.csv",
        "category": "binding",
        "assay/units_raw": "-log (KD [M] )",
        "assay/units": "SPR Kd",
        ...
    }
}
```

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

自变量：

```text
filepath: str
```

含义：一个训练数据文件路径，可以是普通 CSV，也可以是 `.csv.zip`。

输出：

```text
pd.DataFrame | None
```

含义：

```text
读取成功 -> 返回 DataFrame
读取失败/zip 里没有 CSV -> 返回 None
```

注意：如果 zip 里有多个 CSV，当前代码只读取第一个 CSV。

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

自变量：

```text
name: str
metadata_row: dict | None
```

`name` 是数据集名，通常来自文件名。

例子：

```text
"kothiwal2025htp_DCC_spr"
```

`metadata_row` 是 metadata 里对应这个 CSV 的那一行。如果找不到 metadata，就是 `None`。

输出：

```text
tuple[str, str]
```

含义：返回两个字符串。

```text
第一个字符串：assay_family，数据类型分类
第二个字符串：reason，为什么这样分类
```

例子：

```python
("kd", "真实 Kd 类亲和力数据")
("ic50", "IC50 是功能性抑制读数，不等同 Kd")
("mixed_affinity_functional", "同时包含 Kd 与 IC50/EC50，默认不自动拆分")
```

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

### `_numeric_fraction(series)`

自变量：

```text
series: pd.Series
```

含义：DataFrame 中的一列。

例子：

```text
df["KD (nM)"]
df["fitness"]
```

输出：

```text
float
```

含义：这一列里有多少比例的值可以转成数字。

例子：

```text
1.0  表示 100% 都能转成数字
0.5  表示 50% 能转成数字
0.0  表示完全不是数值列
```

用途：判断某列能不能当作亲和力标签列。

### `choose_label_column(df, assay_family)`

自变量：

```text
df: pd.DataFrame
assay_family: str
```

`df` 是刚从 CSV 读出来、已经标准化过 heavy/light 的表格。

`assay_family` 是 `classify_assay()` 的第一个返回值，例如：

```text
"kd"
"ic50"
"binary"
```

输出：

```text
str | None
```

含义：

```text
找到标签列 -> 返回列名，例如 "KD (nM)"、"neg_log_Kd"、"fitness"
找不到可靠标签列 -> 返回 None
```

例子：

```text
CSV 列名: heavy, light, KD (nM), fitness
输出: "KD (nM)"

CSV 列名: heavy, light, neg_log_Kd, fitness
输出: "neg_log_Kd"
```

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

### `_is_log_label(label_col, metadata_row)`

自变量：

```text
label_col: str
metadata_row: dict | None
```

`label_col` 是 `choose_label_column()` 选出来的列名。

例子：

```text
"KD (nM)"
"neg_log_Kd"
"fitness"
```

`metadata_row` 是 metadata 里对应这个 CSV 的说明行。

输出：

```text
bool
```

含义：

```text
True  -> 这列已经是 -logKd / neg_log_Kd，越大越强，不需要再取 -log10
False -> 这列是原始 Kd，越小越强，需要转成 -log10(Kd)
```

例子：

```text
"_is_log_label('neg_log_Kd', row)" -> True
"_is_log_label('KD (nM)', row)" -> False
"_is_log_label('fitness', row)" -> 需要看 metadata
```

人话解释：

```text
列名很明确时，相信列名。
列名很模糊时，比如 fitness，才参考 metadata。
```

### `normalize_label(df, label_col, assay_family, metadata_row)`

自变量：

```text
df: pd.DataFrame
label_col: str
assay_family: str
metadata_row: dict | None
```

`df` 是当前 CSV 对应的数据表。

`label_col` 是亲和力标签列名，例如：

```text
"KD (nM)"
"neg_log_Kd"
"fitness"
```

`assay_family` 目前主要使用 `"kd"`。

`metadata_row` 是当前 CSV 的说明书信息。

输出：

```text
pd.DataFrame
```

含义：返回一个新 DataFrame，比输入多出这些列：

```text
label_raw        原始标签值
label            方向统一后的标签，越大亲和力越强
label_transform  标签如何被处理
label_z          组内 z-score，MSE 使用
label_rank       组内百分位 rank
```

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

自变量：

```text
df: pd.DataFrame
```

含义：从 CSV 读出的原始表格。

输出：

```text
pd.DataFrame | None
```

含义：

```text
成功 -> 返回带 sequence 列的新 DataFrame
失败 -> 返回 None，例如缺少 heavy 列
```

新增的 `sequence` 列是模型真正要送进 ESM2 的序列。

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

自变量：

```text
filepath: str
metadata: dict[str, dict] | None
```

`filepath` 是一个 CSV 或 `.csv.zip` 文件路径。

`metadata` 是 `load_metadata()` 的输出，也就是“文件名 -> 说明书行”的字典。

输出：

```text
pd.DataFrame | None
```

含义：

```text
合格数据集 -> 返回标准化后的 DataFrame
不合格数据集 -> 返回 None
```

不合格包括：

```text
不是 Kd 主任务
缺少 heavy
找不到标签列
数据量小于 MIN_GROUP_SIZE
label 没有排序差异
```

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

自变量：

```text
data_dir: str
metadata_path: str
```

`data_dir` 是 binding CSV 文件所在目录。

`metadata_path` 是 metadata 说明书文件路径。

输出：

```text
dict[str, pd.DataFrame]
```

含义：key 是数据集名，value 是 `load_one_dataset()` 返回的标准化 DataFrame。

例子：

```python
{
    "garbinski2023_kd": DataFrame(...),
    "kothiwal2025htp_DCC_spr": DataFrame(...),
}
```

逻辑：

1. 读取 metadata；
2. 扫描 `data/binding` 下所有 `.csv` / `.csv.zip`；
3. 逐个调用 `load_one_dataset()`；
4. 返回 `dict[str, DataFrame]`。

## 五、embeddings.py

该模块基本沿用旧实现，用 ESM2 提取冻结 embedding。

### `get_esm_model()`

自变量：

```text
无
```

输出：

```text
tuple[EsmTokenizer, EsmModel]
```

含义：

```text
EsmTokenizer: 把氨基酸字符串转成 ESM2 能读的 token id
EsmModel: ESM2 模型本体，用来提取 embedding
```

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

自变量：

```text
seq: str
```

含义：一条已经拼好的抗体序列。

例子：

```text
双链抗体: heavy + linker + light
VHH: heavy
```

输出：

```text
np.ndarray
```

含义：ESM2 提取出的定长向量，默认形状是：

```text
(1280,)
```

这就是后面 MLP 的输入。

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

自变量：

```text
seq: str
cache_dir: str
```

`seq` 是一条抗体序列。

`cache_dir` 是 embedding 缓存目录，例如：

```text
cache/embeddings
```

输出：

```text
np.ndarray
```

含义：这条序列的 ESM2 embedding。

逻辑：

1. 用 MD5 hash 生成缓存文件名；
2. 如果 `.npy` 存在，用 `np.load()` 读取；
3. 否则调用 ESM2 计算并 `np.save()`。

### `embed_all_datasets(datasets, cache_dir)`

自变量：

```text
datasets: dict[str, pd.DataFrame]
cache_dir: str
```

`datasets` 是 `load_all_datasets()` 的输出。

`cache_dir` 是 embedding 缓存目录。

输出：

```text
dict[str, pd.DataFrame]
```

含义：和输入 datasets 结构一样，但每个 DataFrame 多出一列：

```text
embedding
```

这一列的每个值都是一个 `np.ndarray`，形状默认是 `(1280,)`。

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

### `load_cached_datasets(cache_dir)`

自变量：

```text
cache_dir: str
```

含义：embedding 缓存目录。

输出：

```text
dict[str, pd.DataFrame]
```

含义：从 `embedded_datasets.pkl` 读取出的 embedded datasets。

用途：`python train.py --mode train` 时不用重新跑 ESM2。

## 六、dataset.py

### `PairwiseRankingDataset`

用于 `ranknet` 和 `hinge`。

构造函数自变量：

```text
df
label_col: str
group_col: str
max_pairs_per_group: int
min_label_diff: float
seed: int
```

`df` 是训练集 DataFrame，必须包含：

```text
embedding
label
compatible_group
```

`label_col` 指排序标签列，默认是 `label`。

`group_col` 指可比较组列，默认是 `compatible_group`。

`max_pairs_per_group` 限制每个组最多产生多少 pair，避免 O(N²) 爆炸。

`min_label_diff` 表示两个样本的 label 至少差多少才构造 pair。

`seed` 控制随机采样 pair 的可复现性。

Dataset 输出：

```text
__getitem__(idx) -> (emb_pos, emb_neg, fit_pos, fit_neg)
```

含义：

```text
emb_pos: 高亲和力样本的 embedding
emb_neg: 低亲和力样本的 embedding
fit_pos: 高亲和力样本的 label
fit_neg: 低亲和力样本的 label
```

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

构造函数自变量：

```text
df
target_col: str
```

`df` 是训练集 DataFrame，必须包含：

```text
embedding
label_z
```

`target_col` 默认是 `label_z`。

Dataset 输出：

```text
__getitem__(idx) -> (embedding, target)
```

含义：

```text
embedding: 单条抗体序列的 ESM2 向量
target: 组内标准化后的 label_z
```

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

构造函数自变量：

```text
df
label_col: str
```

`df` 是验证集或测试集 DataFrame。

`label_col` 默认是 `label`，用于和模型预测分数计算 Spearman。

Dataset 输出：

```text
__getitem__(idx) -> (embedding, label)
```

含义：

```text
embedding: 单条抗体序列的 ESM2 向量
label: 方向统一后的真实排序标签
```

逻辑：

每次返回：

```text
(embedding, label)
```

真实的 `compatible_group` 保留在 trainer 的 DataFrame 中，用于按组计算 Spearman。

## 七、losses.py

### `MSELoss`

自变量：

```text
score: torch.Tensor
target: torch.Tensor
```

`score` 是模型输出，形状通常是：

```text
[batch, 1]
```

`target` 是真实目标值，这里必须是 `label_z`，形状通常也是：

```text
[batch, 1]
```

输出：

```text
torch.Tensor
```

含义：一个标量 loss，用于反向传播。

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

自变量：

```text
score_pos: torch.Tensor
score_neg: torch.Tensor
fitness_pos: torch.Tensor | None
fitness_neg: torch.Tensor | None
```

`score_pos` 是高亲和力样本的模型分数。

`score_neg` 是低亲和力样本的模型分数。

`fitness_pos` / `fitness_neg` 是为了和其它 pairwise loss 接口一致保留的参数，当前 loss 不使用它们。

输出：

```text
torch.Tensor
```

含义：一个标量 loss。

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

自变量：

```text
score_pos: torch.Tensor
score_neg: torch.Tensor
fitness_pos: torch.Tensor | None
fitness_neg: torch.Tensor | None
```

含义同 `PairwiseHingeLoss`。

输出：

```text
torch.Tensor
```

含义：一个标量 loss。

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

构造函数自变量：

```text
input_dim: int
hidden_dim: int
dropout: float
```

`input_dim` 是输入 embedding 维度，ESM2-650M 默认是 `1280`。

`hidden_dim` 是 MLP 中间层维度，默认是 `256`。

`dropout` 是训练时随机丢弃神经元的比例，默认是 `0.2`。

模型输入：

```text
x: torch.Tensor
```

形状：

```text
[batch, 1280]
```

模型输出：

```text
torch.Tensor
```

形状：

```text
[batch, 1]
```

含义：每条序列的亲和力预测分数。分数越大，模型认为亲和力越强。

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

自变量：

```text
无
```

输出：

```text
无返回值
```

作用：直接修改模型内部 Linear 层的参数初始值。

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

自变量：

```text
embedded_datasets: dict[str, pd.DataFrame]
```

含义：`embed_all_datasets()` 或 `load_cached_datasets()` 的输出。

结构例子：

```python
{
    "garbinski2023_kd": DataFrame(...),
    "kothiwal2025htp_DCC_spr": DataFrame(...),
}
```

每个 DataFrame 必须包含：

```text
embedding
label
label_z
compatible_group
```

输出：

```text
pd.DataFrame
```

含义：把所有 DataFrame 上下拼接成一张大表。

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

自变量：

```text
df: pd.DataFrame
```

含义：`flatten_datasets()` 输出的大表。

必须包含：

```text
compatible_group
```

输出：

```text
tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
```

含义：

```text
(train_df, val_df, test_df)
```

这三张表按 `compatible_group` 整组划分。同一个 group 只会出现在其中一张表里。

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

自变量：

```text
model: AffinityMLP
df: pd.DataFrame
```

`model` 是已经训练好或正在评估的模型。

`df` 是需要推理的数据表，必须包含 `embedding`。

输出：

```text
np.ndarray
```

含义：模型对 df 中每一行样本输出的分数，顺序和 df 行顺序一致。

形状：

```text
(len(df),)
```

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

自变量：

```text
model: AffinityMLP
df: pd.DataFrame
split_name: str
```

`model` 是要评估的模型。

`df` 是验证集或测试集 DataFrame。

`split_name` 是当前评估集合名称，例如：

```text
"val"
"test"
```

输出：

```text
tuple[dict, pd.DataFrame]
```

第一个输出 `summary` 是整体汇总指标。

例子：

```python
{
    "test_mean_spearman": 0.31,
    "test_median_spearman": 0.28,
    "test_weighted_spearman": 0.34,
    "test_n_groups": 6,
}
```

第二个输出 `detail_df` 是逐组明细表，每个 `compatible_group` 一行。

典型列：

```text
split
compatible_group
dataset
n
n_unique_label
spearman
p_value
assay_family
assay_units
```

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

自变量：

```text
train_df: pd.DataFrame
loss_name: str
```

`train_df` 是训练集 DataFrame。

`loss_name` 是损失函数名称，允许：

```text
"mse"
"hinge"
"ranknet"
```

输出：

```text
tuple[DataLoader, nn.Module]
```

第一个输出是 PyTorch `DataLoader`，负责每次喂给模型一个 batch。

第二个输出是损失函数对象，例如 `MSELoss()` 或 `RankNetLoss()`。

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

自变量：

```text
embedded_datasets: dict[str, pd.DataFrame]
output_dir: str
loss_name: str
```

`embedded_datasets` 是已经带有 `embedding` 列的数据集字典。

`output_dir` 是模型和结果文件保存目录。

`loss_name` 是本次训练使用的 loss：

```text
"mse"
"hinge"
"ranknet"
```

输出：

```text
dict
```

含义：本次训练的汇总结果。

典型字段：

```text
loss
n_train
n_val
n_test
n_train_groups
n_val_groups
n_test_groups
best_epoch
val_mean_spearman
test_mean_spearman
model_path
```

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

自变量：

```text
embedded_datasets: dict[str, pd.DataFrame]
output_dir: str
loss_names: list[str] | None
```

`loss_names` 是要跑哪些 loss。

例子：

```python
["mse", "hinge", "ranknet"]
["ranknet"]
```

如果是 `None`，默认跑注册表里的全部 loss。

输出：

```text
pd.DataFrame
```

含义：每一行是一个 loss 的训练结果汇总。

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

自变量：

```text
seed: int
```

含义：随机种子。相同 seed 通常能得到相同的数据划分、pair 采样和训练初始化。

输出：

```text
无返回值
```

作用：修改 PyTorch、CUDA、numpy 的随机状态。

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

自变量：

```text
无直接自变量
```

它从命令行读取参数。

命令行参数包括：

```text
--mode        embed / train / all
--data_dir    数据目录
--cache_dir   embedding 缓存目录
--output_dir  结果输出目录
--loss        mse / hinge / ranknet，可多选
```

输出：

```text
无直接返回值
```

作用：

```text
根据命令行参数执行数据加载、embedding 提取、模型训练，并把结果写入磁盘。
```

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
