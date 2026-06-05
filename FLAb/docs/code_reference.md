# 代码参考文档：FLAb 抗体亲和力预测模型

> 面向想要审查或修改代码的研究人员。本文档覆盖所有关键模块的设计逻辑、函数说明和第三方库调用解释。

---

## 一、整体架构

```
输入：抗体序列（heavy + light）
         │
         ▼
  [data_loader.py]  ← 读取 FLAb CSV，过滤、修正、合并
         │
         ▼
  [embeddings.py]   ← ESM2-650M 提取 1280 维向量，缓存到磁盘
         │
         ▼
  [dataset.py]      ← 构造 Pairwise (A,B) 训练对
         │
         ▼
  [trainer.py]      ← 训练 MLP，用 val_spearman 选最优参数
         │           使用 [losses.py] 中的三种损失函数
         ▼
  [model.py]        ← AffinityMLP 输出排序分数
         │
         ▼
  输出：summary_all_losses.csv（每个 benchmark 的 Spearman）
```

**关键设计决策：**
- ESM2 **冻结**（不参与训练），只作为特征提取器
- 每个 benchmark **独立训练**一个 MLP head（per-benchmark fine-tuning）
- 评估指标：**Spearman 相关系数**（排序相关，≠ 回归精度）
- 三种损失函数并行消融：MSE / Pairwise Hinge / RankNet

---

## 二、config.py

### 设计目的
所有超参数集中在一个 `Config` 类里，修改实验设置只需改这一个文件，不用翻遍各模块。

### 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ESM_MODEL_NAME` | `facebook/esm2_t33_650M_UR50D` | HuggingFace 上的 ESM2-650M 模型名 |
| `ESM_EMBEDDING_DIM` | 1280 | ESM2-650M 最后一层的 hidden size，固定值 |
| `LINKER` | `GGGGSGGGGSGGGGS` | 拼接重链和轻链的 GS 连接肽，模拟 scFv 结构 |
| `MAX_SEQ_LEN` | 512 | Tokenizer 的最大 token 数，超出截断。ESM2 上限是 1022，留 buffer |
| `MAX_PAIRS` | 10000 | 每个 benchmark 最多采样多少个训练对。N 条序列理论上有 N(N-1)/2 对，3000 条→900 万对，需要上限控制 |
| `MIN_DATASET_SIZE` | 30 | 少于此行数的数据集直接跳过。理由：80/10/10 划分后 test set 至少要有 3 条，30×10%=3 |
| `MAX_DATASET_SIZE` | 5000 | 超过此行数的数据集跳过。百万级数据集多为预测值（不是实验 Kd），且跑不完 |
| `MARGIN` | 0.1 | Pairwise Hinge Loss 的间隔参数 |

---

## 三、data_loader.py

### 3.1 `load_one_dataset(filepath)` — 单文件加载

**总体逻辑：** 读取 → 统一列名 → 检查必要列 → 删除缺失值 → 规模过滤 → 修正 fitness 方向 → 拼接序列

#### 步骤1：读取文件

```python
with zipfile.ZipFile(filepath) as z:
    csv_names = [n for n in z.namelist() if n.endswith(".csv")]
    with z.open(csv_names[0]) as f:
        df = pd.read_csv(f, low_memory=False)
```

- `zipfile.ZipFile`：Python 标准库，无需安装。`.namelist()` 返回 zip 内所有文件名的列表，`.open()` 在不解压到磁盘的情况下直接读取内部文件
- `pd.read_csv(f, low_memory=False)`：`low_memory=False` 让 pandas 一次性读取整列而不是分块，避免混合类型警告（DtypeWarning）

#### 步骤2：统一列名

```python
rename_map = {k: v for k, v in COLUMN_ALIASES.items() if k in df.columns}
df = df.rename(columns=rename_map)
```

- `COLUMN_ALIASES`：字典，记录不同数据集的非标准列名到标准列名的映射。例如 AbRank 用 `Ab_heavy_chain_seq` 而不是 `heavy`
- `df.rename(columns=rename_map)`：只重命名存在的列，不会因为键不在 df 里而报错

#### 步骤4：删除缺失值

```python
df = df.dropna(subset=required_cols).reset_index(drop=True)
df["fitness"] = pd.to_numeric(df["fitness"], errors="coerce")
```

- `pd.to_numeric(errors="coerce")`：把无法转为数字的值（如字符串"N/A"）变成 `NaN`，而不是报错。之后再 `dropna` 删除这些行
- `.reset_index(drop=True)`：删行后 index 会有跳空（如 0,2,3,5...），`reset_index` 把 index 重置为 0,1,2,3...，防止后续 `.iloc` 和 `.loc` 行为不一致

#### 步骤6：修正 fitness 方向 ⚠️ 关键逻辑

**问题背景：** FLAb README 声称所有数据集 fitness 越高越好，但实际上部分数据集的 fitness 列是原始 Kd (nM) 或 EC50 (nM)，数值越大亲和力越差。如不修正，pairwise loss 会训练出方向相反的模型。

已确认的问题数据集示例：
- `shanehsazzadeh2024igdesign_*_kd`：fitness = 原始 Kd，范围 [0.27, 3980] nM
- `kothiwal2025htp_*_ec50`：fitness = 原始 EC50，范围 [22, 201] nM

```python
fit = df["fitness"]
all_positive = (fit > 0).all()          # 全为正数
span_ratio   = fit.max() / fit.min()    # 最大/最小 比值
is_binary    = set(fit.unique()).issubset({0.0, 1.0})  # 是否只有 0/1

if all_positive and span_ratio > 5 and not is_binary:
    df["fitness"] = -np.log10(fit)
```

**检测规则：** 若 fitness 全为正数 且 max/min > 5（跨越半个数量级以上），判断为原始物理量，取 `-log10` 修正方向。

**已知局限性：**
1. **漏报**：某些 SPR Kd 数据集（如 `kothiwal_*_spr`）范围较窄（如 12-30 nM，span≈2.4），不会被检测到，但方向仍然是错的
2. **误报**：`makowski2022cooptimization_igg_ant` 的 fitness 是归一化结合得分 [0.06, 1.28]，span≈21 会被误判为原始测量值
3. 根本原因：自动检测无法替代人工核查每个数据集的原始论文

**TODO（已知问题）：** 应维护一个显式的白名单/黑名单字典，对无法自动判断的数据集人工指定方向。

#### 步骤7：拼接输入序列

```python
df["sequence"] = df["heavy"] + cfg.LINKER + df["light"]
# 或（纳米抗体）：
df["sequence"] = df["heavy"]
```

ESM2 是单序列模型，无法直接接收双链输入。用 GS linker（`GGGGSGGGGSGGGGS`）将 VH 和 VL 拼成一条序列，模拟 scFv 结构。linker 本身编码为序列的一部分参与 embedding 计算。

---

### 3.2 `_merge_key(name)` — 提取合并键

**逻辑：** 检查数据集名末尾是否是可合并的 assay 标识词（`_MERGEABLE_SUFFIXES`），若是则返回去掉后缀的名字作为合并键，否则返回 `None`。

**物理依据（决定哪些 assay 可合并）：**

| Assay 类型 | 是否可合并 | 原因 |
|---|---|---|
| `fab` / `igg` | ✅ | 同批突变抗体不同表达格式，Kd 排序保持一致 |
| `spr` / `bli` | ✅ | 均测平衡解离常数 Kd，物理量相同 |
| `titeseq` / `flow` | ✅ | 高通量 Kd 测量方法，测量结果等价 |
| `ec50` | ❌ | 功能性抑制浓度，≠ Kd，受细胞类型等影响 |
| `ic50` | ❌ | 半数抑制浓度，功能性读数 |
| `binary` | ❌ | 离散标签，不能与连续 Kd 混用 |

---

### 3.3 `merge_small_datasets(datasets, min_size, min_test_size=5)` — 小数据集合并

**触发条件：** 同一合并键下有多个文件，且其中至少一个文件的行数 < `min_test_size × 10`（默认 < 50 条），即 10% test set 不足 5 条时触发合并。

**合并过程：**
```python
mu, sigma = piece["fitness"].mean(), piece["fitness"].std()
piece["fitness"] = (piece["fitness"] - mu) / sigma  # z-score 归一化
```

z-score 归一化的必要性：同一批突变抗体用 FAB 格式和 IgG 格式测出的 Kd 绝对值不同（IgG 有双价亲合力效应，表观 Kd 更小），但相对排序一致。归一化后量纲统一，可以合并。

```python
combined = pd.concat(pieces, ignore_index=True)
```
- `pd.concat(ignore_index=True)`：纵向拼接多个 DataFrame，`ignore_index=True` 将拼接后的 index 重置为 0,1,2...，避免重复 index

---

## 四、embeddings.py

### 4.1 `get_esm_model()` — 懒加载 ESM2

```python
_tokenizer = EsmTokenizer.from_pretrained(cfg.ESM_MODEL_NAME)
_model = EsmModel.from_pretrained(cfg.ESM_MODEL_NAME).to(cfg.DEVICE)
_model.eval()
```

- `EsmTokenizer.from_pretrained(name)`：从 HuggingFace Hub 下载并加载对应 ESM2 版本的分词器。分词器负责将氨基酸字符序列转为整数 token ID
- `EsmModel.from_pretrained(name)`：加载 ESM2 的 Transformer encoder 主体（不含 LM head）。`EsmForMaskedLM` 是带 LM 头的完整模型，用于 masked language modeling；`EsmModel` 只输出 hidden states，更轻量
- `.to(cfg.DEVICE)`：将模型参数移到 GPU（如果可用）。这步必须在第一次 forward 之前调用
- `_model.eval()`：切换到推理模式。关闭 Dropout（推理时不随机丢弃神经元）和 BatchNorm 的滑动统计更新。**不调用 eval() 会导致 Dropout 随机性使 embedding 每次不同，缓存失效**

**为什么用全局变量缓存模型：**
`embed_sequence()` 对每条序列调用一次，如果每次都重新 `from_pretrained`，会反复从磁盘加载 2.5GB 的模型文件。用模块级全局变量 `_model` 只加载一次。

---

### 4.2 `embed_sequence(seq)` — 单序列 Embedding

```python
inputs = tokenizer(
    seq,
    return_tensors="pt",    # 返回 PyTorch tensor（而非 list 或 numpy）
    truncation=True,        # 超出 max_length 时截断，不报错
    max_length=cfg.MAX_SEQ_LEN,
    padding=False,          # 单条序列不需要 padding
)
inputs = {k: v.to(cfg.DEVICE) for k, v in inputs.items()}
```

- `return_tensors="pt"`：告诉 tokenizer 返回 `torch.Tensor` 而非 Python list。备选值：`"tf"`（TensorFlow）、`"np"`（numpy）
- `tokenizer()` 的返回值是一个字典，包含 `input_ids`（token ID 序列）、`attention_mask`（标注哪些位置是真实 token，哪些是 padding）等

```python
with torch.no_grad():
    outputs = model(**inputs)
    hidden = outputs.last_hidden_state  # [1, L, 1280]
```

- `torch.no_grad()`：禁用梯度计算。推理阶段不需要反向传播，关闭梯度可节省约 50% 显存并加速计算
- `outputs.last_hidden_state`：ESM2 最后一层 Transformer 输出的 hidden state，形状 `[batch_size, seq_len, hidden_dim]`。这是每个 token 位置的上下文表示

```python
embedding = hidden[0, 1:-1, :].mean(dim=0)
```

**Mean Pooling 的细节：**
- `hidden[0]`：去掉 batch 维（batch_size=1），得到 `[L, 1280]`
- `[1:-1, :]`：去掉首位的 `[CLS]` token 和末位的 `[EOS]` token。这两个特殊 token 是 tokenizer 自动添加的标记，不对应任何氨基酸
- `.mean(dim=0)`：对所有氨基酸位置的 embedding 取平均，得到 `[1280]` 的序列级表示

**为什么用 mean pooling 而不是 `[CLS]` token：**
BERT 系列模型明确训练 `[CLS]` 为句子级表示（Next Sentence Prediction 任务），ESM2 没有类似的预训练目标，其 `[CLS]` token 不具有全局语义。Mean pooling 保留所有位置信息，实验表明更稳定。

---

### 4.3 `embed_all_datasets()` — 批量提取与缓存

```python
all_seqs = set()
for df in datasets.values():
    all_seqs.update(df["sequence"].tolist())
```

跨数据集去重：某些序列出现在多个 benchmark 中（例如同一亲本抗体），去重后只计算一次，节省时间。

```python
df["embedding"] = df["sequence"].map(seq_to_emb)
```

- `Series.map(dict)`：将 Series 中每个值替换为 dict 中对应的值。这里把序列字符串映射为对应的 numpy array。比 `apply(lambda x: seq_to_emb[x])` 更快

```python
with open(pkl_path, "wb") as f:
    pickle.dump(embedded, f)
```

- `pickle.dump`：将 Python 对象序列化（序列化 = 转成二进制字节流）并写入文件。优点：可以存储任意 Python 对象（包括含 numpy array 的 dict）。缺点：格式不跨语言，且大对象文件较大
- `"wb"` 模式：以二进制写入模式打开，pickle 必须用二进制模式

---

## 五、dataset.py

### 5.1 `PairwiseRankingDataset` — 训练集

**核心思路：** 将排序问题转化为二分类。构造所有满足 `fitness_A > fitness_B` 的有序对 (A, B)，模型训练目标是给 A 打出比 B 更高的分数。

```python
valid_pairs = [
    (i, j)
    for i in range(N)
    for j in range(N)
    if i != j and fitness[i] > fitness[j]
]
```

**O(N²) 问题：** N 条序列理论上有 N(N-1)/2 个有效对。N=3000 时约 450 万对，无法全部放入内存训练。

```python
if len(valid_pairs) > max_pairs:
    rng = np.random.default_rng(42)
    indices = rng.choice(len(valid_pairs), size=max_pairs, replace=False)
    valid_pairs = [valid_pairs[k] for k in indices]
```

- `np.random.default_rng(42)`：创建一个新式随机数生成器（Generator），种子为 42。新式 API 比 `np.random.seed()` 更安全（线程独立）
- `rng.choice(n, size=k, replace=False)`：从 0~n-1 中无放回地随机抽取 k 个整数索引。`replace=False` 保证不重复采样

**`__getitem__` 的返回格式：**
```python
return (
    torch.tensor(emb_pos),   # [1280]，正样本 embedding
    torch.tensor(emb_neg),   # [1280]，负样本 embedding
    torch.tensor(fit_pos),   # 标量，正样本 fitness（MSE loss 需要）
    torch.tensor(fit_neg),   # 标量，负样本 fitness（MSE loss 需要）
)
```

Hinge 和 RankNet 只用前两个，MSE 需要后两个。三种 loss 共享同一套 DataLoader，所以返回四个值，接口统一。

---

### 5.2 `ScoringDataset` — 验证/测试集

逐条序列评分，不构造对。用于计算 Spearman 相关系数：

```python
return (
    torch.tensor(self.embeddings[idx]),  # [1280]
    torch.tensor(self.fitness[idx]),     # 标量
)
```

---

## 六、model.py — `AffinityMLP`

### 架构

```
输入 [batch, 1280]
    │
Linear(1280, 256)   ← Xavier 初始化
    │
GELU()              ← 非线性激活
    │
Dropout(0.2)        ← 训练时随机关闭 20% 神经元
    │
Linear(256, 1)      ← 输出单个排序分数
    │
输出 [batch, 1]
```

### Xavier 初始化

```python
nn.init.xavier_uniform_(m.weight)
nn.init.zeros_(m.bias)
```

- `xavier_uniform_`：将权重初始化为均匀分布 U(-a, a)，其中 a 根据输入输出维度自动计算。目的：使每层输出的方差与输入相同，防止梯度消失/爆炸
- 数学依据：对于 Linear(fan_in, fan_out)，a = sqrt(6 / (fan_in + fan_out))
- `zeros_`：将 bias 初始化为 0，标准做法

### 为什么用 GELU 而非 ReLU

GELU（Gaussian Error Linear Unit）：`GELU(x) = x · Φ(x)`，其中 Φ 是标准正态分布的 CDF。与 ReLU 不同，GELU 在 x < 0 时不是硬截断为 0，而是平滑过渡。ESM2 内部使用 GELU，保持激活函数一致性有助于梯度传播。

### 为什么只有两层

每个 benchmark 的训练数据只有 30~4000 条，深层网络极易过拟合。两层 MLP 参数量 = 1280×256 + 256×1 = 328,192，对几百条数据来说已经不小。

---

## 七、losses.py

### 三种损失的接口统一性

所有 loss 的 `forward` 签名相同：
```python
def forward(self, score_pos, score_neg, fitness_pos=None, fitness_neg=None)
```

Hinge/RankNet 不使用 fitness（占位参数），MSE 使用全部四个参数。统一接口使 trainer.py 可以在不修改训练循环的情况下切换 loss。

### 7.1 MSE Loss

```python
self.mse = nn.MSELoss(reduction="mean")
loss = (self.mse(score_pos, fitness_pos) + self.mse(score_neg, fitness_neg)) / 2
```

- `nn.MSELoss(reduction="mean")`：计算 batch 内所有 (预测值, 真实值) 对的 (pred-target)² 的平均
- **主要缺陷**：fitness 量纲不统一（有的是 -logKd ∈ [6,12]，有的是 -log EC50 ∈ [0,3]），MSE 会被量纲大的数据集主导。由于本项目是 per-benchmark 训练，不同数据集间量纲不混合，但同一数据集内 MSE 的绝对值仍与 fitness 量纲相关（见训练日志中的 loss 量级差异）

### 7.2 Pairwise Hinge Loss

```python
diff = score_pos - score_neg
loss = torch.clamp(self.margin - diff, min=0.0)
```

- `torch.clamp(x, min=0.0)`：将 x 中所有小于 0 的值截断为 0，等价于 `max(x, 0)`。实现 hinge 函数：当差距 > margin 时 loss=0，否则线性惩罚
- **与 Ranking SVM 的关系**：Ranking SVM 的完整目标函数是 `(1/2)||w||² + C·Σmax(0, 1-diff)`，包含结构风险项 `||w||²`。本实现只有 hinge loss 项，用 AdamW 的 `weight_decay` 替代结构风险。严格来说应称为 "Pairwise Hinge Loss" 而非 Ranking SVM

### 7.3 RankNet Loss（Bradley-Terry 模型）

```python
loss = F.softplus(score_neg - score_pos)
```

- `F.softplus(x) = log(1 + exp(x))`：softplus 是 `log(1+exp(x))` 的数值稳定实现，等价于 `-log(sigmoid(-x))`
- **数学推导**：
  - 目标：最大化 P(A≻B) = σ(score_A - score_B) 的对数
  - 损失 = -log σ(score_A - score_B)
  - = log(1 + exp(-(score_A - score_B)))
  - = softplus(score_B - score_A)  ✓
- **Bradley-Terry 模型假设**：每个序列有一个真实潜在 fitness，观测到的排序是 "fitness 加 Gumbel 噪声后谁更大"。Gumbel 分布对应 softmax 操作，这也是 BT 模型与 logistic 回归的内在联系

---

## 八、trainer.py

### 8.1 `split_dataset(df)` — 分层划分

```python
fitness_bins = pd.qcut(df["fitness"], q=min(5, len(df)//3), labels=False, duplicates="drop")
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=fitness_bins, random_state=42)
```

- `pd.qcut(col, q=5)`：将连续 fitness 值按分位数分成 5 个等频 bin，用于 stratified split 的分层标签。确保 train/val/test 的 fitness 分布均匀，避免 train 全是低亲和力序列而 test 全是高亲和力序列
- `train_test_split(stratify=bins)`：sklearn 的分层随机划分，保证每个 bin 在各子集中的比例相近
- `duplicates="drop"`：当 fitness 值重复太多时（同一值的行超过 q 等份），`pd.qcut` 可能无法创建 q 个不同分位边界，`drop` 参数允许合并边界相同的 bin

**降级策略：** 若分层划分失败（数据量太少或 fitness 值不够多样），退回到随机划分。

### 8.2 `evaluate_spearman(model, dataset)` — Spearman 计算

```python
model.eval()  # ← 关键：关闭 Dropout
with torch.no_grad():
    scores = model(emb.to(cfg.DEVICE)).squeeze(-1)
corr, pval = spearmanr(all_scores, all_fitness)
```

- `model.eval()`：必须在评估前调用，否则 Dropout 会随机丢弃神经元，导致同一序列每次得分不同，Spearman 不稳定
- `.squeeze(-1)`：去掉最后一维（模型输出是 `[batch, 1]`，squeeze 后变为 `[batch]`），便于 `spearmanr` 接收
- `spearmanr(a, b)`：scipy 函数，计算两个数组的 Spearman 秩相关系数。只考虑排序而非绝对值，与比赛评分指标完全一致。返回 (correlation, p_value)

### 8.3 `train_one_benchmark(name, df, output_dir, loss_name)` — 核心训练循环

#### 优化器和调度器

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg.LR,
    weight_decay=cfg.WEIGHT_DECAY,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)
```

- `AdamW`：Adam 优化器 + 解耦 weight decay。普通 Adam 的 L2 正则化和动量更新耦合在一起（有数学问题），AdamW 将 weight decay 从梯度更新中独立出来，是目前深度学习的标准选择
- `CosineAnnealingLR(T_max=EPOCHS)`：学习率按余弦函数从初始值衰减到接近 0。余弦退火在训练后期平滑降低学习率，帮助模型在损失曲面的平坦区域收敛，避免在局部极小值附近震荡

#### 训练步骤

```python
optimizer.zero_grad()                           # 1. 清空上一步的梯度
score_pos = model(emb_pos)                      # 2. 正向传播
loss = loss_fn(score_pos, score_neg, ...)       # 3. 计算损失
loss.backward()                                  # 4. 反向传播（计算梯度）
nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 5. 梯度裁剪
optimizer.step()                                 # 6. 更新参数
scheduler.step()                                 # 7. 更新学习率
```

- `optimizer.zero_grad()`：PyTorch 默认**累积**梯度（每次 backward 将梯度加到已有梯度上），每步都必须先清零，否则梯度会被累加多倍
- `loss.backward()`：自动微分，从 loss 开始反向传播，计算所有 `requires_grad=True` 参数的梯度并存储在 `.grad` 属性中
- `clip_grad_norm_(model.parameters(), max_norm=1.0)`：将所有参数梯度的 L2 范数裁剪到 1.0 以内。防止梯度爆炸（梯度突然变得很大导致参数更新步长过大，模型发散）

#### Early Stopping（简化版）

```python
if val_spearman > best_val_spearman:
    best_val_spearman = val_spearman
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
```

- `model.state_dict()`：返回模型所有参数的 dict（key = 参数名，value = tensor）
- `.clone()`：深拷贝 tensor。如果不 clone，直接存 `v` 只是存了引用，后续训练会修改这些 tensor，导致"保存"的参数实际上跟着改变
- 最终用 `model.load_state_dict(best_state)` 恢复验证集最优参数

---

## 九、已知问题与待修复项

| 问题 | 严重程度 | 位置 | 说明 |
|------|----------|------|------|
| fitness 方向自动检测不完整 | 高 | `data_loader.py` 步骤6 | 窄范围 Kd 数据集（span<5）无法被检测，方向仍然错误。建议维护显式黑名单 |
| `makowski_igg_ant` 被误判为需修正 | 中 | `data_loader.py` 步骤6 | span=21 触发修正，但该数据集 fitness 是归一化结合得分，已是"高=好"方向 |
| kothiwal SPR 系列方向未修正 | 高 | `data_loader.py` 步骤6 | `kothiwal_*_spr` 的 SPR Kd 范围窄，未被检测。所有 SPR Kd 数据均应修正 |
| `shanker_SA58-XBB_Kd` 含异常值 | 低 | 数据质量 | fitness 最大值 7.64e19，显然是数据录入错误，应过滤 |
| MSE loss 对 fitness 量纲敏感 | 中 | `losses.py` | 不同数据集 fitness 绝对值差异大，MSE loss 量级不稳定（训练日志中可见 loss 从 0.1 到 30+ 变化） |
