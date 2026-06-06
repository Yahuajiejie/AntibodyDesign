# affinity_model v3 技术方案

## 一、v3 目标

v3 暂时只写方案，不写代码。v3 的目标是把模型从 antibody-only 升级为 antigen-context model：

```text
score = f(antibody, antigen_context)
```

也就是：每条抗体不再只根据自身序列打分，而是根据“它面对哪个抗原”打分。

v3 不替代 v2.1。v2.1 应继续保留为干净的 antibody-only baseline：

```text
v2.1 = heavy/light antibody-only baseline
v3   = antigen-aware / MSA-aware model
```

这样后续实验才能明确回答：

```text
抗原 embedding 是否真的提升了排序？
MSA-aware antigen embedding 是否比 single-sequence antigen embedding 更好？
```

## 二、数据现状

已统计到的数据覆盖情况：

```text
FLAb binding 全部：
  总条数 4,606,793
  至少有有效抗原名 4,604,645
  直接有抗原序列 343,065

FLAb 真实 Kd 子集：
  总条数 88,785
  至少有有效抗原名 86,637
  直接有抗原序列 364

proteinbase:
  总条数 5,253
  有 target name 5,248
  Nipah target 3,754
  Nipah experimental 1,201
  Nipah Kd 102

ANDD:
  总条数 30,333
  Ag_Name 30,333
  Ag_Seq 30,333
  Affinity_Kd 1,735

SAbDab summary:
  总条数 20,700
  有 antigen_name 16,333
  protein antigen 14,113
  有 affinity 1,307
```

结论：

- 抗原名字覆盖很高；
- 抗原序列直接覆盖很低；
- 要做 antigen embedding，必须先做 `antigen_registry.csv`；
- 对没有抗原序列的样本，不能假装有 embedding，必须记录来源和置信度。

## 三、v3 数据资产

### `antigen_registry.csv`

这是 v3 最关键的数据表。没有这个表，不允许训练 antigen-aware 模型。

字段设计：

```text
compatible_group
dataset
antigen_name
antigen_type
antigen_sequence
sequence_source
sequence_accession
sequence_confidence
has_antigen_sequence
is_protein
is_glycoprotein
is_peptide
is_small_molecule
is_carbohydrate
ligand_smiles
glycan_info
msa_source
msa_cache_path
notes
```

字段解释：

- `compatible_group`：和训练数据中的 group 对齐；
- `antigen_name`：最少必须存在的抗原有效名字；
- `antigen_type`：protein / glycoprotein / peptide / small_molecule / carbohydrate / unknown；
- `antigen_sequence`：蛋白或肽抗原序列；
- `sequence_source`：CSV / TASKS / UniProt / PDB / paper / manual；
- `sequence_confidence`：high / medium / low；
- `ligand_smiles`：小分子抗原使用；
- `glycan_info`：糖链或糖基化信息使用；
- `msa_cache_path`：该抗原对应的 A3M/Stockholm 文件路径。

质检点：

- 同一个 `compatible_group` 只能映射到一个主抗原；
- 小分子不能写进 `antigen_sequence` 假装是蛋白；
- `sequence_source` 不能为空；
- 低置信度序列不能默认参与主实验。

### `antigen_embedding_cache`

建议缓存为：

```text
cache/antigen_embeddings/
  single_esm2/{antigen_id}.npy
  msa_esm1b/{antigen_id}.npy
  ligand/{antigen_id}.npy
  glycan/{antigen_id}.npy
```

每个 cache 文件都应有配套 manifest：

```text
antigen_id
embedding_type
model_name
embedding_dim
source_sequence_hash
created_at
```

## 四、v3 预处理流程

### Step 1：识别抗原

输入：

```text
FLAb/data/binding/*.csv
FLAb/data/flab_metadata.csv
TASKS.md
proteinbase_all_data_28_01_2026.csv
ANDD.xlsx
SAbDab summary / PDB
```

输出：

```text
antigen_registry.csv
```

实现策略：

```text
1. 优先读取行内字段：Ag_name / Antigen / Target / Ag_label；
2. 再读取行内序列字段：Ag_seq / antigen_seq / Ag_Seq；
3. 如果行内没有，使用 metadata 或文件名推断 dataset-level antigen_name；
4. 对每个 compatible_group 生成唯一 antigen_id；
5. 标记是否有可靠 antigen_sequence。
```

质检点：

- 只从名字推断抗原时，`sequence_confidence` 不能写 high；
- fluorescein 等半抗原必须标记为 small_molecule；
- Nipah G protein 使用 TASKS 中给出的序列，置信度 high。

### Step 2：补充抗原序列

输入：

```text
antigen_registry.csv
```

输出：

```text
antigen_registry.filled.csv
```

实现策略：

```text
1. CSV/TASKS/ANDD 中已有序列：直接使用；
2. PDB/SAbDab 有结构：从 antigen_chain 提取序列；
3. 只有抗原名字：人工或脚本查询 UniProt / NCBI / PDB；
4. 小分子：补 ligand_smiles，不补 antigen_sequence；
5. 糖抗原：补 glycan_info，不补蛋白序列。
```

质检点：

- 补序列必须记录 accession；
- 多 isoform 时不能静默选一个，要写 notes；
- 低置信度条目可进入 exploratory ablation，但不能进入主模型默认训练。

### Step 3：同源序列搜索

输入：

```text
antigen_sequence
```

输出：

```text
homologs/{antigen_id}.fasta
```

候选工具：

```text
BLAST
MMseqs2
HHblits
ColabFold search
```

候选数据库：

```text
UniRef
UniProt
PDB
NCBI NR
```

实现策略：

```text
1. 对 protein/glycoprotein/peptide 抗原做 homolog search；
2. 过滤 identity < 30% 的远缘序列；
3. 去重；
4. 限制 homolog 数，例如 64 / 128 / 256；
5. 保存 FASTA 和搜索日志。
```

质检点：

- 数据库很大，不能假设训练时联网；
- homolog search 应是离线预处理；
- 训练阶段只读 cache；
- search 参数必须进入技术文档。

### Step 4：构建 MSA

输入：

```text
原始 antigen_sequence
homologs/{antigen_id}.fasta
```

输出：

```text
msa/{antigen_id}.a3m
```

候选工具：

```text
MAFFT
MUSCLE
HHblits
ColabFold
```

建议格式：

```text
A3M
```

原因：

```text
ESM-MSA-1b 常用 MSA 输入格式是 A3M/FASTA-like MSA。
```

实现策略：

```text
1. 把 query antigen 放在第一条；
2. 加入过滤后的 homolog；
3. MSA 数量不足时仍保存，但标记 has_msa=False 或 msa_depth_low；
4. 对超深 MSA 采样到固定深度；
5. 保存 MSA 质量统计。
```

质检点：

- query 序列必须能从 MSA 第一行还原；
- MSA depth 太低不能假装有高质量进化信息；
- MSA cache 需要 hash，避免序列更新后读旧文件。

### Step 5：MSA-aware embedding

输入：

```text
msa/{antigen_id}.a3m
```

输出：

```text
cache/antigen_embeddings/msa_esm1b/{antigen_id}.npy
```

模型：

```text
ESM-MSA-1b
```

注意：

```text
ESM-MSA-1b 的 embedding 维度是 768，不是 ESM2-650M 的 1280。
```

实现策略：

```text
1. 读取 MSA；
2. 限制 MSA depth 和序列长度；
3. 输入 ESM-MSA-1b；
4. 对 query 序列位置做 mean pooling；
5. 得到 antigen_msa_embedding [768]；
6. 保存到 cache。
```

质检点：

- 记录模型名和 embedding_dim；
- MSA 为空或质量低时不能生成伪 MSA embedding；
- 训练时不重复跑 ESM-MSA-1b。

### Step 6：单序列抗原 embedding

输入：

```text
antigen_sequence
```

输出：

```text
cache/antigen_embeddings/single_esm2/{antigen_id}.npy
```

模型：

```text
ESM2-650M
```

实现策略：

```text
1. 使用和抗体相同的 ESM2；
2. 对 antigen sequence 做 mean pooling；
3. 得到 antigen_single_embedding [1280]；
4. 保存到 cache。
```

质检点：

- 单序列 embedding 和 MSA embedding 分开缓存；
- 不同抗原共享同一 ESM2 模型；
- 超长抗原需要明确截断策略或分块策略。

## 五、v3 模型结构

### v3.0：Antigen Single Embedding

第一版只加入单序列抗原 embedding，不做 MSA。

输入：

```text
heavy_embedding          [1280]
light_embedding          [1280]
antigen_single_embedding [1280]
antigen_type_embedding   [D_type]
flags                    [D_flag]
```

模型：

```text
feature = concat(heavy, light, antigen_single, antigen_type, flags)
score = MLP(feature)
```

目标：

```text
验证“抗原信息本身”是否提升 Spearman。
```

### v3.1：MSA-aware Antigen Embedding

第二版加入 MSA embedding。

输入：

```text
heavy_embedding          [1280]
light_embedding          [1280]
antigen_single_embedding [1280]
antigen_msa_embedding    [768]
antigen_type_embedding   [D_type]
flags                    [D_flag]
```

模型：

```text
antigen_context = Projection(
    concat(antigen_single, antigen_msa, antigen_type, flags)
)
feature = concat(heavy, light, antigen_context)
score = MLP(feature)
```

质检点：

- `antigen_msa_embedding` 不能被当成 1280 维；
- 投影层必须记录输入维度；
- 对没有 MSA 的抗原，使用 mask/flag，而不是静默零向量。

### v3.2：糖蛋白、小分子和糖抗原分支

糖蛋白：

```text
antigen_sequence -> ESM2 / ESM-MSA
glycosylation flags -> motif count / known glycosite mask / is_glycoprotein
```

小分子：

```text
ligand_smiles -> Morgan fingerprint 或 ChemBERTa embedding
antigen_type = small_molecule
```

糖抗原：

```text
glycan_info -> glycan fingerprint / learned glycan token
antigen_type = carbohydrate
```

第一版不要把糖分支做得太复杂。最低要求：

```text
有 type embedding
有 flags
有 notes 记录为什么不能用 protein embedding
```

### v3.3：Token-level Antibody-Antigen Attention

这是后续高风险版本，不建议作为 v3 第一版。

输入：

```text
antibody token embeddings
antigen token embeddings
```

结构：

```text
antibody_tokens query
antigen_tokens key/value
cross_attention
pool
score
```

建议限制：

```text
只让 CDR token attend antigen token
```

原因：

- 全长 token attention 显存更高；
- 非 CDR 区域噪声更大；
- CDR-attention 更容易解释。

## 六、v3 训练与评估

### 训练目标

RankNet/Hinge 逻辑保持不变：

```text
score_pos = f(pos_antibody, antigen_context)
score_neg = f(neg_antibody, antigen_context)
loss = softplus(score_neg - score_pos)
```

也就是说：

- 不改 pair 方向；
- 不改 RankNet 公式；
- 不改 compatible_group 内构造 pair 的原则；
- 只改变 `score = f(...)` 的输入。

### split 策略

v3 必须新增一个评估：

```text
antigen-held-out split
```

目的：

```text
检查模型是否真的学会“抗原上下文”，而不是只记住某些 group。
```

建议同时保留：

```text
group split
antigen-held-out split
Nipah-specific split
```

### 实验矩阵

建议最小实验：

```text
v2.1 chain_concat baseline
v3.0 + antigen_single_embedding
v3.1 + antigen_single_embedding + antigen_msa_embedding
v3.2 + antigen type / glyco / ligand flags
```

每次只加一个变量。

主指标：

```text
val_weighted_spearman
test_weighted_spearman
test_median_spearman
antigen_heldout_weighted_spearman
```

## 七、v3 计划新增模块

### `antigen_registry.py`

计划函数：

```text
build_antigen_registry(...)
load_antigen_registry(...)
validate_antigen_registry(...)
```

职责：

```text
从 CSV、metadata、TASKS、proteinbase、ANDD、SAbDab 中建立抗原索引。
```

### `homolog_search.py`

计划函数：

```text
run_homolog_search(...)
filter_homologs(...)
write_homolog_fasta(...)
```

职责：

```text
为 protein/glycoprotein/peptide 抗原搜索同源序列。
```

### `msa_builder.py`

计划函数：

```text
build_msa(...)
validate_msa(...)
sample_msa_depth(...)
```

职责：

```text
构建和质检 MSA cache。
```

### `antigen_embeddings.py`

计划函数：

```text
embed_antigen_single(...)
embed_antigen_msa(...)
embed_ligand(...)
load_antigen_embedding_cache(...)
```

职责：

```text
生成并读取抗原相关 embedding。
```

### `antigen_context_dataset.py`

计划函数：

```text
build_antigen_context_feature_matrix(...)
```

职责：

```text
把 antibody embedding 和 antigen context embedding 拼成 v3 模型输入。
```

### `antigen_context_model.py`

计划类：

```text
AntigenContextMLP
AntigenContextProjector
```

职责：

```text
对 antigen_single / antigen_msa / type / flags 做投影，再和 antibody feature 融合打分。
```

## 八、v3 风险

主要风险：

- 抗原序列补错，导致模型学到错误上下文；
- MSA 搜索数据库过大，集群环境难以复现；
- MSA 质量不均匀，有些抗原有深 MSA，有些只有 query；
- 小分子/糖抗原不能用蛋白 embedding；
- v3 变量太多，容易无法解释提升来自哪里。

控制方法：

- 先做 `antigen_registry.csv`，人工质检；
- 所有补充序列记录来源和置信度；
- 训练只读 cache，不在线搜索；
- 实验逐步加模块；
- 保留 v2.1 作为 baseline。

## 九、v3 暂不实现的部分

当前阶段只写技术方案，不写代码。

暂不实现：

- homolog search 脚本；
- MSA 构建脚本；
- ESM-MSA-1b embedding；
- antigen-context model；
- token-level antibody-antigen attention；
- glycan graph embedding；
- ligand embedding。

下一步如果进入 v3 实现，应先从 `antigen_registry.csv` 开始，而不是直接写模型。
