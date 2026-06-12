# 数据预处理脚本说明

本目录包含将原始抗体亲和力数据集转换为统一训练表格格式的全套脚本。
适用于 AffinityTransformer 的 **binding（结合亲和力）** 类别，共 86 个数据集（84 个 ready，2 个 blocked）。

---

## 目录结构

```
scripts/prepare/
├── validate_processed_table.py          # 通用 Schema 校验器
└── binding/
    ├── prepare_all.sh                   # 一键运行所有 ready 数据集
    ├── phillips2021binding/
    │   ├── cr6261_h1_kd/
    │   │   ├── convert.py              # 数据转换脚本（每个数据集一份）
    │   │   └── prepare.sh             # 单数据集运行入口
    │   └── ...（其余子表同结构）
    ├── hie2023efficient/
    │   └── CoV2_S309_Kd/
    │       ├── convert.py
    │       └── prepare.sh
    └── ...（共 84 个 ready 数据集）

processed/binding/
├── manifest.csv                         # 全部 86 个数据集总表（含状态）
├── antigen_missing_summary.csv          # 抗原序列缺失查找表
└── {study_id}/{table_id}/
    ├── records.parquet                  # 主输出（列式存储，供训练使用）
    └── records.csv                      # 同上（人类可读版本）
```

---

## 总表：manifest.csv

**位置**：`processed/binding/manifest.csv`

记录全部 86 个 binding 数据集的基本信息与处理状态，字段如下：

| 字段 | 说明 |
|------|------|
| `study_id` | 研究标识，如 `phillips2021binding` |
| `table_id` | 子表标识，如 `cr6261_h1_kd` |
| `csv_name` | 原始数据文件名 |
| `antibody_type` | 抗体格式：`Fv` / `Fab` / `IgG` / `scFv` / `VHH` |
| `antigen_key` | 抗原标准化 key，用于构造 `group_id` |
| `antigen_name` | 抗原全名 |
| `antigen_source` | `provided`（CSV 中有序列）/ `missing`（待补充） |
| `metric_name` | 度量指标，如 `neg_log10_kd_M` |
| `label_kind` | `experimental` / `predicted` / `binary` |
| `status` | **ready**（脚本已就绪）/ **planned**（待实现）/ **blocked**（有阻断问题） |
| `notes` | 特殊说明 |

当前状态汇总：

- **ready（84 个）**：脚本完整，可直接运行（含原有 39 个 + 新增 45 个）
- **blocked（2 个）**：`rawat2022abcov` — CSV 中无抗原列，无法构造有效 `group_id`，需先从论文中找到抗原信息

---

## 抗原序列缺失查找表：antigen_missing_summary.csv

**位置**：`processed/binding/antigen_missing_summary.csv`

对 65 个抗原序列未包含在 CSV 中的数据集，提供逐一的检索建议：

| 字段 | 说明 |
|------|------|
| `antigen_key` | 标准化 key |
| `antigen_name` | 抗原全名 |
| `is_protein` | 是否为蛋白质（`False` 表示小分子如荧光素，无序列） |
| `likely_uniprot_or_pdb` | 推荐 UniProt 编号或 PDB ID |
| `antigen_species` | 物种来源 |
| `retrieval_notes` | 具体检索建议（用哪条序列，注意哪些剪接/信号肽等） |

**典型示例**：

```
hutchinson2023enhancement / HEL → UniProt P00698（鸡溶菌酶，全长成熟序列）
shanehsazzadeh2024igdesign / IL17A → UniProt Q16552（人 IL-17A 成熟形式）
hie2023efficient / CoV2_WT_S6P → PDB 7LYL（S6P 预稳定化六脯氨酸突变体）
```

补充抗原序列后，在对应 `convert.py` 中将：
```python
ANTIGEN_SEQ    = None
ANTIGEN_SOURCE = "missing"
```
改为：
```python
ANTIGEN_SEQ    = "KVFERCELART..."   # 实际序列
ANTIGEN_SOURCE = "retrieved"
```

---

## 快速开始

### 环境要求

```bash
pip install pandas pyarrow
```

### 首次运行前（一次性）：允许序列中保留 X 残基

```bash
bash scripts/prepare/binding/patch_allow_X.sh
```

这会将所有 84 个 convert.py 中的 `_VALID_AA` 从 20 种标准氨基酸改为 20+X，使含 X 的序列不再被置 null，而是直接传给 ESMC 处理。validate_processed_table.py 已同步更新。

### 运行所有 ready 数据集（推荐）

在仓库根目录执行：

```bash
bash scripts/prepare/binding/prepare_all.sh
```

脚本启动时会自动：
1. 运行 `gen_manifest.py` → 写出 `scripts/prepare/binding/manifest.csv` 和 `antigen_missing_summary.csv`
2. 将这两个文件复制到 `processed/binding/`

因此每次运行结束后，`processed/binding/` 中都会同时包含元数据文件和各数据集的 `records.parquet`。

加日志：

```bash
bash scripts/prepare/binding/prepare_all.sh 2>&1 | tee prepare_all.log
```

### 运行单个数据集

```bash
bash scripts/prepare/binding/hie2023efficient/CoV2_S309_Kd/prepare.sh
```

### 单独验证输出

```bash
python3 scripts/prepare/validate_processed_table.py \
    processed/binding/hie2023efficient/CoV2_S309_Kd/records.parquet
```

---

## 运行 prepare_all.sh 会得到什么

### 输出位置

```
processed/binding/
├── manifest.csv               ← 全部 86 个数据集总表（含状态）
├── antigen_missing_summary.csv← 抗原序列缺失查找表
├── phillips2021binding/
│   ├── cr6261_h1_kd/
│   │   ├── records.parquet    ← 主输出（列式存储，供训练使用）
│   │   └── records.csv        ← 同上（人类可读版本）
│   ├── cr6261_h9_kd/
│   └── ...
├── hie2023efficient/
│   └── CoV2_S309_Kd/
│       ├── records.parquet
│       └── records.csv
└── ...（共 84 个目录）
```

### 输出规模（实测 + 估算）

| 研究 | 子表数 | 大致记录数 |
|------|--------|-----------|
| phillips2021binding | 4 | ~3,800 |
| hutchinson2023enhancement | 8 | ~378 |
| hie2023efficient | 9 | ~146 |
| shanehsazzadeh2024igdesign | 7 | ~580 |
| rosace2023automated | 2 | ~73 |
| kothiwal2025htp | 20 | ~3,600 |
| shanker2024unsupervised | 7 | ~700 |
| adams2017measuring | 2 | ~200 |
| makowski2022cooptimization | 4 | ~1,600 |
| peterson2024integrated | 2 | ~500 |
| kirby2024retrospective | 2 | ~400 |
| cognano / AVIDa-hTNFa | 1 | ~数千 |
| tsuruta2024sarscov2 / binary | 1 | **77,003** |
| tsuruta2024avida / hIL6_binary | 1 | **573,891** |
| engelhart2022dataset | 1 | **352,139** |
| li2023machine / affinity1+2 | 2 | ~数百万（流式写入） |
| AbRank / dataset | 1 | **169,531**（Kd + IC50 双记录） |
| 其余 ready 数据集 | 15 | ~15,000+ |

### 输出字段（records.parquet / records.csv）

每行代表一条抗体–抗原亲和力记录：

| 字段 | 类型 | 说明 |
|------|------|------|
| `record_id` | str | 全局唯一 ID：`{study_id}/{table_id}/{source_row}` |
| `dataset_id` | str | `{study_id}/{table_id}` |
| `source_file` | str | 原始 CSV 相对路径 |
| `source_row` | int | 在原始 CSV 中的行号（从 2 开始，1 为表头） |
| `antibody_type` | str | `Fv` / `Fab` / `IgG` / `scFv` / `VHH` / `unknown` |
| `heavy_chain` | str\|null | 重链氨基酸序列（大写，标准 20 种 AA） |
| `light_chain` | str\|null | 轻链序列；含 X 等非标准字符时置 null |
| `single_chain_sequence` | str\|null | 单链格式（scFv/VHH 可用） |
| `antigen_key` | str | 抗原标准化 key |
| `antigen_name` | str | 抗原全名 |
| `antigen_sequence` | str\|null | 抗原序列；当前多为 null，待后续补充 |
| `antigen_source` | str | `provided` / `retrieved` / `missing` |
| `assay_name` | str | 实验方法（如 `SPR`、`Octet BLI`） |
| `metric_name` | str | 指标名（如 `neg_log10_kd_M`、`neg_log10_kd_nM`） |
| `metric_value_raw` | str | 原始值（字符串，保留来源信息） |
| `metric_value_numeric` | float\|null | 原始值的浮点解析 |
| `metric_unit` | str | 单位说明（如 `-log10(KD/M)`） |
| `metric_direction` | str | `higher_is_better`（所有 ready 数据集均已转换） |
| `transform_rule` | str | 变换说明（如何从原始值得到 `rank_label`） |
| `rank_label` | float\|null | **训练标签**；始终为越大越好；null 时该行被丢弃 |
| `label_kind` | str | `experimental` / `predicted` / `binary` |
| `group_id` | str | 配对比较组 ID（同 group 内的抗体才能互相排名） |
| `keep_for_training` | bool | 是否纳入训练 |
| `drop_reason` | str\|null | 丢弃原因（如 `missing_or_invalid_heavy_chain`） |

### 控制台输出示例

```
══════════════════════════════════════════════════
  hie2023efficient/CoV2_S309_Kd
══════════════════════════════════════════════════
[09:41:03] Converting data/binding/hie2023efficient_CoV2_S309_Kd.csv ...
[hie2023efficient/CoV2_S309_Kd]  total=20  keep=20  drop=0
  -> processed/binding/hie2023efficient/CoV2_S309_Kd/records.parquet
[09:41:03] Validating schema ...
  rows=20  keep=20  drop=0
PASS  processed/binding/hie2023efficient/CoV2_S309_Kd/records.parquet
Done: processed/binding/hie2023efficient/CoV2_S309_Kd

══════════════════════════════════════════════════
  SUMMARY:  pass=84  fail=0
══════════════════════════════════════════════════
```

---

## 脚本实现思路

### convert.py（每个数据集一份）

每个 `convert.py` 分为两层：

**配置层（文件顶部的常量块）**

```python
STUDY_ID       = "hie2023efficient"
TABLE_ID       = "CoV2_S309_Kd"
SOURCE_FILE    = "data/binding/hie2023efficient_CoV2_S309_Kd.csv"
ANTIBODY_TYPE  = "IgG"
ANTIGEN_KEY    = "CoV2_WT_S6P"
ANTIGEN_NAME   = "SARS-CoV-2 WT Spike S6P"
ANTIGEN_SEQ    = None               # 待补充
ANTIGEN_SOURCE = "missing"
METRIC_NAME    = "neg_log10_kd_M"
TRANSFORM_RULE = "fitness = neg_log_Kd = -log10(KD/M)"
FITNESS_COL    = "fitness"
```

与底层处理逻辑完全分离，修改数据集参数只需动配置层。

**处理层（`_rl` 函数 + `convert` 函数）**

`_rl(raw)` 函数负责将原始值转换为 `rank_label`（始终越大越好）：

```python
# 默认：fitness 列已是 rank_label
def _rl(raw) -> float:
    return float(raw)

# 特殊情况：fitness 是原始 KD(nM)，需要转换方向
# （仅 shanehsazzadeh2024igdesign 7 个数据集）
def _rl(raw) -> float:
    return -math.log10(float(raw) * 1e-9)
```

`convert` 函数处理每一行：
1. 用 `_seq(v)` 净化序列（全大写，含 X 等非标准 AA 则返回 None）
2. 计算 `rank_label`（非有限值或异常则为 None）
3. 决定 `keep_for_training`：重链为 None 或 rank_label 为 None 时丢弃
4. 轻链含 X 时置 null 但**不丢弃**（phillips 等 DMS 数据集的亲本轻链含固定 X 位点，重链才是变异区）
5. 写出 `records.parquet` 和 `records.csv`

### validate_processed_table.py

对输出文件做 8 项检查（任意失败则 exit 1）：

1. 必需列全部存在（27 列）
2. `record_id` 非空且唯一
3. `group_id` 非空
4. `keep_for_training=True` 的行 `rank_label` 必须为有限浮点数
5. `keep_for_training=True` 的行 `heavy_chain` 非空
6. 序列中仅含标准 20 种氨基酸
7. 枚举字段合法（`antibody_type`、`antigen_source`、`assay_type` 等）
8. `source_row >= 2`

### group_id 设计

`group_id` 决定哪些记录可以互相配对做排名训练：

```
{study_id}/{table_id}/{antigen_key}/{metric_name}/{label_kind}
```

例如：`hie2023efficient/CoV2_S309_Kd/CoV2_WT_S6P/neg_log10_kd_M/experimental`

同一 `group_id` 内的抗体才会被采样为训练 pair（RankNet 正负对），跨 group 不会配对，因此不同抗原、不同度量单位的数据天然隔离。

---

## 注意事项

### 关于序列中的 X 残基（phillips 轻链等）

IUPAC 中 X 代表"任意氨基酸"，在 DMS 数据集的亲本序列中偶尔出现（例如 phillips2021binding 四张子表的轻链在固定位点含 X）。

**当前处理方式：保留 X，不置 null**

- `_VALID_AA` 包含 `X`（需先运行 `patch_allow_X.sh`）
- 含 X 的序列正常写入 `heavy_chain`/`light_chain` 字段
- ESMC 有专属 X token，能对该位点生成"分布加权"表示
- 对 pairwise ranking 无影响：同组所有抗体共享同一亲本轻链，排名仅由重链驱动

若未来从原论文找到 X 位点的真实氨基酸，直接在对应 convert.py 的 `ANTIGEN_SEQ` 附近备注，或对数据做后处理替换即可。

### 关于 shanehsazzadeh2024igdesign 的单位错误

`flab_metadata.csv` 对 7 张 igdesign 子表的单位标注为 `-log(Kd[nM])`（错误），实际数据为**原始 KD (nM)（越小越好）**。`convert.py` 已做修正：

```python
rank_label = -log10(KD_nM × 1e-9)
```

### 脚本由人工运行，非自动化流水线

这批脚本是**离线批处理**脚本，最终训练数据由程序员手动运行获得，`processed/` 目录不纳入版本控制（`.gitignore`）。

---

## 特殊实现说明（新增 45 个数据集）

### 大文件流式处理（li2023machine、AbRank）

`li2023machine/affinity1`（~700 MB）、`affinity2`（~700 MB）、`AbRank/dataset`（~283 MB）采用分块流式写入，避免内存溢出：

```python
# pd.read_csv 分块迭代 + pyarrow.parquet.ParquetWriter 追加写入
for chunk in pd.read_csv(fh, chunksize=50_000):
    tbl = pa.Table.from_pandas(...)
    writer.write_table(tbl)
```

这些数据集的 `records.csv` 被省略（文件过大），仅输出 `records.parquet`。

### VHH 信号肽与 His-tag 截除（cognano、tsuruta × 2）

原始序列带 22 AA 分泌信号肽 + C 端 His-tag，convert.py 自动截除：

```python
_SP  = "MKYLLPTAAAGLLLLAAQPAMA"   # 22 AA 信号肽
_HIS = re.compile(r"H{4,}$")      # C 端 ≥4× His-tag

def _strip_vhh(seq):
    if seq.startswith(_SP): seq = seq[len(_SP):]
    seq = _HIS.sub("", seq)
    return seq or None
```

### 截尾测量值（AbRank IC50/Kd）

AbRank 中部分数值以 `<5.00e-01` 形式标注（只知上界，不知精确值），无法参与排名，标记为 `drop_reason = "censored_measurement"`。

### li2023machine affinity2 前 6 行跳过

affinity2 CSV 前 6 行为版权声明，`skiprows=list(range(0,6))` 跳过，行号从第 8 行（1-indexed）开始计数。

### AbRank 多指标拆分

每个源行最多产生 2 条记录（Kd 一条、IC50 一条），`group_id` 中编码指标名以隔离不同度量：

```
AbRank/dataset/{ag_key}/neg_log10_kd_M/experimental
AbRank/dataset/{ag_key}/neg_log10_ic50_ugml/experimental
```

### makowski iso_ant 文件名拼写错误

源文件名含笔误（`makowksi` 缺少字母 `a`），manifest.csv 和 convert.py 均已与实际文件名保持一致，无需修正。

---

## 工具脚本

| 脚本 | 用途 |
|------|------|
| `scripts/prepare/binding/gen_manifest.py` | 从所有 convert.py 提取元数据，重新生成 manifest.csv 和 antigen_missing_summary.csv（prepare_all.sh 自动调用） |
| `scripts/prepare/binding/patch_allow_X.sh` | 一次性：将所有 84 个 convert.py 的 `_VALID_AA` 更新为包含 X |
| `scripts/prepare/validate_processed_table.py` | 校验单个数据集输出是否符合 27 列 schema |

---

## 待办

- **首次运行前**：执行 `bash scripts/prepare/binding/patch_allow_X.sh` 启用 X 残基支持
- **抗原序列补充**：大多数数据集 `antigen_sequence = null`，需人工从 UniProt/PDB 检索并填入各 convert.py 的 `ANTIGEN_SEQ` 常量（参见 `processed/binding/antigen_missing_summary.csv`）
- **blocked 数据集**：`rawat2022abcov` 等 2 个需先从论文找到抗原信息才能实现
