# AffinityTransformer 编程规范

版本：v0.5  
依据：新版 `README.md`  
角色设定：本文件按项目组长给开发组员发任务的方式编写。它不是论文说明，也不是泛泛的 Python 风格指南；它规定哪些代码该写、函数输入输出是什么、怎么验收、哪些行为禁止。

v0.3 变更：在 §5.2 中新增 listwise 视图（`AffinityGroupExample` / `build_groups` /
`ListwiseAffinityDataset`），与既有的 pairwise 视图（`build_pairs` /
`PairwiseAffinityDataset`）并列，二者共享同一份 `AffinityRecordDataset` /
`AffinityExample` 基座。目的是为后续"对照试验：哪种上游任务（pairwise /
listwise / pointwise）效果最好"留出数据结构上的空间，本次改动不影响 §5.1
和既有 pairwise 代码/测试。

v0.4 变更：明确 §5.3/§5.4 的"依赖注入"边界——本项目不内置共享词表/tokenizer，
也不在框架代码里构造任何随机初始化的 transformer。`collate_rank_batch` /
`collate_pair_batch` 改为接收外部传入的 `antibody_tokenizer` /
`antigen_tokenizer`（满足 HuggingFace 风格的 `__call__(sequences, padding=True,
return_tensors="pt") -> {"input_ids", "attention_mask"}` 接口即可，具体用
ESM2、AbLang-2 还是别的由调用方决定），并新增"抗体序列拼接规则"（单链直接用、
重链+轻链按 AbLang-2 的 `"{VH}|{VL}"` 格式拼接、仅有一条链时单独使用）。
`AffinityRanker.__init__` 改为接收已经构建好的 `antibody_encoder` /
`antigen_encoder`（任意满足 `forward(input_ids, attention_mask) ->
FloatTensor[B, L, d_model]` 且不产生 NaN 的 `nn.Module`），forward 逻辑本身
不变。字符串到具体模型/tokenizer 对象的映射留给 §5.7 trainer.py（尚未实现）。
本次改动不影响 §5.1 `ModelConfig` 字段定义、§5.2 及更早内容。

v0.5 变更：实现 §5.6-§5.9（`metrics.py`/`trainer.py`/`user_entry.py`/
`utils.py`）及 §7.3 最小集成测试，并补充三处此前未落定的设计细节：

1. §5.6 `summarize_group_spearman`：在 `compute_group_spearman` 之上，按
   `label_kind` 分组（外加一个合并所有 group 的 `"overall"`）分别计算
   macro/weighted Spearman 和 valid/skipped group 计数，详见该节新增内容。
2. §5.7 `build_model_and_tokenizers`：把 `ModelConfig.antibody_encoder` /
   `antigen_encoder` 这两个字符串映射为真实的 ESM2 encoder + tokenizer。当前
   只支持固定的 ESM2 short-name 列表（`esm2_t6_8M` ... `esm2_t48_15B`），且
   "短名是否受支持"和"`d_model` 是否与该短名匹配"两项校验在 import
   `transformers`、联网下载权重之前完成，避免不支持的配置在跑很久之后才报错。
   其它 encoder 家族（如 AbLang-2）尚未实现。
3. §5.8 "model + tokenizer 打包约定"（新增，见该节）：`load_model` 返回的
   `AffinityRanker` 实例上会附加两个普通属性 `model.antibody_tokenizer` /
   `model.antigen_tokenizer`，供 `score_antibodies`/`rank_antibodies`
   读取，从而在不改变这两个函数对外签名（只接收 `model: AffinityRanker`）
   的前提下拿到匹配的 tokenizer。

本次改动不影响 §5.1-§5.5 已实现内容，也不影响 §6/§9 的通用规则。

## 0. 总体判断

新版 `README.md` 的科学性基本站得住。项目把异构亲和力数据建模为同质 group 内排序学习，而不是全局绝对值回归；这与当前数据特征一致，也与 Spearman 评价指标一致。

当前没有必须推翻的科学问题，但实现时必须守住四条边界：

1. 模型输出的 `score` 只在同一 `group_id` 内比较，不解释为跨数据集的绝对亲和力。
2. `Kd`、`IC50`、`EC50`、`fitness`、`Pred_affinity`、`bind/no bind` 必须先明确方向和标签质量，再进入训练。
3. 缺失抗原信息可以标记，不能伪造成真实特征；缺失抗原对应的位置属于无效 token，必须被 mask 掉，不得当作 valid token 送入 attention。
4. Cross-Attention 是特征交互模块，不是三维结构模拟；任何汇报和注释都不要过度宣称生物物理含义。

## 1. 项目代码边界

本项目分两类代码：数据处理脚本和通用训练模块。不要把二者混在一起。

### 1.1 数据处理代码

数据处理阶段不追求通用 Python module。每个数据集、每个子目录可以写独立 shell 脚本，脚本可以硬编码该数据集的列名、文件名和转换逻辑。

理由很简单：原始表格来源太异构，强行写一个通用 parser 会浪费时间，还会把真实差异藏进复杂分支里。数据处理这一步的目标只有一个：产出标准化的抗原-抗体训练表。只要最终表正确、可复现、可审计，脚本本身不需要优雅地处理所有未知输入。

数据脚本允许：

1. 按数据集写死列名。
2. 按数据集写死 label 转换规则。
3. 使用 shell、awk、csvkit、短 Python 脚本或 pandas 一次性转换脚本。
4. 每个脚本只服务一个数据源。

数据脚本不允许：

1. 静默丢弃记录。任何被过滤、排除或无法进入训练表的记录，都必须写入 QC 表或日志，并标明原因。
2. 原地修改 `data/` 下的原始文件。
3. 输出不符合标准 schema 的训练表。
4. 把不同 assay、不同 metric、不同 label 质量的数据无说明地合并。
5. 依赖人工交互输入。

### 1.2 通用训练模块

通用模块只负责标准表之后的流程：

1. 读取标准化记录。
2. 构造 group 内 pair。
3. 构造 dataset/dataloader。
4. 编码抗体和抗原。
5. 训练 RankNet 或后续排序模型。
6. 评估 group-level Spearman。
7. 给外部用户提供调用入口。

通用模块不得再理解原始 CSV 的奇怪列名。看到原始数据列名出现在 `dataset.py`、`trainer.py` 或 `user_entry.py` 里，直接打回。

## 2. 推荐目录结构

```text
AffinityTransformer/
  README.md
  docs/
    programming_spec.md
  data/
    binding/
    expression/
    ...
  scripts/
    prepare/
      binding/
        prepare_all.sh
        manifest.csv
        AbRank/
          dataset/
            prepare.sh
            convert.py
        li2023machine/
          affinity1/
            prepare.sh
            convert.py
          affinity2/
            prepare.sh
            convert.py
        kothiwal2025htp/
          DCC_ec50/
            prepare.sh
            convert.py
          DCC_spr/
            prepare.sh
            convert.py
          ...
      validate_processed_table.py
  processed/
    binding/
      AbRank/
        dataset/
          records.parquet
          qc_summary.csv
          dropped_records.csv
      kothiwal2025htp/
        DCC_ec50/
          records.parquet
          qc_summary.csv
          dropped_records.csv
       	...
    all_records.parquet
    total_records.csv
  configs/
    baseline_ranknet.yaml
  affinity_transformer/
    __init__.py
    config.py
    dataset.py
    dataloader.py
    model.py
    trainer.py
    user_entry.py
    losses.py
    metrics.py
    utils.py
  tests/
    test_dataset.py
    test_dataloader.py
    test_model.py
    test_losses.py
    test_metrics.py
    test_user_entry.py
  outputs/
```

目录结构说明：

1. `scripts/prepare/` 是数据处理区，可以数据集专用。
2. `processed/` 是数据处理的最终结果。
3. `affinity_transformer/` 是通用模型代码，不能依赖某个原始数据集的特殊列。
4. `__init__.py` 可以写的很简单，甚至可以只声明版本号；不要在 import 时加载模型或读大文件。
5. 无论早期代码量有多大，`losses.py`、`metrics.py` 都不要合并在 `trainer.py`，只要函数超过两个就拆出来。
6. 上面的 `AbRank`、`li2023machine`、`kothiwal2025htp` 只是目录形态示例，不代表只处理这些 binding 数据集。
7. 当前 `data/flab_metadata.csv` 中 `category = "binding"` 的源文件数为 86；`scripts/prepare/binding/manifest.csv` 必须覆盖所有计划纳入训练或明确暂不纳入的 binding 源文件。

## 3. 标准训练表 schema

每个数据处理脚本最终必须输出 `records.parquet` 或 `records.csv`。通用模块只接受这个 schema。

必需字段：

```text
record_id: str
dataset_id: str
study_id: str
table_id: str
source_file: str
source_row: int

antibody_id: str | None
antibody_type: "Fv" | "scFv" | "VHH" | "Fab" | "IgG" | "unknown"

heavy_chain: str | None
light_chain: str | None
single_chain_sequence: str | None

antigen_key: str | None
antigen_name: str | None
antigen_sequence: str | None
antigen_source: "provided" | "retrieved" | "missing"

assay_name: str | None
assay_type: "binding" | "neutralization" | "fitness" | "expression" | "unknown"

metric_name: str
metric_value_raw: str | float | int | None
metric_value_numeric: float | None
metric_unit: str | None
metric_direction: "higher_is_better" | "lower_is_better" | "unknown"
transform_rule: str | None

rank_label: float | None
label_kind: "experimental" | "predicted" | "binary" | "unknown"

group_id: str
keep_for_training: bool
drop_reason: str | None
```

字段规则：

1. `rank_label` 必须是越大越好。
2. binding 数据的 `dataset_id` 建议固定为 `{study_id}/{table_id}`，用于汇总后保留源表层级。
3. `group_id` 至少由 `dataset_id`、`antigen_key`、`assay_name`、`metric_name`、`label_kind` 生成。
4. `label_kind = "binary"` 的记录只能产生正负 pair，同标签之间不能产生 pair。
5. `antigen_sequence = None` 时，`antigen_source` 必须是 `"missing"`。
6. `keep_for_training = False` 的记录可以留在表里，但 dataset 必须过滤掉它。



## 4. 数据处理脚本规范

每个 binding 源表必须对应一个明确的处理单元。处理单元可以按 `study_id/table_id/` 两层目录组织，每个处理单元必须包含一个 `prepare.sh`，作为该源表的唯一入口。

示例结构：

```text
scripts/prepare/binding/AbRank/dataset/
  prepare.sh
  convert.py

scripts/prepare/binding/kothiwal2025htp/DCC_ec50/
  prepare.sh
  convert.py
```

命名规则：

1. `study_id` 来自源文件名前缀或论文/数据集简称，例如 `li2023machine`、`kothiwal2025htp`、`phillips2021binding`。
2. `table_id` 表示该 study 下的具体表，例如 `affinity1`、`DCC_ec50`、`cr6261_h1_kd`。
3. 如果一个 study 只有一个表，也仍然保留 `table_id`，可以使用 `dataset`、`default` 或更具体的表名。
4. `processed/binding/{study_id}/{table_id}/` 必须和 `scripts/prepare/binding/{study_id}/{table_id}/` 一一对应。

`scripts/prepare/binding/manifest.csv` 必须记录所有 binding 源表：

```text
source_file
study_id
table_id
prepare_dir
processed_dir
status
reason
```

`status` 只允许：

```text
planned
ready
excluded
blocked
```

规则：

1. 当前 metadata 中的 86 个 binding 源文件都必须在 manifest 中出现。
2. 暂不处理的源文件也必须写入 manifest，并给出 `excluded` 或 `blocked` 的原因。
3. `prepare_all.sh` 只读取 manifest 中 `status = "ready"` 的行并逐个运行对应 `prepare.sh`。

`prepare.sh` 规范：

```bash
#!/usr/bin/env bash
set -euo pipefail

# 1. Define input and output paths.
# 2. Run the dataset-specific converter.
# 3. Run the shared schema validator.
# 4. Print row counts and output paths.
```

要求：

1. `prepare.sh` 从仓库根目录运行时必须成功。
2. binding 输出固定写入 `processed/binding/{study_id}/{table_id}/`。
3. 转换脚本必须生成 `records.parquet` 或 `records.csv`。
4. 同时生成 `qc_summary.csv` 和 `dropped_records.csv`。
5. 每次运行覆盖自己的输出目录可以接受，但不得删除其他数据集输出。
6. 如果脚本需要外部下载抗原序列，必须写入 `antigen_source` 和来源说明；早期 MVP 可以直接标记 missing。
7. `prepare_all.sh` 可以把所有 `ready` 处理单元的 `records.parquet` 汇总为 `processed/binding/all_records.parquet`，但汇总表必须保留 `study_id`、`table_id`、`dataset_id` 和 `source_file`。

共享验证脚本 `scripts/prepare/validate_processed_table.py` 是允许存在的，因为它不是通用 raw parser，只检查最终表是否符合 schema。

最低验收：

1. 必需列全部存在。
2. `record_id` 不为空且唯一。
3. `group_id` 不为空。
4. `rank_label` 对 `keep_for_training = True` 的记录必须是有限数值。
5. 合法训练记录必须有 `heavy_chain`。
6. `heavy_chain` 和 `light_chain` 只包含合法氨基酸字符。
7. `label_kind`  `metric_direction` `assay_type` `antigen_source ` `antibody_type`均提供了取值集合，故只能取允许值。

## 5. 通用模块职责

### 5.1 `config.py`

负责配置读取和默认值管理。

建议对象：

```python
@dataclass
class DataConfig:
    train_path: Path
    valid_path: Path | None
    max_pairs_per_group: int
    seed: int

@dataclass
class ModelConfig:
    antibody_encoder: str
    antigen_encoder: str | None
    d_model: int
    use_cross_attention: bool

@dataclass
class TrainConfig:
    batch_size: int
    lr: float
    epochs: int
    device: str

@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
```

函数：

```python
def load_config(path: Path) -> Config:
    ...
```

该函数负责把一次实验所需的所有可变参数统一设置好，包括数据路径、模型结构开关、训练超参数、随机种子等。
消融实验和对照实验只是通过切换不同配置文件来实现。

验收：

1. 缺少必要字段时抛出 `ValueError`。
2. 路径不存在时抛出 `FileNotFoundError`。
3. 不在函数里偷偷改默认随机种子。

### 5.2 `dataset.py`

负责读取标准训练表、过滤可训练记录、并构造排序训练样本。

本模块只处理已经通过 schema 校验的 processed table，不负责原始数据清洗、指标方向转换或 group_id 生成，具体流程如下：

```
processed table
→ schema validation
→ filter trainable records
→ build AffinityExample
→ build pairs if task = pairwise ranking      → PairwiseDataset
→ build groups if task = listwise ranking     → ListwiseDataset
→ collate_fn padding/mask
→ DataLoader
→ Trainer
```

`AffinityRecordDataset` / `AffinityExample` 是所有上游任务共享的基座：

1. **pointwise**：直接消费 `AffinityRecordDataset`，每条样本自带一个 `rank_label`，不需要额外的类。
2. **pairwise**（当前 baseline，RankNet）：`build_pairs` + `PairwiseAffinityDataset` + `AffinityPairExample`。
3. **listwise**（ListMLE / LambdaRank 式动态加权 / differentiable-Spearman 等后续上游任务）：`build_groups` + `ListwiseAffinityDataset` + `AffinityGroupExample`。

三者都从同一份 `filter_trainable_records` 输出出发，互不依赖、互不修改；某次训练用哪个视图，是 `dataloader.py`/`trainer.py` 之后要加的 config 级开关（本规范暂不展开，§5.2 当前只交付数据结构本身）。

现在可以先将这些功能放进一个文件，等到需要写的函数实在太多时，再把相关函数分配给不同文件放置，到时候可以这样组织文件

```
schema.py   标准训练表字段、枚举、校验
records.py  读取 processed table，过滤 keep_for_training
pairs.py    build_pairs
groups.py   build_groups
dataset.py  AffinityDataset / PairwiseAffinityDataset / ListwiseAffinityDataset
collate.py  RankBatch / PairBatch / GroupBatch / collate_fn
```

核心类：

```python
class AffinityRecordDataset(torch.utils.data.Dataset):
    def __init__(self, records: pd.DataFrame) -> None:
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> AffinityExample:
        ...


class PairwiseAffinityDataset(torch.utils.data.Dataset):
    def __init__(self, records: pd.DataFrame, pairs: pd.DataFrame) -> None:
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> AffinityPairExample:
        ...


class ListwiseAffinityDataset(torch.utils.data.Dataset):
    def __init__(self, records: pd.DataFrame, groups: pd.DataFrame) -> None:
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> AffinityGroupExample:
        ...
```

核心函数：

```python
def load_records(path: Path) -> pd.DataFrame:
    ...

def filter_trainable_records(records: pd.DataFrame) -> pd.DataFrame:
    ...

def build_pairs(
    records: pd.DataFrame,
    max_pairs_per_group: int,
    seed: int,
) -> pd.DataFrame:
    ...

def build_groups(
    records: pd.DataFrame,
    max_group_size: int | None,
    seed: int,
) -> pd.DataFrame:
    ...
```

`AffinityRecordDataset` 的一个样本`AffinityExample` 至少包含如下信息：

```text
record_id: str
dataset_id: str

heavy_chain: str | None
light_chain: str | None
single_chain_sequence: str | None
antibody_type: str

antigen_sequence: str | None
antigen_key: str | None

rank_label: float
label_kind: str
group_id: str
```

`PairwiseAffinityDataset`的一个样本`AffinityPairExample` 至少包含如下信息：

```text
pair_id: str
group_id: str
left: AffinityExample
right: AffinityExample
y_ij: float
```

`ListwiseAffinityDataset` 的一个样本 `AffinityGroupExample` 至少包含如下信息：

```text
group_id: str
label_kind: str
examples: tuple[AffinityExample, ...]   # 同一 group 内全部存活记录，按 record_id 排序
```

**Pair 构造规则**

1. 只使用 `keep_for_training = True` 的记录。
2. 只在同一 `group_id` 内构造 pair。
3. `rank_label` 为 `None`、`NaN`、`inf` 或 `-inf` 的记录不得进入 pair。
4. `label_i == label_j` 时不构造 pair。
5. `label_i > label_j` 时 `y_ij = 1`，否则 `y_ij = 0`。
6. 每个 group 的 pair 数不得超过 `max_pairs_per_group`。
7. `label_kind = "binary"` 的记录只允许正负样本之间配对。
8. 同一对 record 默认只出现一次，不生成反向重复 pair。

验收：

1. 固定 seed 时 pair 结果可复现。
2. 输出 pair 不跨 group。
3. 空 group 或单一标签 group 不报错，但不产生 pair。

**Group 构造规则**（`build_groups`，listwise 视图，与上面 Pair 构造规则共用第 1/3 条）

1. 只使用 `keep_for_training = True` 的记录（同 Pair 规则 1）。
2. 只在同一 `group_id` 内构成 group，输出按 `(group_id, record_id)` 排序。
3. `rank_label` 为 `None`、`NaN`、`inf` 或 `-inf` 的记录不得进入 group（同 Pair 规则 3）。
4. 一个 group 内 `rank_label` 唯一值少于 2 个时，整个 group 不产生输出（无序可学，呼应 §5.6 的 `n_unique_labels < 2` 跳过规则）。
5. 每个 group 的成员数不得超过 `max_group_size`（`None` 表示不裁剪）；裁剪时按 `f"{seed}:{group_id}"` 派生的随机数确定性采样。
6. `max_group_size` 非 `None` 时必须 `>= 2`，否则抛出 `ValueError`。
7. `label_kind = "binary"` 的 group 不做特殊处理：只要正负两类都存在（`n_unique_labels == 2`），整个 group（含同类内的多条记录）原样保留——listwise loss 下"全部正例排在全部负例之前"本身就是合法的目标排列。

验收：

1. 固定 seed 时 group 结果可复现。
2. 输出 group 不跨 `group_id`。
3. 空 group 或单一标签 group 不报错，但不产生输出。

### 5.3 `dataloader.py`

如果默认 PyTorch dataloader 足够，可以不写复杂封装；但只要涉及 padding、mask、pair batch，就必须集中放在这里。

**Tokenizer 接口（依赖注入）**：本模块不内置任何共享词表，也不假设具体的预训练模型。`collate_rank_batch` / `collate_pair_batch` 通过参数接收外部传入的 tokenizer，签名约定为：

```python
class Tokenizer(Protocol):
    def __call__(
        self, sequences: list[str], padding: bool = True, return_tensors: str = "pt"
    ) -> Mapping[str, torch.Tensor]:
        """返回至少包含:
        - input_ids: LongTensor[B, L]
        - attention_mask: LongTensor[B, L]，1 = 真实 token，0 = padding
        """
```

HuggingFace 的 `PreTrainedTokenizerBase`（ESM2、AbLang-2 等）天然满足该签名；其他 tokenizer 如接口不同，由调用方包一层适配后再传入。`attention_mask` 中 1=真实 token / 0=padding，与下面 mask 约定（True=valid / False=padding）方向一致，转换时只需 `.bool()`，不需要取反。tokenizer 对象本身、以及"`ModelConfig.antibody_encoder`/`antigen_encoder` 字符串 -> 具体 tokenizer/模型"的映射，都是 §5.7 trainer.py（尚未实现）的职责，不在本模块内构造。

核心对象：

```python
@dataclass
class RankBatch:
    antibody_tokens: torch.Tensor
    antibody_mask: torch.Tensor
    antigen_tokens: torch.Tensor | None
    antigen_mask: torch.Tensor | None
    labels: torch.Tensor
    record_ids: list[str]
    group_ids: list[str]

@dataclass
class PairBatch:
    left: RankBatch
    right: RankBatch
    y_ij: torch.Tensor
```

函数：

```python
def collate_rank_batch(
    examples: Sequence[AffinityExample],
    antibody_tokenizer: Tokenizer,
    antigen_tokenizer: Tokenizer | None = None,
) -> RankBatch:
    ...

def collate_pair_batch(
    examples: Sequence[AffinityPairExample],
    antibody_tokenizer: Tokenizer,
    antigen_tokenizer: Tokenizer | None = None,
) -> PairBatch:
    ...
```

**抗体序列拼接规则**（`AffinityExample` -> 喂给 `antibody_tokenizer` 的字符串）：

1. `single_chain_sequence` 非 `None` 时直接使用。
2. 否则若 `heavy_chain`、`light_chain` 均非 `None`，拼接为 `f"{heavy_chain}|{light_chain}"`（重链在前，`|` 分隔——AbLang-2 paired-chain 输入格式约定）。
3. 否则使用非 `None` 的那一条链（典型情况是 VHH：只有 `heavy_chain`）。
4. 三者都是 `None` 时抛 `ValueError`（这类记录不应出现在 `filter_trainable_records` 的输出中）。

**抗原 tokenize 规则**：

1. `antigen_tokenizer is None`，或 batch 内所有 example 的 `antigen_sequence` 均为 `None`：整批 `antigen_tokens = antigen_mask = None`。
2. 否则逐条 tokenize；某条 example 的 `antigen_sequence is None` 时用占位字符串 tokenize后，强制将该行 `antigen_mask` 整行置为 `False`（即便占位字符串本身被 tokenizer 编码出"看似有效"的 token）。

mask 约定：

1. `True = valid token`。
2. `False = padding or missing`。
3. 缺失抗原不能作为有效 token 进入 attention。
4. batch 内 Fv、scFv、VHH 混合时必须能 collate。
5. `mask.shape` 必须与 token 序列维度一致，即 `[B, L]`，而不是与 embedding 维度完全一致。

验收：

1. batch size 为 1 时可以运行。
2. 全部抗原缺失时 `antigen_tokens is None`，不产生 NaN。
3. 部分抗原缺失的混合 batch：缺失行 `antigen_mask` 全 `False`，其余行不受影响。
4. token ids 场景：`mask.shape == token_ids.shape`。
5. `collate_pair_batch` 正确拆分出 `left`/`right` 两个 `RankBatch`，并产出 `y_ij`。

### 5.4 `model.py`

负责模型本体。早期不要拆太复杂。

**Encoder 接口（依赖注入）**：`antibody_encoder` / `antigen_encoder` 是外部已经构建好、传入的 `nn.Module`，本模块不在内部构造任何预训练模型或随机初始化网络。接口约定：

```text
forward(input_ids: LongTensor[B, L], attention_mask: BoolTensor[B, L]) -> FloatTensor[B, L, d_model]
```

即：输入 token ids 与 mask（mask 约定同 §5.3，True=valid），输出每个 token 的隐藏表示，最后一维等于 `d_model`，且对任意输入（包括某一行 `attention_mask` 全 `False`）都不产生 `NaN`。真实场景下这是一个"预训练模型 + 投影到 `d_model` 的适配器"，适配器的构造属于 §5.7 trainer.py（尚未实现），不在本模块范围内。

最低接口：

```python
class AffinityRanker(nn.Module):
    def __init__(
        self,
        antibody_encoder: nn.Module,
        antigen_encoder: nn.Module | None,
        d_model: int,
        use_cross_attention: bool,
    ) -> None:
        ...

    def forward(self, batch: RankBatch) -> torch.Tensor:
        ...
```

输出：

```text
score: FloatTensor[B]
```

规则：

1. `score` 不经过 sigmoid、softmax 或 clamp。

2. 当 `antigen_encoder is None`，或 `batch.antigen_tokens is None`，或某一行 `antigen_mask` 全 `False` 时，模型不得执行普通 antigen attention。

   采用 antibody-only baseline：缺失抗原时，跳过 cross-attention，只使用 antibody representation（抗原侧用零向量）打分。

   > 注：learned missing-antigen token（可学习占位符）作为后续消融实验候选，不纳入当前 baseline。

3. attention 中的 invalid token 必须通过 additive/key_padding mask 屏蔽；`forward` 内必须保证 encoder 输出中任何 `NaN`（例如某行 mask 全 `False` 导致的退化情况）不会传播到 `score`。

4. 模型代码里不要写原始数据集名称。同时，模型 forward 只允许依赖 batch 中的张量字段，不应依赖 record_ids、group_ids 等元信息进行预测。

验收：

1. Fv 输入可以 forward。
2. VHH 输入可以 forward。
3. Fv 和 VHH 混合 batch 可以 forward。
4. 缺失抗原输入（`antigen_tokens is None`，或某行 `antigen_mask` 全 `False`）可以 forward，且输出无 NaN。
5. 支持 backward：loss 对模型参数的梯度可计算，且不含 NaN。

### 5.5 `losses.py`

核心函数：

```python
def ranknet_loss(
    score_i: torch.Tensor,
    score_j: torch.Tensor,
    y_ij: torch.Tensor,
    sigma: float = 1.0,
) -> torch.Tensor:
    ...
```

规则：

1. 使用 logits 形式实现。
2. `logit = sigma * (score_i - score_j)`。
3. 优先使用 `torch.nn.functional.binary_cross_entropy_with_logits`。
4. `y_ij` 只能是 0 或 1。

验收：

```text
ranknet_loss(score_i=2, score_j=1, y_ij=1)
<
ranknet_loss(score_i=1, score_j=2, y_ij=1)
```

### 5.6 `metrics.py`

核心函数：

```python
def compute_group_spearman(predictions: pd.DataFrame) -> pd.DataFrame:
    ...
```

输入字段：

```text
record_id
group_id
rank_label
score
label_kind
dataset_id
```

输出字段：

```text
group_id
dataset_id
label_kind
n_records
n_unique_labels
spearman
```

规则：

1. `n_unique_labels < 2` 的 group 不计算 Spearman（该 group 仍在输出里占一行，
   `spearman = NaN`，方便上层统计 skipped group 数量）。`score` 在 group 内
   完全相同（无法定义相关系数）时同样得到 `spearman = NaN`。
2. 同时报告 macro average （这个需要考虑大batch与小batch的问题）和按 group size 加权 average。
3. 不允许只报 overall metric。
4. binary label 的 Spearman 必须单独统计。

实现：规则 2-4 由第二个函数完成：

```python
def summarize_group_spearman(
    group_metrics: pd.DataFrame,
) -> dict[str, dict[str, float | int]]:
    ...
```

输入是 `compute_group_spearman` 的输出。输出是一个 dict：

- 一个 `"overall"` key：合并所有 group。
- 每个出现过的 `label_kind` 各一个 key（例如 `"experimental"`、
  `"binary"`），对应规则 4。

每个 entry 是：

```text
n_groups
n_valid_groups       # spearman 非 NaN 的 group 数
n_skipped_groups      # n_groups - n_valid_groups
macro_spearman         # valid group 的简单平均，n_valid_groups == 0 时为 NaN
weighted_spearman      # valid group 按 n_records 加权平均，n_valid_groups == 0 时为 NaN
```

`group_metrics` 为空时只有 `"overall"`，计数全为 0、两个 average 均为 NaN。

`Trainer.evaluate`（§5.7）把这个 dict 摊平成 `dict[str, float]`：
`"overall"` 的四个字段分别变成 `valid_macro_spearman` /
`valid_weighted_spearman` / `n_valid_groups` / `n_skipped_groups`（与 §8
`metrics.json` 的最低字段对应）；其余每个 `label_kind` 变成
`valid_{label_kind}_macro_spearman` / `valid_{label_kind}_weighted_spearman`
/ `valid_{label_kind}_n_valid_groups` / `valid_{label_kind}_n_skipped_groups`。

依赖说明：本项目 sandbox 未安装 `scipy`；Spearman 用
`x.rank().corr(y.rank())`（Pearson correlation of ranks）计算，数值上与
`scipy.stats.spearmanr` 一致。

### 5.7 `trainer.py`

负责训练循环、验证、checkpoint、日志。

最低接口：

```python
class Trainer:
    def __init__(
        self,
        model: AffinityRanker,
        config: Config,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader | None = None,
        valid_record_metadata: pd.DataFrame | None = None,
        output_dir: Path | None = None,
        early_stopping_patience: int | None = None,
        early_stopping_metric: str = "valid_weighted_spearman",
    ) -> None:
        ...

    def fit(self) -> None:
        ...

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        ...

    def save_checkpoint(self, path: Path) -> None:
        ...

    def load_checkpoint(
        self, path: Path, map_location: str | torch.device | None = None
    ) -> dict[str, object]:
        ...
```

`Trainer` 是依赖注入的：`model` 和两个 `DataLoader` 在传入前已经构建完成
（包括它们 `collate_fn` 用的 tokenizer），`Trainer` 本身不关心这些 tokenizer
来自真实 ESM2 还是测试用的 fake encoder。

构造参数补充说明：

- `train_dataloader`：产出 `PairBatch`（§5.3），例如
  `PairwiseAffinityDataset` + `collate_pair_batch`。
- `valid_dataloader`：产出 `RankBatch`（§5.3），例如
  `AffinityRecordDataset` + `collate_rank_batch`；为 `None` 时 `fit`
  跳过验证。
- `valid_record_metadata`：包含 `record_id, dataset_id, label_kind` 三列，
  覆盖 `valid_dataloader` 里出现的所有 `record_id`——这是
  `compute_group_spearman`（§5.6）需要、但不在 `RankBatch` 里的字段。通常取
  `filter_trainable_records` 输出里对应的三列。`evaluate` 依赖这个参数；不提供
  时调用 `evaluate` 抛 `ValueError`。
- `output_dir`：NaN loss 时写 `error_context.json` 的目录（规则 5）；为
  `None` 时仍会立刻停止，但不写文件。
- `early_stopping_patience` / `early_stopping_metric`（规则 6）：
  `early_stopping_metric` 取 `evaluate` 输出里的某个 key（默认
  `"valid_weighted_spearman"`），连续 `early_stopping_patience` 个 epoch 未
  严格提升（`NaN` 视为未提升）则提前停止。`early_stopping_patience=None`
  （默认）关闭早停。

`evaluate` 的输出是 §5.6 `summarize_group_spearman` 的摊平结果，详见 §5.6。

`save_checkpoint` 写入的 dict 字段：`model_state_dict`、`config`（完整
`Config`）、`global_step`、`seed`（即 `config.data.seed`）。`load_checkpoint`
读取同一个 dict，把 `model_state_dict` 加载进 `self.model`、恢复
`self.global_step`，并原样返回整个 dict；`self.config` 不会被覆盖——调用方
需保证自己构造的 `Trainer.model` 架构与 `checkpoint["config"].model` 匹配。

规则：

1. trainer 不读取原始 CSV。
2. trainer 不构造 group 规则，只消费 dataset 输出的 pair。
3. 每个 epoch 输出 train loss 和 validation group-level metrics。
4. checkpoint 保存的训练快照必须包含模型权重、配置、训练步数、随机种子。
5. 出现 NaN loss 立刻停止并保存错误上下文。
6. 可以引入早停法，防止出现过拟合

最低验收：

1. 能在 10 条以内 toy data 上跑通一个 epoch。
2. 能保存并重新加载 checkpoint。
3. 固定 seed 时，toy run 的 pair 顺序一致。

#### 5.7.1 `build_model_and_tokenizers`：encoder/tokenizer 字符串映射

```python
def build_model_and_tokenizers(
    model_config: ModelConfig,
) -> tuple[AffinityRanker, Tokenizer, Tokenizer | None]:
    ...
```

这是 v0.4 留给 §5.7 的"字符串 -> 具体模型/tokenizer 对象"映射。它是
`affinity_transformer` 里唯一会 `import transformers`、联网下载预训练权重的
地方，且这个 import 在函数体内（lazy import），不在模块顶层（§9：禁止在
import package 时加载大模型/读取大数据）。

输入：`model_config`（即 `Config.model`）。`antibody_encoder`，以及非 `None`
时的 `antigen_encoder`，必须是下表中的 ESM2 short name，且 `d_model` 必须与
对应隐藏维度一致：

```text
esm2_t6_8M    -> 320
esm2_t12_35M  -> 480
esm2_t30_150M -> 640
esm2_t33_650M -> 1280
esm2_t36_3B   -> 2560
esm2_t48_15B  -> 5120
```

输出：`(model, antibody_tokenizer, antigen_tokenizer)`。`model` 是用新构建的
ESM2 encoder 包装出的 `AffinityRanker`（`d_model`/`use_cross_attention` 取自
`model_config`）；`antibody_tokenizer`/`antigen_tokenizer` 是对应的
HuggingFace tokenizer，满足 §5.3 `Tokenizer` 协议；`antigen_encoder is None`
时 `antigen_tokenizer` 为 `None`。

规则：

1. "short name 是否受支持"和"`d_model` 是否匹配该 short name"两项校验，必须
   在 import `transformers`、联网下载权重**之前**完成——不支持的 encoder
   （目前只支持 ESM2，AbLang-2 等尚未实现）要立刻报 `ValueError`，不要等到
   网络请求之后才失败。
2. 其它 encoder 家族（AbLang-2 等）目前不支持，遇到时报错，不允许静默退化为
   某个默认 encoder。

### 5.8 `user_entry.py`

这是外部用户调用模型的入口。它只暴露稳定、少量、好理解的 API。

核心函数：

```python
def load_model(checkpoint_path: Path, config_path: Path | None = None) -> AffinityRanker:
    ...

def score_antibodies(
    antigen_sequence: str | None,
    antibodies: Sequence[AntibodyInput],
    model: AffinityRanker,
) -> pd.DataFrame:
    ...

def rank_antibodies(
    antigen_sequence: str | None,
    antibodies: Sequence[AntibodyInput],
    model: AffinityRanker,
) -> pd.DataFrame:
    ...
```

`AntibodyInput`：

```text
antibody_id: str
heavy_chain: str
light_chain: str | None
single_chain_sequence: str | None
antibody_type: "Fv" | "scFv" | "VHH" 
```

当前还没有支持"Fab" | "IgG" | "unknown"，后续可以补充

输出：

```text
antibody_id
score
rank
```

规则：

1. 外部用户不需要知道 pair、group、loss。
2. 单个抗原和一组抗体输入时，输出按 `score` 降序排列。
3. 输入序列非法时抛出 `ValueError`，不要静默修正。
4. 没有抗原序列时允许调用，但必须走模型定义的 missing-antigen 分支。

#### 5.8.1 model + tokenizer 打包约定

`score_antibodies`/`rank_antibodies` 要把 `AntibodyInput`/抗原序列 tokenize
成 `RankBatch`（§5.3），需要 `antibody_tokenizer`（以及可能的
`antigen_tokenizer`）；但它们对外的签名只接收 `model: AffinityRanker`，不接收
tokenizer。

解决方式：`load_model` 在返回前，把 `build_model_and_tokenizers`（§5.7.1）给
出的两个 tokenizer 作为**普通属性**（非 `nn.Module`、不参与
`state_dict`/`parameters()`）挂在它返回的 `AffinityRanker` 实例上：

```text
model.antibody_tokenizer: Tokenizer
model.antigen_tokenizer: Tokenizer | None
```

`score_antibodies`/`rank_antibodies` 通过 `getattr(model,
"antibody_tokenizer", None)` 读取；如果 `model` 上没有这个属性（例如直接用
`AffinityRanker(...)` 构造、未经过 `load_model`），抛出 `ValueError`，提示
调用方用 `load_model` 或手动挂上同名属性。这样 `AffinityRanker` 本身
（`model.py`，§5.4）继续保持与 tokenizer 无关。

### 5.9 `utils.py`

只放真正通用的小工具。

允许：

1. `set_seed(seed: int)`.
2. `hash_text(text: str)`.
3. `validate_amino_acid_sequence(seq: str)`.
4. `ensure_dir(path: Path)`.
5. 简单日志工具。

禁止：

1. 把数据集专用列名放进 utils。
2. 把 label 转换规则藏进 utils。
3. 把模型逻辑藏进 utils。

## 6. 函数编写标准

每个公共函数必须回答四个问题：

1. 输入是什么。
2. 输出是什么。
3. 失败时怎么办。
4. 用什么最小例子验收。

docstring 模板：

```python
def build_pairs(records: pd.DataFrame, max_pairs_per_group: int, seed: int) -> pd.DataFrame:
    """Build pairwise ranking examples within each group.

    Args:
        records: Standard processed table. Must contain record_id, group_id,
            rank_label, and label_kind.
        max_pairs_per_group: Maximum sampled pairs per group.
        seed: Random seed for reproducible sampling.

    Returns:
        DataFrame with pair_id, group_id, record_id_i, record_id_j, label_i,
        label_j, and y_ij.

    Raises:
        ValueError: If required columns are missing.
    """
```

通用代码规则：

1. 公共函数、公共类方法和 dataclass 字段必须有类型标注。
2. 函数名应表达动作或计算目标，优先使用动词开头；loss、metric 等约定名称可以例外。
3. 一个函数只做一件事，避免同时完成读取、转换、训练、保存等多个职责。
4. 函数不得硬编码隐藏路径，所有输入输出路径必须由参数或 Config 显式传入。
5. 不使用裸 except；必须捕获具体异常，并保留必要错误信息。
6. 不用 print 代替返回值、异常或 logging；提交代码前应移除临时 print。
7. 不用可变全局变量保存训练状态；训练状态必须由 Trainer、Config 或 checkpoint 管理。
8. 改变随机性的函数必须显式接收 seed 或 random generator。
9. 除非函数名明确表示 inplace，否则不得原地修改输入 DataFrame。
10. 模块不得越权调用：model 不依赖 dataset_id，loss 不构造 pair，trainer 不读取原始 CSV。

## 7. 测试与验收

### 7.1 数据脚本验收

每个 `prepare.sh` 合并前，组长检查：

1. 能从仓库根目录运行。
2. 输出标准表和 QC 表。
3. 标准表通过 `validate_processed_table.py`。
4. `rank_label` 方向有最小样例说明。
5. dropped records 有原因。
6. `group_id` 构造规则写在脚本注释或 README 中。

### 7.2 通用模块单测

必须覆盖：

```text
load_records rejects missing required columns
build_pairs never crosses group_id
build_pairs skips equal labels
build_pairs is reproducible with fixed seed
build_groups never crosses group_id
build_groups skips single-label groups
build_groups is reproducible with fixed seed
collate_rank_batch produces valid masks
collate_rank_batch builds antibody sequence per chain-combination rules (single-chain / heavy|light / heavy-only)
collate_rank_batch handles all-antigen-missing batch -> antigen_tokens is None
collate_rank_batch handles mixed batch with some antigen_sequence missing -> per-row mask
collate_pair_batch splits left/right RankBatch and produces y_ij
ranknet_loss rewards correct ordering
compute_group_spearman skips one-label groups
user_entry.rank_antibodies returns descending ranks
model.forward handles missing antigen without NaN
model.forward + loss supports backward without NaN gradients
```

### 7.3 最小集成测试

必须有一个 toy processed table，包含：

1. 一个 Fv group。
2. 一个 VHH group。
3. 一个 missing antigen group。
4. 一个 binary label group。
5. 一个只有单一 label 的 group。

集成测试要能完成：

```text
load config -> load records -> build pairs -> create dataloader
-> forward -> ranknet loss -> one optimizer step -> evaluate
```

## 8. 实验输出规范

每次训练输出到 `outputs/{run_id}/`。

必需文件：

```text
config.yaml
metrics.json
group_metrics.csv
predictions.csv
checkpoint.pt
run.log
```

`predictions.csv` 字段：

```text
record_id
dataset_id
group_id
rank_label
score
label_kind
```

`metrics.json` 至少包含：

```text
train_loss
valid_macro_spearman
valid_weighted_spearman
n_valid_groups
n_skipped_groups
```

## 9. 禁止项

1. 禁止写一个试图处理所有原始表格的通用 Python raw parser。
2. 禁止通用训练模块依赖原始 CSV 列名。
3. 禁止跨 `group_id` 构造 pair。
4. 禁止对异构亲和力指标做全局 Z-score 后混合回归。
5. 禁止把缺失抗原零向量作为有效 token 送入 attention。
6. 禁止静默丢弃记录。
7. 禁止只报 overall Spearman。
8. 禁止把 predicted label 和 experimental label 混成同一个主指标。
9. 禁止在 import package 时加载大模型或读取大数据。
10. 禁止在 baseline 跑通前把 MSA fusion 设为默认路径。

## 10. PR 代码评审清单

组长评审时按这个顺序看：

1. 这次 PR 属于数据脚本还是通用模块。
2. 如果是数据脚本，是否只产出标准表，不污染通用代码。
3. 如果是通用模块，是否只消费标准表，不理解原始数据列。
4. 新函数是否写清 input、output、failure。
5. 是否有最小测试覆盖正常输入和错误输入。
6. 是否改变 label 方向、group 构造、pair 采样或 metric 计算。
7. 是否可能造成数据泄漏。
8. 是否对大数据集做了不必要的全量内存操作。
9. 是否保存了配置、随机种子和输出指标。
10. 是否有任何过度生物学宣称。

## 11. 当前开发优先级

第一阶段只做这些：

1. 为几个核心 binding 数据集分别写 `prepare.sh`，产出标准表。
2. 写 `validate_processed_table.py`。
3. 写 `dataset.py` 和 `build_pairs`。
4. 写 `ranknet_loss` 和 `compute_group_spearman`。
5. 写最小 `AffinityRanker`，可以先 antibody-only。
6. 写 `Trainer` 跑通 toy data。
7. 写 `user_entry.rank_antibodies`，让外部用户能输入一个抗原和一组抗体，拿到排序结果。

先把这个闭环跑通，再讨论 Exact/MSA、AbLang-2、Cross-Attention 和 Attention Pooling 的复杂实现。
