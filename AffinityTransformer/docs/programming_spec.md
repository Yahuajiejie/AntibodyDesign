# AffinityTransformer 编程规范

版本：v0.2  
依据：新版 `README.md`  
角色设定：本文件按项目组长给开发组员发任务的方式编写。它不是论文说明，也不是泛泛的 Python 风格指南；它规定哪些代码该写、函数输入输出是什么、怎么验收、哪些行为禁止。

## 0. 总体判断

新版 `README.md` 的科学性基本站得住。项目把异构亲和力数据建模为同质 group 内排序学习，而不是全局绝对值回归；这与当前数据特征一致，也与 Spearman 评价指标一致。

当前没有必须推翻的科学问题，但实现时必须守住四条边界：

1. 模型输出的 `score` 只在同一 `group_id` 内比较，不解释为跨数据集的绝对亲和力。
2. `Kd`、`IC50`、`EC50`、`fitness`、`Pred_affinity`、`bind/no bind` 必须先明确方向和标签质量，再进入训练。
3. 缺失抗原信息可以标记，不能伪造成真实特征；attention 中的无效 token 必须被 mask 掉。
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
        AbRank/
          prepare.sh
          convert.py
        li2023machine/
          prepare.sh
          convert.py
        tsuruta2024avida_hIL6/
          prepare.sh
          convert.py
      validate_processed_table.py
  processed/
    binding/
      AbRank/
        records.parquet
        qc_summary.csv
        dropped_records.csv
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

## 3. 标准训练表 schema

每个数据处理脚本最终必须输出 `records.parquet` 或 `records.csv`。通用模块只接受这个 schema。

必需字段：

```text
record_id: str
dataset_id: str
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

group_id: str | None
keep_for_training: bool
drop_reason: str | None
```

字段规则：

1. `rank_label` 必须是越大越好。
2. `group_id` 至少由 `dataset_id`、`antigen_key`、`assay_name`、`metric_name`、`label_kind` 生成。
3. `label_kind = "binary"` 的记录只能产生正负 pair，同标签之间不能产生 pair。
4. `antigen_sequence = None` 时，`antigen_source` 必须是 `"missing"`。
5. `keep_for_training = False` 的记录可以留在表里，但 dataset 必须过滤掉它。

建议字段：

```text
metric_unit: str | None
transform_rule: str
sequence_hash: str
notes: str | None
```

## 4. 数据处理脚本规范

每个数据集脚本必须包含一个 `prepare.sh`，作为该数据集的唯一入口。

示例结构：

```text
scripts/prepare/binding/AbRank/
  prepare.sh
  convert.py
```

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
2. 输出固定写入 `processed/{category}/{dataset_id}/`。
3. 转换脚本必须生成 `records.parquet` 或 `records.csv`。
4. 同时生成 `qc_summary.csv` 和 `dropped_records.csv`。
5. 每次运行覆盖自己的输出目录可以接受，但不得删除其他数据集输出。
6. 如果脚本需要外部下载抗原序列，必须写入 `antigen_source` 和来源说明；早期 MVP 可以直接标记 missing。
7. 最后可以根据每个数据集、数据集的每个子目录的分表格，合成一个总表格

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

负责读取标准训练表、过滤可训练记录、并构造pairwise ranking训练样本。

本模块只处理已经通过 schema 校验的 processed table，不负责原始数据清洗、指标方向转换或 group_id 生成，具体流程如下：

```
processed table
→ schema validation
→ filter trainable records
→ build AffinityExample
→ build pairs if task = pairwise ranking
→ PairwiseDataset
→ collate_fn padding/mask
→ DataLoader
→ Trainer
```

现在可以先将这些功能放进一个文件，等到需要写的函数实在太多时，再把相关函数分配给不同文件放置，到时候可以这样组织文件

```
schema.py   标准训练表字段、枚举、校验
records.py  读取 processed table，过滤 keep_for_training
pairs.py    build_pairs
dataset.py  AffinityDataset / PairwiseAffinityDataset
collate.py  RankBatch / PairBatch / collate_fn
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

### 5.3 `dataloader.py`

如果默认 PyTorch dataloader 足够，可以不写复杂封装；但只要涉及 padding、mask、pair batch，就必须集中放在这里。

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
def collate_rank_batch(examples: Sequence[AffinityExample]) -> RankBatch:
    ...

def collate_pair_batch(examples: Sequence[PairExample]) -> PairBatch:
    ...
```

mask 约定：

1. `True = valid token`。
2. `False = padding or missing`。
3. 缺失抗原不能作为有效 token 进入 attention。
4. batch 内 Fv、scFv、VHH 混合时必须能 collate。
5. `mask.shape` 必须与 token 序列维度一致，即 `[B, L]`，而不是与 embedding 维度完全一致。

验收：

1. batch size 为 1 时可以运行。
2. 全部抗原缺失时不产生 NaN。
3. token ids 场景：`mask.shape == token_ids.shape`
   embedding 场景：`mask.shape == hidden.shape[:2]`

### 5.4 `model.py`

负责模型本体。早期不要拆太复杂。

最低接口：

```python
class AffinityRanker(nn.Module):
    def forward(self, batch: RankBatch) -> torch.Tensor:
        ...
```

输出：

```text
score: FloatTensor[B]
```

规则：

1. `score` 不经过 sigmoid、softmax 或 clamp。

2. 当 antigen_tokens is None 或 antigen_mask 全 False 时，模型不得执行普通 antigen attention。

   可以采用如下两种代替方案：

   ```
   方案 A：antibody-only baseline
   - 缺失抗原时，只使用 antibody representation 打分。
   
   方案 B：learned missing-antigen embedding
   - 使用一个可学习的 missing antigen token 作为抗原占位。
   - 该 token 必须被视为 valid token，而不是 padding。
   ```

3. attention 中的 invalid token 必须通过 additive mask 屏蔽。

4. 模型代码里不要写原始数据集名称。同时，模型 forward 只允许依赖 batch 中的张量字段，不应依赖 record_ids、group_ids 等元信息进行预测。

验收：

1. Fv 输入可以 forward。
2. VHH 输入可以 forward。
3. Fv 和 VHH 混合 batch 可以 forward。
4. 缺失抗原输入可以 forward，且输出无 NaN。

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

1. `n_unique_labels < 2` 的 group 不计算 Spearman。
2. 同时报告 macro average （这个需要考虑大batch与小batch的问题）和按 group size 加权 average。
3. 不允许只报 overall metric。
4. binary label 的 Spearman 必须单独统计。

### 5.7 `trainer.py`

负责训练循环、验证、checkpoint、日志。

最低接口：

```python
class Trainer:
    def fit(self) -> None:
        ...

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        ...

    def save_checkpoint(self, path: Path) -> None:
        ...
```

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
antibody_type: "Fv" | "scFv" | "VHH"
```

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
collate_rank_batch produces valid masks
ranknet_loss rewards correct ordering
compute_group_spearman skips one-label groups
user_entry.rank_antibodies returns descending ranks
model.forward handles missing antigen without NaN
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
