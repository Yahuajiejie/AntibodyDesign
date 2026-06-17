# AffinityTransformer 编程规范

版本：v0.7  
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
不变。字符串到具体模型/tokenizer 对象的映射留给 §5.7.1 的构造函数。
本次改动不影响 §5.1 `ModelConfig` 字段定义、§5.2 及更早内容。

v0.5 变更：实现 §5.6-§5.9（`metrics.py`/`trainer.py`/`user_entry.py`/
`utils.py`）及 §7.3 最小集成测试，并补充三处此前未落定的设计细节：

1. §5.6 `summarize_group_spearman`：在 `compute_group_spearman` 之上，按
   `label_kind` 分组（外加一个合并所有 group 的 `"overall"`）分别计算
   macro/weighted Spearman 和 valid/skipped group 计数，详见该节新增内容。
2. §5.7.1 `build_model_and_tokenizers`：提供最小的 encoder/tokenizer 构造函数，
   把 `ModelConfig.antibody_encoder` / `antigen_encoder` 字符串映射为真实模型对象。
   MVP 阶段不强制拆出 `encoders.py`；该函数可以先放在 `train.py`/`user_entry.py`
   附近，当前只支持固定的 ESM2 short-name 列表（`esm2_t6_8M` ...
   `esm2_t48_15B`）。"短名是否受支持"和"`d_model` 是否匹配"两项校验必须在
   import `transformers`、联网下载权重之前完成。其它 encoder 家族（如 AbLang-2）
   尚未实现。
3. §5.8 "比赛用户入口"：普通用户不需要传入 `AffinityRanker`。正式入口通过
   `model_name` 选择已调好的模型，由 `user_entry.py` 内部加载 checkpoint、
   config、tokenizer 并构造 `AffinityPredictor`；`AffinityRanker` 仅作为内部
   tensor 模型使用。

本次改动不影响 §5.1-§5.5 已实现内容，也不影响 §6/§9 的通用规则。

v0.5 本轮补充：补齐从 processed records 到可复现实验的训练闭环规范。新增
`requirements.txt`、根目录 `train.py` / `predict.py`、`scripts/prepare/binding/merge_records.py`
和 `affinity_transformer/splits.py` 的职责边界；明确 `train.py` 是训练总入口，
但 merge、split、dataset、trainer、metrics 仍保持可测试的独立模块；新增
`test_path`、自动 split 配置、split 泄露检查、all-records 合并产物和相关测试要求。
本轮补充不要求立即拆出独立 `encoders.py`；MVP 阶段可以先在 `train.py` /
`user_entry.py` 中保留一个很薄的 encoder/tokenizer 构造函数，等 encoder 家族变多
后再拆成单独模块。

v0.6 变更：修订 §5.2 pair 采样规范，解决大 group 训练前 OOM 的工程问题。
`build_pairs` 不得再先枚举 group 内所有两两组合再抽样；对于大 group，必须走
memory-safe 的 rank-label 分块采样。算法只在同一 `group_id` 内工作，先做
group 大小与 label 连续性诊断，再按 `rank_label` 排序切成等频 block，随后分别
采样跨 block pair 和同 block pair。跨 block pair 提供强排序监督，同 block pair
保留局部细粒度分辨能力。v0.6 不引入全局 `min_label_gap`，也不对 label 做正态化；
任何 label-gap 或 quantile-gap 过滤都必须作为后续显式消融，而不是救火补丁。

v0.7 变更：新增"采样器对照"规范。v0.6 的 memory-safe block sampler 必须保留，
作为默认稳态采样器；v0.7 在其下方新增 `quantile_difficulty` 采样器，用于比较
更科学的 pair 采样策略。新版采样器以 group 内经验分位数为核心，不假设 label
服从正态、均匀或任何参数分布；它先估计经验分位数 `q`，再按 `quantile_gap`
进入 easy/medium/hard/local 难度桶，并用少量随机 probe pair 估计该 group 自身的
raw label gap 下限。v0.7 的重点不是替换主实验默认采样器，而是让 g03 能比较
`block_quantile` 与 `quantile_difficulty` 在 pair 分布、运行时间、峰值内存和模型指标
上的差异。

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
  requirements.txt
  requirements-dev.txt              # 可选；只放开发/测试依赖
  train.py                          # 项目训练总入口
  predict.py                        # 外部用户/比赛推理入口
  docs/
    programming_spec_v0.4.md
    programming_spec_v0.5.md
    programming_spec_v0.6.md
    programming_spec_v0.7.md
  data/
    binding/
    expression/
    ...
  scripts/
    filter_records.py                # 兼容入口，转发到 scripts/data/filter_records.py
    data/
      inspect_records.py             # 全量/按数据集 QC 汇总
      filter_records.py              # 从标准表生成可复现训练子集
      build_splits.py                # 从标准表生成固定 train/valid/test split
    experiments/
      run_many.py                    # 按 config 列表批量运行 train.py
      collect_results.py             # 汇总 outputs/*/metrics.json
    runs/
      g00_qc_and_splits.sh           # 数据 QC、筛选、固定 split
      g01_core_ablation.sh           # 抗体 only / concat antigen / cross-attention
      g02_label_source_ablation.sh   # experimental / no_predicted / all label kinds
      g03_pair_sampling_ablation.sh  # pair 采样强度与比例采样
      g04_antigen_subset_ablation.sh # 抗原子集与去 AbRank 对照
    prepare/
      binding/
        prepare_all.sh
        merge_records.py
        gen_manifest.py
        manifest.csv
        antigen_missing_summary.csv
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
      all_records.parquet
      all_records_summary.csv
      splits/
        cov2_rbd_group_holdout/
          filtered_records.parquet
          filter_summary.csv
          train.parquet
          valid.parquet
          test.parquet
          split_summary.csv
          leakage_report.csv
        debug_record_split/
          train.parquet
          valid.parquet
          test.parquet
          split_summary.csv
          leakage_report.csv
        group_holdout_split/
          train.parquet
          valid.parquet
          test.parquet
          split_summary.csv
          leakage_report.csv
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
  configs/
    filters/
      cov2_rbd.yaml
      g00_max_antigen_context.yaml
      g02_experimental_only.yaml
      g02_no_predicted.yaml
      g04_cov2_rbd.yaml
      g04_influenza_ha.yaml
      g04_lysozyme.yaml
      g04_no_abrank.yaml
      g04_vegf.yaml
    experiments/
      g01_maxctx_antibody_only.yaml
      g01_maxctx_concat_antigen.yaml
      g01_maxctx_cross_attention.yaml
      g02_*.yaml
      g03_*.yaml
      g03_block_quantile_abs200.yaml
      g03_quantile_difficulty_abs200.yaml
      g03_quantile_difficulty_quota.yaml
      g04_*.yaml
    model_registry.yaml
    debug_toy.yaml
    baseline_antibody_only_ranknet.yaml
    baseline_group_holdout_ranknet.yaml
    ablation_concat_antigen_ranknet.yaml
    ablation_cross_attention_ranknet.yaml
    cross_attention_cov2_rbd_ranknet.yaml
  affinity_transformer/
    __init__.py
    config.py
    record_filter.py
    splits.py
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
    test_splits.py
    test_train_entry.py
    test_predict_entry.py
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
8. `train.py` 是对外训练入口，允许一条命令完成 split、训练、验证和测试评估；但它只负责编排，不实现 pair 构造、loss、metric、模型 forward 或 raw 数据转换。
9. `affinity_transformer/splits.py` 是 split 逻辑的唯一归属；`train.py` 调用它，测试也直接测它。不要把 split 规则散落在 notebook、shell 或 trainer 里。
10. `scripts/prepare/binding/merge_records.py` 只合并已经标准化的 `records.parquet`，不读取原始 CSV，不理解原始列名。
11. `requirements.txt` 用于最小复现；远程服务器上的 CUDA 版 `torch` 可以按服务器环境单独安装，不要在通用依赖里写死某个 CUDA wheel。
12. `predict.py` 是外部用户/比赛推理入口，负责读取用户输入表、选择模型、调用
    `user_entry.py`，并写出排序结果；它不负责训练、split 或数据集 prepare。
13. `scripts/runs/gXX_*.sh` 是服务器运行入口。组号只表示实验组，不表示代码模块；
    脚本必须调用 `scripts/data/`、`scripts/experiments/` 和 `train.py`，不得把核心
    split、pair、metric 或模型逻辑写进 shell。

### 2.1 运行环境与依赖

`requirements.txt` 放最小运行依赖：

```text
pandas
pyarrow
pyyaml
scipy
transformers
torch
```

`requirements-dev.txt` 可选，放测试和开发依赖：

```text
pytest
```

规则：

1. 不在 `requirements.txt` 中写死某个 CUDA 专用 wheel URL。
2. `torch` 版本可以设宽松下限；服务器上如果需要 CUDA 版 PyTorch，由部署说明或集群环境单独处理。
3. 所有测试依赖必须能在 CPU 环境安装；GPU 只影响训练速度，不应影响导入、配置解析和 toy tests。
4. 不把 notebook、画图、下载器等非 MVP 依赖塞进主 requirements。

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
7. `antigen_source = "retrieved"` 时，对应 `convert.py` 必须额外定义常量 `ANTIGEN_SOURCE_NOTE = "retrieved: <accession>, <片段范围/物种等依据>"`。`ANTIGEN_SOURCE_NOTE` 不是 `records.parquet` 的字段（`antigen_source` 列仍只取 `"provided" | "retrieved" | "missing"` 枚举值，满足 `validate_processed_table.py` 的校验），仅供 `gen_manifest.py` 提取后写入 `manifest.csv` 的 `notes` 列，作为随 git 跟踪、可追溯的抗原来源记录。



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

`scripts/prepare/binding/gen_manifest.py` 从所有 `convert.py` 的常量定义中自动重新生成两个元数据文件，必须可重复运行且结果确定（不依赖网络、不依赖运行顺序）：

- `scripts/prepare/binding/manifest.csv` —— 全部 binding 源表清单
- `scripts/prepare/binding/antigen_missing_summary.csv` —— 抗原检索参考表，数据源是脚本内人工维护的 `ANTIGEN_REFS` 列表

两个文件生成后都要复制到 `processed/binding/`；`prepare_all.sh` 在运行任何 `prepare.sh` 之前必须先执行一次 `gen_manifest.py`。

`manifest.csv` 必须记录所有 binding 源表，列为：

```text
study_id          数据集标识，对应 scripts/prepare/binding/{study_id}/
table_id          子表标识，对应 .../{study_id}/{table_id}/
csv_name          源 CSV 文件名（取自 SOURCE_FILE），用于追溯原始数据
antibody_type     Fv | scFv | VHH | Fab | IgG | unknown
antigen_key       抗原标识，与 antigen_missing_summary.csv 的 antigen_key 对应
antigen_name      抗原名称
antigen_source    provided | retrieved | missing
metric_name       指标名
label_kind        experimental | predicted | binary | unknown
status            ready | blocked（见下）
notes             备注，由脚本根据 convert.py 内容自动推断（zip/VHH裁剪/抗原缺失等）
```

早期草案中的 `source_file`、`reason` 分别由 `csv_name`、`notes` 承担；`prepare_dir`（`scripts/prepare/binding/{study_id}/{table_id}/`）与 `processed_dir`（`processed/binding/{study_id}/{table_id}/`）不作为独立列，按命名规则 4 由 `study_id`/`table_id` 直接推出，不再重复存储。

`status` 只允许：

```text
ready
blocked
```

`ready` 表示该表已实现且 `prepare.sh` 可跑通——即使 `antigen_source = "missing"` 也算 `ready`，因为缺失抗原不阻塞训练记录生成（见 §3 规则 5）。`blocked` 表示存在结构性问题（例如源 CSV 没有抗原列、无法构造 `group_id`），手工维护于 `gen_manifest.py` 的 `BLOCKED` 列表，原因写入 `notes`。`planned`、`excluded` 为预留取值，分别供"已规划但未实现"、"明确排除"的源表使用；目前所有源表均为 `ready` 或 `blocked`。

`antigen_missing_summary.csv` 列为：

```text
antigen_key            抗原标识，与 manifest.csv 的 antigen_key 对应
antigen_name           抗原名称
is_protein             是否为蛋白质（小分子抗原如 fluorescein 为 False）
likely_uniprot_or_pdb  候选 UniProt accession 或 PDB ID
antigen_species        物种
retrieval_notes        检索依据、片段范围、不确定性说明
source_url             由 likely_uniprot_or_pdb 推导的可点击来源链接
```

`source_url` 推导规则：`likely_uniprot_or_pdb` 形如 UniProt accession（如 `P00698`）时取 `https://www.uniprot.org/uniprotkb/{accession}/entry`；形如 PDB ID（如 `7LYL`）时取 `https://www.rcsb.org/structure/{PDB_ID}`；无法映射为单一记录（如 "check paper"、列出多个候选）时留空。

`manifest.csv` 中 `antigen_source == "missing"` 的每个 `antigen_key`，都应在 `antigen_missing_summary.csv` 中有对应行（来自 `ANTIGEN_REFS`），作为待检索清单。一旦某个 `antigen_key` 在所有引用它的 `convert.py` 中都已写入 `ANTIGEN_SOURCE = "retrieved"`（及 `ANTIGEN_SOURCE_NOTE`），应将其从 `ANTIGEN_REFS` 中移除，使重新生成的 `antigen_missing_summary.csv` 自动只反映剩余未解决的抗原；已解决抗原的溯源记录永久保留在对应 `convert.py` 的 `ANTIGEN_SOURCE_NOTE` 中（随 git 跟踪）。

规则：

1. 当前 86 个 binding 源文件都必须在 manifest 中出现。
2. 暂不处理的源文件也必须写入 manifest，并在 `notes` 中给出 `blocked`（或未来 `excluded`）的原因。
3. `prepare_all.sh` 只读取 manifest 中 `status = "ready"` 的行并逐个运行对应 `prepare.sh`。

### 标准表合并器 `merge_records.py`

`scripts/prepare/binding/merge_records.py` 负责把所有 `status = "ready"` 的
binding 处理单元合并为一个标准训练总表。它仍属于数据准备阶段，但它处理的是
标准表，不是原始表。

最低接口：

```python
def merge_binding_records(
    manifest_path: Path,
    processed_root: Path,
    output_path: Path,
    summary_path: Path,
) -> pd.DataFrame:
    ...
```

输入：

```text
scripts/prepare/binding/manifest.csv
processed/binding/{study_id}/{table_id}/records.parquet
```

输出：

```text
processed/binding/all_records.parquet
processed/binding/all_records_summary.csv
```

`all_records.parquet` 要求：

1. 只合并 manifest 中 `status = "ready"` 的条目。
2. 每个输入表必须先通过 `validate_processed_table.py`。
3. 必须保留 `record_id`、`study_id`、`table_id`、`dataset_id`、`source_file`。
4. 全局 `record_id` 必须唯一；重复时立刻报错，不允许自动重命名。
5. 不允许在合并阶段重新计算 `group_id`、`rank_label` 或 `antigen_source`。
6. 不允许因为某个 group 太小而在合并阶段丢弃记录；是否进入训练由
   `keep_for_training` 和后续 dataset/split 逻辑决定。
7. 输出排序必须稳定，建议按 `dataset_id, record_id` 排序，方便 diff 和排查。

`all_records_summary.csv` 至少包含：

```text
dataset_id
study_id
table_id
n_records
n_trainable_records
n_groups
n_trainable_groups
label_kind_counts
antigen_source_counts
records_path
```

验收：

1. 合并后的行数等于所有 ready `records.parquet` 行数之和。
2. 合并后的 `dataset_id` 集合等于 ready manifest 的 `dataset_id` 集合。
3. 任一 ready 表缺失或 schema 不合格时失败。
4. `merge_records.py` 中不得出现任何原始 CSV 的列名。

### 抗原序列检索工作流

为避免一次性大批量写入 `ANTIGEN_SEQ`/`ANTIGEN_SOURCE` 导致错误扩散到多张表，抗原序列检索必须按以下流程进行：

1. **分批**：以 `antigen_missing_summary.csv` 的行为单位，每批处理少量（建议 3–5 个）`antigen_key`。
2. **候选先行**：每个 `antigen_key` 先给出候选 `likely_uniprot_or_pdb`、`source_url` 及匹配依据（物种、序列长度、关键词、与论文描述的一致性），不直接修改 `convert.py`；依据不充分的（名称模糊、多个候选、物种不确定）必须标注为低置信度，等待人工确认。
3. **确认后写入**：经确认/修正后才写入对应 `convert.py` 的 `ANTIGEN_SEQ`/`ANTIGEN_SOURCE`/`ANTIGEN_SOURCE_NOTE`：`ANTIGEN_SEQ` 为实际序列；`ANTIGEN_SOURCE = "retrieved"`（枚举值，写入 `records.parquet`/`manifest.csv` 的 `antigen_source` 列）；`ANTIGEN_SOURCE_NOTE = "retrieved: <accession>, <片段范围/物种等依据>"`（详细溯源，与 `antigen_missing_summary.csv` 的 `source_url` 对应，确保可溯源，见 §3 规则 7）。
4. **同步移除待办项**：对每个已写入 `ANTIGEN_SOURCE = "retrieved"` 的 `antigen_key`，从 `gen_manifest.py` 的 `ANTIGEN_REFS` 中移除对应条目。
5. **重新生成与验证**：每批完成后运行 `gen_manifest.py` 重新生成 `manifest.csv`/`antigen_missing_summary.csv`，重新运行受影响表的 `prepare.sh`，并跑通测试套件。
6. 新增的检索/生成机制（如本节、`gen_manifest.py`、`ANTIGEN_REFS`、`antigen_missing_summary.csv` 及其列定义）必须先写入本文档，再实现。

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
6. 如果脚本需要外部下载抗原序列，必须遵循「抗原序列检索工作流」一节，写入 `antigen_source` 和来源说明（并与 `antigen_missing_summary.csv` 的 `source_url` 对应）；早期 MVP 可以直接标记 `missing`。
7. `prepare_all.sh` 可以在所有 ready 单元处理完成后调用 `merge_records.py`，生成
   `processed/binding/all_records.parquet` 和 `processed/binding/all_records_summary.csv`。
   不允许在 `prepare_all.sh` 里重新写一份合并逻辑。

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
    all_records_path: Path | None
    train_path: Path | None
    valid_path: Path | None
    test_path: Path | None
    split_strategy: "none" | "debug_record_split" | "group_holdout_split"
    split_dir: Path | None
    valid_fraction: float
    test_fraction: float
    max_pairs_per_group: int
    pair_sampler: "block_quantile" | "quantile_difficulty"
    pair_sample_strategy: "absolute_cap" | "capped_proportional"
    pair_fraction: float | None
    min_pairs_per_group: int
    large_group_threshold: int = 10000
    pair_enumeration_limit: int = 100000
    label_block_count: int = 5
    intra_block_pairs_per_large_group: int = 50
    discrete_label_unique_threshold: int = 32
    discrete_label_ratio_threshold: float = 0.05
    difficulty_bucket_quotas: dict[str, float] | None = None
    difficulty_bucket_edges: dict[str, tuple[float, float]] | None = None
    gap_probe_pairs_per_group: int = 10000
    raw_gap_min_quantile: float = 0.05
    min_quantile_gap: float = 0.0
    label_kind_pair_quotas: dict[str, float] | None = None
    dataset_pair_quotas: dict[str, float] | None = None
    seed: int
    record_filter: RecordFilterConfig  # YAML 字段名为 data.filter

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

数据路径有两种合法模式：

1. **显式 split 模式**：`split_strategy = "none"`，调用方直接提供
   `train_path`，`valid_path` 和 `test_path` 可选。该模式用于复现已经切好的实验。
2. **自动 split 模式**：`split_strategy` 为 `debug_record_split` 或
   `group_holdout_split`，调用方提供 `all_records_path` 和 `split_dir`；`train.py`
   调用 `affinity_transformer/splits.py` 生成 `train.parquet`、`valid.parquet` 和
   `test.parquet` 后再训练。

`valid_fraction` 和 `test_fraction` 只在自动 split 模式下生效。`test_path` 只用于
训练结束后的最终评估，不参与 early stopping，不用于选择 checkpoint。

`filter` 为可选字段，只在自动 split 模式下生效。它先作用于
`all_records_path`，再进入 `splits.py`。如果启用了非空筛选条件，`train.py`
必须在 `split_dir` 写出：

```text
filtered_records.parquet
filter_summary.csv
```

显式 split 模式下，调用方已经直接指定 `train_path`/`valid_path`/`test_path`，
因此不再自动应用 `filter`；需要子集实验时应先用 `scripts/filter_records.py`
生成子集文件，再以 `split_strategy = "none"` 复现实验。

验收：

1. 缺少必要字段时抛出 `ValueError`。
2. 路径不存在时抛出 `FileNotFoundError`。
3. 不在函数里偷偷改默认随机种子。
4. `split_strategy = "none"` 时，`train_path` 必须存在。
5. `split_strategy != "none"` 时，`all_records_path` 必须存在，`split_dir` 可以不存在但必须可创建。
6. `valid_fraction + test_fraction` 必须大于 0 且小于 1。
7. `filter` 字段未知或类型错误时抛出 `ValueError`，不得静默忽略。

### 5.1.1 `record_filter.py`

负责从标准 processed table 中选择训练子集。它只消费 `all_records.parquet`
或已切好的标准表，不读取原始 CSV，不改写 label，不重新生成 `group_id`。

典型用途：

1. 指定一个或几个数据集作为训练集。
2. 指定单抗原或一组抗原训练。
3. 指定单抗体或一组抗体训练。
4. 指定抗原-抗体 pair 训练。
5. 做抗原来源、label_kind、抗体类型等消融实验。

建议对象：

```python
@dataclass
class AntigenAntibodyPair:
    antigen_key: str
    antibody_id: str

@dataclass
class RecordFilterConfig:
    include_dataset_ids: tuple[str, ...]
    exclude_dataset_ids: tuple[str, ...]
    include_study_ids: tuple[str, ...]
    exclude_study_ids: tuple[str, ...]
    include_table_ids: tuple[str, ...]
    exclude_table_ids: tuple[str, ...]
    include_antigen_keys: tuple[str, ...]
    exclude_antigen_keys: tuple[str, ...]
    include_antibody_ids: tuple[str, ...]
    exclude_antibody_ids: tuple[str, ...]
    include_antibody_sequence_hashes: tuple[str, ...]
    exclude_antibody_sequence_hashes: tuple[str, ...]
    include_group_ids: tuple[str, ...]
    exclude_group_ids: tuple[str, ...]
    include_record_ids: tuple[str, ...]
    exclude_record_ids: tuple[str, ...]
    include_label_kinds: tuple[str, ...]
    exclude_label_kinds: tuple[str, ...]
    include_antibody_types: tuple[str, ...]
    exclude_antibody_types: tuple[str, ...]
    include_antigen_sources: tuple[str, ...]
    exclude_antigen_sources: tuple[str, ...]
    include_antigen_antibody_pairs: tuple[AntigenAntibodyPair, ...]
    require_antigen_sequence: bool
    require_antibody_id: bool
    min_records_per_group: int | None
    min_trainable_records_per_group: int | None
    min_unique_labels_per_group: int | None
```

核心函数：

```python
def filter_records(records: pd.DataFrame, config: RecordFilterConfig) -> pd.DataFrame:
    ...

def antibody_sequence_hashes(records: pd.DataFrame) -> pd.Series:
    ...

def write_filter_outputs(
    before: pd.DataFrame,
    after: pd.DataFrame,
    config: RecordFilterConfig,
    output_path: Path,
    summary_path: Path,
) -> None:
    ...
```

筛选语义：

1. 同一个 `include_*` 字段内多个值是 OR。
2. 不同 `include_*` 字段之间是 AND。
3. `exclude_*` 在 include 之后执行，命中即排除。
4. `include_antigen_antibody_pairs` 表示 `(antigen_key, antibody_id)` 精确匹配；
   如果某些数据集没有 `antibody_id`，应使用 `include_record_ids` 或
   `include_antibody_sequence_hashes`。
5. `include_antibody_sequence_hashes` 使用
   `heavy_chain + "|" + light_chain + "|" + single_chain_sequence` 的稳定 hash，
   用于解决不同数据集没有统一 `antibody_id` 的情况。
6. `min_records_per_group`、`min_trainable_records_per_group` 和
   `min_unique_labels_per_group` 在记录级筛选之后执行。
7. 筛选结果必须保留原始 schema 和原始 `record_id`，不得重新编号。

示例：单抗原/一组抗原训练。

```yaml
data:
  all_records_path: processed/binding/all_records.parquet
  split_strategy: group_holdout_split
  split_dir: processed/binding/splits/cov2_rbd_group_holdout
  filter:
    include_antigen_keys:
      - CoV2_Wuhan_RBD
      - CoV2_WT_RBD
      - CoV2_RBD_representative
    require_antigen_sequence: true
    min_trainable_records_per_group: 2
    min_unique_labels_per_group: 2
```

示例：指定若干数据集训练。

```yaml
filter:
  include_dataset_ids:
    - phillips2021binding/cr9114_h1_kd
    - phillips2021binding/cr9114_h3_kd
```

示例：指定抗原-抗体 pair。

```yaml
filter:
  include_antigen_antibody_pairs:
    - antigen_key: VEGF_A
      antibody_id: G6
```

CLI：

```bash
python scripts/data/filter_records.py \
  --input processed/binding/all_records.parquet \
  --filter-config configs/filters/cov2_rbd.yaml \
  --output processed/binding/filtered/cov2_rbd/all_records.parquet
```

`scripts/filter_records.py` 仅作为兼容入口保留，新脚本应调用
`scripts/data/filter_records.py`。

输出：

```text
filtered_records.parquet 或用户指定的 output
filter_summary.csv
```

`filter_summary.csv` 至少包含：

```text
stage
n_records
n_trainable_records
n_groups
n_dataset_ids
label_kind_counts
antigen_source_counts
filter_active
```

禁止：

1. 禁止在 `record_filter.py` 中读取原始 CSV。
2. 禁止在筛选阶段修改 `rank_label`、`group_id` 或 `antigen_sequence`。
3. 禁止筛选失败时静默回退到全量数据。
4. 禁止把筛选逻辑散落到 `train.py`、`trainer.py` 或 notebook 中。

### 5.1.2 `splits.py`

负责从 `all_records.parquet` 生成训练/验证/测试集，并输出泄露检查报告。它只消费
标准表，不读取原始 CSV，也不负责训练。

建议对象：

```python
@dataclass
class SplitResult:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    summary: pd.DataFrame
    leakage_report: pd.DataFrame
```

核心函数：

```python
def build_splits(
    records: pd.DataFrame,
    strategy: str,
    valid_fraction: float,
    test_fraction: float,
    seed: int,
) -> SplitResult:
    ...

def write_splits(result: SplitResult, output_dir: Path) -> None:
    ...
```

支持策略：

```text
debug_record_split
group_holdout_split
```

`debug_record_split`：

1. 用于本地 smoke test 和快速跑通模型。
2. 只保证 `record_id` 在 train/valid/test 之间不重叠。
3. 允许同一个 `group_id` 跨 split，因此不得作为正式实验结果汇报。
4. `split_summary.csv` 中必须写明 `strategy = debug_record_split`。

`group_holdout_split`：

1. 用于正式 baseline 和主要消融实验。
2. `group_id` 不允许跨 train/valid/test 重叠。
3. 每个 split 内只保留原始记录，不重新计算 label 或 group。
4. valid/test 中可计算 Spearman 的 group 至少应有 2 条 `keep_for_training = True`
   且 `rank_label` 有效的记录；不足时不强行复制样本，但必须在 summary 中报告。
5. 对长尾数据，单个 group 的记录数如果超过单个 holdout split 的目标记录数，
   该 group 应优先固定在 train。其余 group 再按 group 数比例随机切入
   train/valid/test，避免一个超大 group 被随机扔进 valid/test，也避免 train
   只剩少数超大 group。

`split_summary.csv` 至少包含：

```text
split
strategy
n_records
n_trainable_records
n_groups
n_trainable_groups
n_spearman_eligible_groups
label_kind_counts
antigen_source_counts
```

`leakage_report.csv` 至少包含：

```text
check_name
status
n_violations
details
```

最低泄露检查：

1. `record_id` 不允许跨 split 重叠。
2. `group_holdout_split` 下 `group_id` 不允许跨 split 重叠。
3. 如果后续加入 `antibody_sequence_holdout`，则标准化后的抗体序列不允许跨 split 重叠。
4. 任一必需泄露检查失败时，`build_splits` 必须抛错，不允许只写 warning。

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
    pair_sampler: str = "block_quantile",
    pair_sample_strategy: str = "absolute_cap",
    pair_fraction: float | None = None,
    min_pairs_per_group: int = 1,
    large_group_threshold: int = 10000,
    pair_enumeration_limit: int = 100000,
    label_block_count: int = 5,
    intra_block_pairs_per_large_group: int = 50,
    discrete_label_unique_threshold: int = 32,
    discrete_label_ratio_threshold: float = 0.05,
    difficulty_bucket_quotas: dict[str, float] | None = None,
    difficulty_bucket_edges: dict[str, tuple[float, float]] | None = None,
    gap_probe_pairs_per_group: int = 10000,
    raw_gap_min_quantile: float = 0.05,
    min_quantile_gap: float = 0.0,
    label_kind_pair_quotas: dict[str, float] | None = None,
    dataset_pair_quotas: dict[str, float] | None = None,
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
9. pair 采样策略由 `pair_sample_strategy` 控制：
   `absolute_cap` 表示每个 group 最多取 `max_pairs_per_group` 个 pair；
   `capped_proportional` 表示按 `ceil(candidate_pairs * pair_fraction)` 取样，
   同时受 `min_pairs_per_group` 和 `max_pairs_per_group` 约束。
   比例采样不得复制样本；候选 pair 少于目标数量时保留全部候选 pair。
10. 对小 group，允许精确枚举候选 pair 后抽样。小 group 必须同时满足：
    `n_trainable_records < large_group_threshold` 且
    `candidate_pairs <= pair_enumeration_limit`。
11. 对大 group，禁止构造完整候选 pair 列表。大 group 必须走分块采样，内存复杂度
    不得随 `candidate_pairs` 二次增长。

**采样器选择**

`pair_sampler` 控制采样器实现：

1. `block_quantile`：v0.6 默认采样器。小 group 可以精确枚举，大 group 走
   rank-label 等频分块，采样跨 block pair 和少量同 block pair。
2. `quantile_difficulty`：v0.7 对照采样器。所有大 group 都基于经验分位数、
   难度桶、raw gap probe 和可选 quota 采样。该采样器暂不替换默认主实验，先用于
   g03 采样器消融和性能比较。
3. 未知 `pair_sampler` 必须抛 `ValueError`，不得静默退化为默认采样器。

实现时不要删除或重写 `block_quantile`。`quantile_difficulty` 的 helper 函数应放在
现有大 group block sampler 下方，入口由 `pair_sampler` 分发，便于同一份数据、同一
seed 下做 A/B 对照和性能 profiling。

**`block_quantile` 大 group 分块采样规则**

该规则只改变采样实现，不改变 pair 的科学语义：仍然只比较同一 `group_id` 内
`rank_label` 不相等的记录。

1. 对每个 `group_id`，先基于 `filter_trainable_records` 的输出计算：
   `n_records`、`n_unique_labels`、`unique_label_ratio` 和
   `candidate_pairs = C(n_records, 2) - sum_label C(count(label), 2)`。
2. 连续/离散诊断只用于选择高效采样路径和写日志，不用于过滤 label：
   `label_kind == "binary"`、`n_unique_labels <= discrete_label_unique_threshold`
   或 `unique_label_ratio <= discrete_label_ratio_threshold` 视为离散/重复标签 group；
   其余视为连续型 group。连续型 group 不需要原始尺度上的 `min_label_gap`。
3. `label_kind == "binary"` 或 `n_unique_labels == 2` 的大 group 不进入 quantile block
   分块。它们必须走 class-aware label bucket sampler：直接在两个 label bucket
   之间采样，输出不同 label 的 pair；不生成同 label pair，也不采同 block pair。
4. 其它大 group 按 `rank_label` 从低到高排序，`record_id` 作为稳定 tie-breaker；再按样本数切成
   `label_block_count` 个尽量等大的等频 block。不得按原始 label 等宽切分，也不得假设
   label 服从正态或均匀分布。
5. 跨 block 采样：目标数量由既有 `pair_sample_strategy` 决定，通常为
   `max_pairs_per_group`。每次抽两个不同 block，再从两个 block 中抽取
   `rank_label` 不相等的记录。block-pair 可以按 block 距离或可用候选 pair 数加权，
   但最终输出必须去重且固定 seed 可复现。
6. 同 block 采样：额外抽取最多 `intra_block_pairs_per_large_group` 个同 block 内
   `rank_label` 不相等的 pair，用于保留局部细粒度排序信息。若某个 block 内所有
   label 相同，则该 block 不贡献同 block pair。
7. 输出 pair 的 `record_id_i` / `record_id_j` 按稳定规则排序；`y_ij` 只由两个
   `rank_label` 的大小决定。排序规则不得被误用为"左边一定更强"。
8. 若 group 的合法候选 pair 少于目标数量，保留能采到的全部合法 pair；不得复制 pair
   或生成反向重复 pair 来凑数。
9. 大 group 分块采样的目标不是重塑 label 分布，而是在不枚举 `O(n^2)` pairs 的前提下，
   同时获得强排序监督（跨 block）和局部排序监督（同 block）。

**`quantile_difficulty` 采样器规则**

该采样器用于研究"更科学 pair 采样"是否优于 `block_quantile`。它不假设 label
服从正态、均匀或任何参数分布，只使用 group 内经验排序和少量随机 probe。

算法步骤：

1. 对每个 `group_id` 先调用 `filter_trainable_records`，只保留
   `keep_for_training = True` 且 `rank_label` 有限的记录。
2. 在 group 内按 `(rank_label, record_id)` 升序排序，计算经验分位数 `q`：
   `q = rank_index / (n_records - 1)`；当 `n_records = 1` 时该 group 不产生 pair。
3. 不枚举全量 pair。采样器通过随机抽 record index 生成候选 pair；候选 pair 必须
   同 group、不同 record、`rank_label` 不相等，并经 canonical order 去重。
4. 对每个候选 pair 计算：
   `quantile_gap = abs(q_i - q_j)` 和 `raw_label_gap = abs(label_i - label_j)`。
5. 先用最多 `gap_probe_pairs_per_group` 个随机候选 pair 估计 raw label gap 下限：
   收集非零 `raw_label_gap`，取其 `raw_gap_min_quantile` 分位数作为
   `raw_label_gap_min`。若 probe 得不到足够非零 gap，则退化为只要求
   `label_i != label_j`。`label_kind = "binary"` 的 group 不额外应用 raw gap 下限。
6. 若 `raw_label_gap < raw_label_gap_min`，候选 pair 进入 rejected 计数，不进入训练；
   若 `quantile_gap < min_quantile_gap`，同样拒绝。默认 `min_quantile_gap = 0.0`，
   即不启用全局分位数 gap 过滤。
7. 对通过 gap 过滤的候选 pair，按 `quantile_gap` 进入难度桶。默认桶定义：
   `local: [0.01, 0.05)`，`hard: [0.05, 0.20)`，
   `medium: [0.20, 0.50)`，`easy: [0.50, 1.00]`。
   `quantile_gap < 0.01` 的 pair 默认视为过近，除非调用方显式修改
   `difficulty_bucket_edges`。
8. 按 `difficulty_bucket_quotas` 从各桶取样。默认配额：
   `easy = 0.40`、`medium = 0.30`、`hard = 0.20`、`local = 0.10`。
   配额只决定目标比例，不允许复制 pair；某桶候选不足时，其剩余名额可按
   `easy -> medium -> hard -> local` 的顺序重新分配给仍有候选的桶。
9. 采样器还可受 `label_kind_pair_quotas` 和 `dataset_pair_quotas` 控制。quota
   只在单次调用的全局 pair table 上生效，用于防止 predicted/binary 或超大 dataset
   支配训练。quota 默认关闭；关闭时只受 per-group `max_pairs_per_group` 控制。
10. 固定 `seed` 时，probe、候选生成、桶内采样、quota 重分配和最终 pair 排序都必须
    可复现。

`quantile_difficulty` 输出不得声称 pair 的 raw label 分布均匀。它的目标分布是
"难度桶配额"，即按 `quantile_gap` 近似控制 easy/medium/hard/local 的比例。
如果需要研究 raw label 差值分布，必须额外输出报告，不要在训练逻辑中隐式假设。

**采样器性能与分布报告**

任何新增或修改 pair sampler 时，都必须能生成采样器 QC 报告。建议输出到
`reports/pair_sampling/`，至少包含：

```text
pair_sampler
config_name
n_input_records
n_trainable_records
n_groups_seen
n_groups_with_pairs
n_pairs_total
n_pairs_by_label_kind
n_pairs_by_dataset
n_pairs_by_difficulty_bucket
quantile_gap_min
quantile_gap_p05
quantile_gap_p50
quantile_gap_p95
raw_label_gap_min
raw_label_gap_p05
raw_label_gap_p50
raw_label_gap_p95
n_rejected_same_label
n_rejected_raw_gap
n_rejected_quantile_gap
build_pairs_wall_seconds
build_pairs_peak_rss_mb
```

比较 `block_quantile` 与 `quantile_difficulty` 时，至少报告三类结果：

1. 采样分布：difficulty bucket、label_kind、dataset、group-level pair 数。
2. 工程性能：`build_pairs_wall_seconds`、`build_pairs_peak_rss_mb`、最终 pair 数。
3. 训练效果：同一 split、同一模型、同一 epoch 设置下的 valid/test group-level
   Spearman，且必须按 `label_kind` 汇总。

验收：

1. 固定 seed 时 pair 结果可复现。
2. 输出 pair 不跨 group。
3. 空 group 或单一标签 group 不报错，但不产生 pair。
4. `capped_proportional` 缺少 `pair_fraction` 或 `pair_fraction` 不在 `(0, 1]`
   时必须抛 `ValueError`。
5. 构造 20,000 条以上、单个 `group_id` 的连续型 toy records 时，`build_pairs`
   必须在不枚举全部候选 pair 的情况下返回有限数量 pair，且内存占用不得随
   `C(n, 2)` 增长。
6. 构造严重不均衡的 binary toy records 时，大 group 采样仍必须产生正负之间的 pair，
   不得因为随机反复抽到同类而超时，也不得调用 quantile block 构造。
7. 大 group 采样结果应同时覆盖跨 block pair；当 block 内存在至少两个不同 label 时，
   应覆盖同 block pair。
8. `pair_sampler = "quantile_difficulty"` 时，输出 pair table 必须能关联到
   `quantile_gap`、`raw_label_gap` 和 difficulty bucket 统计；训练用 pair table
   可以不保存这些调试列，但 sampler QC 报告必须保存。
9. `quantile_difficulty` 固定 seed 后结果可复现；改变 seed 后允许 pair 顺序和样本
   变化，但仍必须满足不跨 group、不复制 pair、不生成同 label pair。
10. `raw_gap_min_quantile`、`min_quantile_gap`、`difficulty_bucket_quotas` 配置非法时
    必须抛 `ValueError`。
11. quota 开启时，采样器必须报告 quota 前后的 pair 数，不得静默丢弃大量数据。

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

HuggingFace 的 `PreTrainedTokenizerBase`（ESM2、AbLang-2 等）天然满足该签名；其他 tokenizer 如接口不同，由调用方包一层适配后再传入。`attention_mask` 中 1=真实 token / 0=padding，与下面 mask 约定（True=valid / False=padding）方向一致，转换时只需 `.bool()`，不需要取反。tokenizer 对象本身、以及"`ModelConfig.antibody_encoder`/`antigen_encoder` 字符串 -> 具体 tokenizer/模型"的映射，都是 §5.7.1 构造函数的职责，不在本模块内构造。

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

即：输入 token ids 与 mask（mask 约定同 §5.3，True=valid），输出每个 token 的隐藏表示，最后一维等于 `d_model`，且对任意输入（包括某一行 `attention_mask` 全 `False`）都不产生 `NaN`。真实场景下这是一个"预训练模型 + 投影到 `d_model` 的适配器"，适配器的构造属于 §5.7.1 构造函数，不在本模块范围内。

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

 **`build_model_and_tokenizers`：encoder/tokenizer 构造函数**

```python
def build_model_and_tokenizers(
    model_config: ModelConfig,
) -> tuple[AffinityRanker, Tokenizer, Tokenizer | None]:
    ...
```

这是"字符串 -> 具体模型/tokenizer 对象"的最小构造函数。MVP 阶段不强制新建
`affinity_transformer/encoders.py`：如果当前只支持 ESM2 antibody-only baseline，
该函数可以先放在根目录 `train.py` 中，并由 `user_entry.py` 复用；等 ESM2、
AbLang-2、MSA encoder、frozen/unfrozen 等组合真的变多，再拆到独立 `encoders.py`。

无论函数放在哪里，`transformers` 都必须 lazy import，不得在 package import 时加载
大模型或读取大文件（§9）。

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

1. "short name 是否受支持"和"`d_model` 是否匹配该 short name"两项校验，必须在 import `transformers`、联网下载权重**之前**完成——不支持的 encoder（目前只支持 ESM2，AbLang-2 等尚未实现）要立刻报 `ValueError`，不要等到网络请求之后才失败。
2. 其它 encoder 家族（AbLang-2 等）目前不支持，遇到时报错，不允许静默退化为
   某个默认 encoder。

### 5.7.1 根目录 `train.py`

`train.py` 是项目对外训练总入口。目标是一条命令完成必要的数据装配、split、训练、
验证、测试评估和输出保存。

推荐命令：

```bash
python train.py --config configs/baseline_antibody_only_ranknet.yaml
```

职责：

1. 调用 `load_config` 读取 YAML。
2. 若配置为自动 split 模式，读取 `all_records.parquet`；如果 `data.filter`
   非空，先调用 `record_filter.py` 生成训练子集和 `filter_summary.csv`。
3. 调用 `splits.py` 生成
   train/valid/test 文件。
4. 读取 train/valid/test 标准表，调用 `dataset.py` 构造 records、pairs 和 metadata。
5. 构造 tokenizer、encoder 和 `AffinityRanker`。
6. 构造 dataloader 和 `Trainer`。
7. 调用 `Trainer.fit()`，再对 valid/test 输出 metrics 和 predictions。
8. 保存 `config.yaml`、`metrics.json`、`group_metrics.csv`、`predictions.csv`、
   `checkpoint.pt` 和 `run.log`。

禁止：

1. 禁止在 `train.py` 里写原始 CSV 解析逻辑。
2. 禁止在 `train.py` 里重新实现 pair 构造、loss、metric 或模型 forward。
3. 禁止为了跑通而自动把 valid/test 设成 train 的同一份文件。
4. 禁止在没有写入 `leakage_report.csv` 的情况下汇报正式 valid/test 指标。

最低验收：

1. `python train.py --config configs/debug_toy.yaml` 能在 toy data 上完成一个 epoch。
2. 自动 split 模式会在 `split_dir` 写出三份 parquet 和两张报告。
3. 显式 split 模式不会重新切分数据。
4. 输出目录中保存完整 config 和随机种子。
5. 配置了 `data.filter` 时，split 前写出 `filtered_records.parquet` 和
   `filter_summary.csv`，且后续 split 只基于过滤后的 records。

### 5.8 `user_entry.py`

这是外部用户调用模型的入口。它只暴露稳定、少量、好理解的 API。比赛场景下，用户只提供抗原、抗体和可选的`model_name`，不需要也不应该手动传入 `AffinityRanker`。

核心函数：

```python
@dataclass
class AffinityPredictor:
    model_name: str
    model: AffinityRanker
    config: Config
    antibody_tokenizer: Tokenizer
    antigen_tokenizer: Tokenizer | None
    checkpoint_path: Path


def load_predictor(
    model_name: str = "best",
    registry_path: Path | None = None,
) -> AffinityPredictor:
    ...


def load_model(checkpoint_path: Path, config_path: Path | None = None) -> AffinityRanker:
    ...


def score_antibodies(
    antigen_sequence: str | None,
    antibodies: Sequence[AntibodyInput],
    model_name: str = "best",
) -> pd.DataFrame:
    ...


def rank_antibodies(
    antigen_sequence: str | None,
    antibodies: Sequence[AntibodyInput],
    model_name: str = "best",
) -> pd.DataFrame:
    ...


def rank_antibody_table(
    input_table: pd.DataFrame,
    model_name: str = "best",
) -> pd.DataFrame:
    ...


def score_antibodies_with_predictor(
    antigen_sequence: str | None,
    antibodies: Sequence[AntibodyInput],
    predictor: AffinityPredictor,
) -> pd.DataFrame:
    ...


def rank_antibodies_with_predictor(
    antigen_sequence: str | None,
    antibodies: Sequence[AntibodyInput],
    predictor: AffinityPredictor,
) -> pd.DataFrame:
    ...


def rank_antibody_table_with_predictor(
    input_table: pd.DataFrame,
    predictor: AffinityPredictor,
) -> pd.DataFrame:
    ...
```

`AntibodyInput`：

```text
antibody_id: str
heavy_chain: str
light_chain: str | None
single_chain_sequence: str | None
antibody_type: "Fv" | "scFv" | "VHH" | "Fab" | "IgG" | "unknown"
```

`Fab`、`IgG` 和 `unknown` 不得仅因为 `antibody_type` 被拒绝。当前模型仍然只消费序列本身，不显式建模 Fc、糖基化、价态或完整抗体构象；因此这三类输入在用户入口
按可用序列处理：

1. 有 `single_chain_sequence` 时优先使用单链序列。
2. 有 `heavy_chain` + `light_chain` 时按 §5.3 抗体序列拼接规则处理。
3. 仅有 `heavy_chain` 时作为单链输入处理。
4. `antibody_type = "unknown"` 时只要序列字段合法，也允许打分；不得静默改写为
   其它类型。

输出：

```text
query_id
antibody_id
score
rank
model_name
```

**外部调用方式**

比赛/普通用户优先使用命令行：

```bash
python predict.py \
  --model best \
  --input input.csv \
  --output rankings.csv
```

Python 调用用于 notebook、服务封装或单次预测：

```python
from affinity_transformer.user_entry import AntibodyInput, rank_antibodies

result = rank_antibodies(
    antigen_sequence="...",
    antibodies=[
        AntibodyInput(
            antibody_id="ab_001",
            heavy_chain="...",
            light_chain="...",
            single_chain_sequence=None,
            antibody_type="IgG",
        ),
    ],
    model_name="best",
)
```

批量输入表 `input.csv` / `input.tsv` 每行代表一个候选抗体。必需字段：

```text
query_id
antibody_id
antigen_sequence
heavy_chain
light_chain
single_chain_sequence
antibody_type
```

字段规则：

1. `query_id` 表示一次排序任务；rank 只在同一个 `query_id` 内计算，不跨
   `query_id` 排序。
2. 同一个 `query_id` 下的 `antigen_sequence` 必须一致；不一致时抛
   `ValueError`。
3. `(query_id, antibody_id)` 必须唯一；重复时抛 `ValueError`。
4. `antigen_sequence` 可以为空，表示走 missing-antigen 分支；但列本身必须存在。
5. `heavy_chain`、`light_chain`、`single_chain_sequence` 至少有一个可用抗体序列。
6. `antibody_type` 必须属于标准枚举：
   `"Fv" | "scFv" | "VHH" | "Fab" | "IgG" | "unknown"`。

批量输出表 `rankings.csv` 字段：

```text
query_id
antibody_id
score
rank
model_name
```

输出规则：

1. 每个 `query_id` 内按 `score` 降序排列。
2. `rank` 从 1 开始，1 表示该 `query_id` 下预测最优的候选抗体。
3. 默认严格模式：任一输入行非法时整体失败，不写部分结果。
4. 输出 `score` 是模型相对排序分数，不解释为 Kd/IC50/EC50 等绝对物理量。

规则：

1. 外部用户不需要知道 pair、group、loss。
2. 外部用户不需要手动传入 `AffinityRanker`、tokenizer、checkpoint path 或 config path。
3. `model_name` 从模型注册表解析为 checkpoint/config/tokenizer 配置；默认值
   `"best"` 指向当前推荐模型。
4. 单个抗原和一组抗体输入时，输出按 `score` 降序排列。
5. 输入序列非法时抛出 `ValueError`，不要静默修正。
6. 没有抗原序列时允许调用，但必须走模型定义的 missing-antigen 分支。
7. `score_antibodies` / `rank_antibodies` 可以每次内部调用 `load_predictor`，用于
   比赛或简单脚本；批量推理应先调用 `load_predictor`，再复用
   `*_with_predictor`，避免每次重复加载大模型。

**模型注册表**

默认模型注册表建议放在 `configs/model_registry.yaml`。它把比赛用户能选择的
`model_name` 映射到具体 checkpoint 和 config。

```yaml
default: best
models:
  best:
    checkpoint_path: outputs/best/checkpoint.pt
    config_path: outputs/best/config.yaml
    description: Current recommended model for ranking.
  antibody_only:
    checkpoint_path: outputs/baseline_antibody_only/checkpoint.pt
    config_path: outputs/baseline_antibody_only/config.yaml
    description: Antibody-only RankNet baseline.
```

规则：

1. `load_predictor` 负责读取注册表、加载 config/checkpoint、构造模型和 tokenizer。
2. `load_model` 只作为内部/开发者 API：输入 checkpoint/config，返回裸
   `AffinityRanker`；它不挂 tokenizer，也不作为比赛用户主入口。
3. `AffinityPredictor` 是推理打包对象，包含 `model + tokenizer + config`。
   tokenizer 不进入 `AffinityRanker.state_dict()`，也不参与训练参数。
4. 未知 `model_name` 必须抛 `ValueError`，并列出可用模型名。
5. `score_antibodies_with_predictor` / `rank_antibodies_with_predictor` 才负责把
   `AntibodyInput` 和抗原序列 tokenize 成 `RankBatch`，再调用
   `predictor.model.forward`。

**根目录 `predict.py`**

`predict.py` 是 `user_entry.py` 的薄命令行包装：

1. 解析 `--model`、`--input`、`--output`、`--registry`、`--format`。
2. 读取 CSV/TSV 为 `pd.DataFrame`。
3. 调用 `rank_antibody_table(input_table, model_name=...)`。
4. 写出 `rankings.csv`。

禁止：

1. 禁止在 `predict.py` 里写模型结构、loss、训练循环或 split 逻辑。
2. 禁止让用户必须提供 checkpoint path；普通用户只选择 `model_name`。
3. 禁止跨 `query_id` 排名。
4. 禁止把 `score` 输出命名为 `kd`、`ic50`、`affinity` 等绝对指标。

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
7. `merge_records.py` 合并后，`all_records.parquet` 行数等于 ready 表行数之和。
8. `all_records_summary.csv` 能定位每个 `dataset_id` 的记录数、可训练记录数和抗原来源分布。

### 7.2 通用模块单测

必须覆盖：

```text
load_records rejects missing required columns
build_pairs never crosses group_id
build_pairs skips equal labels
build_pairs is reproducible with fixed seed
build_pairs large groups do not enumerate O(n^2) candidate pairs
build_pairs large continuous groups return reproducible block-sampled pairs
build_pairs imbalanced binary groups sample positive-negative pairs
build_pairs large block sampler can include intra-block fine-grained pairs
build_pairs quantile_difficulty assigns difficulty buckets by quantile_gap
build_pairs quantile_difficulty estimates raw_gap_min from probe pairs
build_pairs quantile_difficulty filters tiny raw/quantile gaps
build_pairs quantile_difficulty respects difficulty bucket quotas when feasible
build_pairs quantile_difficulty reports quota shortfalls instead of copying pairs
build_pairs sampler QC reports pair distribution and build performance
build_groups never crosses group_id
build_groups skips single-label groups
build_groups is reproducible with fixed seed
record_filter supports dataset/antigen/antibody/group/record include-exclude filters
record_filter supports antigen-antibody pair filters
record_filter supports antibody sequence hash filters
record_filter applies group-level minimum record/label thresholds
build_splits debug_record_split has no record_id leakage
build_splits group_holdout_split has no group_id leakage
build_splits writes split_summary and leakage_report
collate_rank_batch produces valid masks
collate_rank_batch builds antibody sequence per chain-combination rules (single-chain / heavy|light / heavy-only)
collate_rank_batch handles all-antigen-missing batch -> antigen_tokens is None
collate_rank_batch handles mixed batch with some antigen_sequence missing -> per-row mask
collate_pair_batch splits left/right RankBatch and produces y_ij
ranknet_loss rewards correct ordering
compute_group_spearman skips one-label groups
user_entry.rank_antibodies resolves model_name and returns descending ranks
user_entry.rank_antibodies_with_predictor reuses loaded predictor
user_entry.rank_antibody_table ranks within query_id only
user_entry.rank_antibody_table rejects inconsistent antigen_sequence within query_id
predict.py reads input.csv and writes rankings.csv with required columns
model.forward handles missing antigen without NaN
model.forward + loss supports backward without NaN gradients
train.py debug config completes one epoch on toy data
train.py automatic split applies data.filter before writing split files
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
load config -> optionally build splits -> load train/valid records
-> build pairs -> create dataloader -> forward
-> ranknet loss -> one optimizer step -> evaluate
-> save checkpoint/metrics/predictions
```

## 8. 实验输出规范

每次训练输出到 `outputs/{run_id}/`。

必需文件：

```text
config.yaml
metrics.json
group_metrics.csv
predictions.csv
test_predictions.csv          # 有 test set 时必需
checkpoint.pt
run.log
split_summary.csv             # 自动 split 模式必需
leakage_report.csv            # 自动 split 模式必需
```

`predictions.csv` 字段：

```text
record_id
dataset_id
group_id
rank_label
score
label_kind
split
```

`metrics.json` 至少包含：

```text
train_loss
valid_macro_spearman
valid_weighted_spearman
n_valid_groups
n_skipped_groups
test_macro_spearman       # 有 test set 时必需
test_weighted_spearman    # 有 test set 时必需
```

规则：

1. valid 指标可用于 early stopping 和 checkpoint 选择。
2. test 指标只在训练结束后计算一次，不参与早停、不参与调参。
3. 使用 `debug_record_split` 的输出目录必须在 `metrics.json` 中标记
   `split_strategy = "debug_record_split"`，不得作为正式实验结果汇报。
4. 使用 `group_holdout_split` 汇报正式结果时，必须同时保存 `leakage_report.csv`。

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
11. 禁止把同一份文件同时作为 train/valid/test 来汇报模型性能。
12. 禁止用 test set 做 early stopping、checkpoint 选择或人工调参依据。

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

下一阶段只做这些：

1. 写 `requirements.txt` / `requirements-dev.txt`，保证远程服务器能复现最小环境。
2. 写 `scripts/prepare/binding/merge_records.py`，生成 `all_records.parquet` 和
   `all_records_summary.csv`。
3. 写 `affinity_transformer/splits.py`，先实现 `debug_record_split` 和
   `group_holdout_split`。
4. 写根目录 `train.py`，用一个 config 完成自动 split、训练、验证、测试评估和输出保存。
5. 写 `configs/debug_toy.yaml`、`configs/baseline_antibody_only_ranknet.yaml`、
   `configs/baseline_group_holdout_ranknet.yaml` 和 `configs/model_registry.yaml`。
6. 写根目录 `predict.py` 和 `user_entry.py` 的 `model_name` 推理入口，保证外部用户
   只给抗原、抗体和模型名即可得到排序结果。
7. 写 `tests/test_splits.py`、`tests/test_train_entry.py` 和 `tests/test_predict_entry.py`，
   专门防止 split 泄露、训练入口假跑通和预测入口跨 `query_id` 排名。
8. 把高置信抗原分批补入对应 `convert.py`；不确定抗原只进入候选表，不伪造序列。

本阶段不做这些：

1. 不新增通用 raw CSV parser。
2. 不为每个子数据集训练单独 task head。
3. 不把 AbLang-2、MSA fusion 或复杂 cross-attention 设成默认路径。
4. 不为了追分把 test set 用于调参。

先把 antibody-only RankNet 在 `group_holdout_split` 上可复现地跑通，再讨论 Exact/MSA、
AbLang-2、Cross-Attention 和 Attention Pooling 的复杂实现。
