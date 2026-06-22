# 训练流程

本文说明当前项目从标准表到训练、评估、调用的主流程。核心原则：原始 CSV 的奇怪字段只在各数据集自己的 `convert.py` 或 shell 脚本里处理；进入通用训练流程后，只允许消费标准表。

## 0. 入口总览

通用训练闭环只有一条主线：

```text
processed/binding/**/records.parquet
  -> scripts/prepare/binding/merge_records.py
  -> processed/binding/all_records.parquet
  -> scripts/data/filter_records.py 或 train.py 内置 filter
  -> affinity_transformer.splits
  -> train.py
  -> outputs/<run_name>/
  -> predict.py 或 affinity_transformer.user_entry
```

不要把 raw parser、split、pair sampler、trainer 混在一个文件里。

## 1. 准备标准表

每个数据集、每个子目录可以保留自己的转换脚本。只要最终产物满足标准 schema 即可：

```text
scripts/prepare/binding/<study>/<table>/convert.py
  -> processed/binding/<study>/<table>/records.parquet
```

质检：

```bash
python scripts/prepare/validate_processed_table.py \
  --input processed/binding/<study>/<table>/records.parquet
```

禁止项：

1. 通用模块不解析 raw CSV。
2. `affinity_transformer/dataset/` 不负责补抗原、不负责转换 label 方向。
3. `keep_for_training` 和 `rank_label` 必须在进入训练前确定。

## 2. 合并 binding records

合并所有 ready 的标准表：

```bash
python scripts/prepare/binding/merge_records.py \
  --input-root processed/binding \
  --output processed/binding/all_records.parquet \
  --summary processed/binding/all_records_summary.csv
```

产物：

```text
processed/binding/all_records.parquet
processed/binding/all_records_summary.csv
```

## 3. 做 G00 数据质检和 split

本地直接跑：

```bash
bash scripts/runs/g00_qc_and_splits.sh
```

Slurm 上跑：

```bash
sbatch scripts/slurm/g00_qc_and_splits.sbatch
```

G00 需要确认：

1. `reports/` 里有全量 records、trainable records、group 数、抗原来源分布。
2. `leakage_report.csv` 对正式 `group_holdout_split` 必须 PASS。
3. 正式实验使用 `group_holdout_split`，debug 结果不能汇报为主结果。

## 4. 单个 config 训练

最小训练命令：

```bash
python train.py \
  --config configs/experiments/g01_maxctx_cross_attention.yaml \
  --output-dir outputs/g01_maxctx_cross_attention
```

`train.py` 做的事情：

```text
load_config
  -> resolve data paths
  -> 如配置要求，filter all_records
  -> 如配置要求，build_splits
  -> build_model_and_tokenizers
  -> filter_trainable_records
  -> build_pairs
  -> PairwiseAffinityDataset
  -> collate_pair_batch
  -> Trainer.fit
  -> valid/test predictions
  -> group Spearman metrics
```

输出目录至少应包含：

```text
checkpoint.pt
config.yaml
metrics.json
predictions.csv
group_metrics.csv
run.log
```

如果配置包含 test split，还会输出：

```text
test_predictions.csv
test_group_metrics.csv
```

## 5. 批量实验

当前实验按组号组织：

```text
g01: 主模型消融，antibody_only / concat_antigen / cross_attention
g02: label source 消融，experimental / no_predicted / all_label_kinds
g03: pair sampling 强度消融
g04: antigen subset 消融
```

本地顺序跑某组：

```bash
GROUP_SCRIPT=scripts/runs/g01_core_ablation.sh \
bash scripts/runs/g01_core_ablation.sh
```

更推荐直接运行组脚本：

```bash
bash scripts/runs/g01_core_ablation.sh
bash scripts/runs/g02_label_source_ablation.sh
bash scripts/runs/g03_pair_sampling_ablation.sh
bash scripts/runs/g04_antigen_subset_ablation.sh
```

收集结果：

```bash
python scripts/experiments/collect_results.py \
  --runs-root outputs \
  --output reports/experiment_results.csv
```

## 6. Slurm 训练

先确认环境和 ESM2 缓存：

```bash
sbatch scripts/slurm/setup_affitest_env.sbatch
sbatch scripts/slurm/warmup_esm2_cache.sbatch
```

跑单个 config：

```bash
sbatch \
  --job-name=aff-g01-cross \
  --export=ALL,CONFIG=configs/experiments/g01_maxctx_cross_attention.yaml \
  scripts/slurm/run_config.sbatch
```

跑一组实验：

```bash
sbatch \
  --job-name=aff-g01-core \
  --export=ALL,GROUP_SCRIPT=scripts/runs/g01_core_ablation.sh \
  scripts/slurm/run_group.sbatch
```

依赖链提交：

```bash
bash scripts/slurm/submit_g00_g01_chain.sh
```

规则：

1. login 节点只提交任务，不跑训练。
2. `run_group.sbatch` 是一张卡顺序跑一组，省卡但慢。
3. `run_config.sbatch` 是一个 config 一个 job，适合把多个对照实验并行提交。
4. 训练 OOM 时先降 batch size、pair 数、序列长度或 encoder 尺寸，不要直接把所有实验堆到同一张卡。

## 7. 训练后调用

对外调用应该走用户入口，而不是让外部用户自己拼 tokenizer / Dataset：

```text
用户输入:
  antigen sequence
  antibody candidates
  model short name 或 checkpoint path

项目内部:
  load_model / registry
  tokenize antigen + antibody
  build RankBatch
  model forward
  score_antibodies / rank_antibodies

输出:
  antibody_id
  score
  rank
```

训练产物进入调用前，需要至少确认：

1. `config.yaml` 和 `checkpoint.pt` 配套。
2. encoder short name 和 `d_model` 匹配。
3. antibody type 覆盖训练集里的 `Fv`、`scFv`、`VHH`、`Fab`、`IgG`、`unknown`。
4. 抗原缺失时的行为由 config 明确控制，不能静默把缺失抗原当有效 token。

## 8. 开发者验收清单

提交训练相关代码前，至少跑：

```bash
python -m py_compile train.py predict.py affinity_transformer/*.py affinity_transformer/dataset/*.py affinity_transformer/dataset/pair_sampling/*.py
pytest tests/test_dataset.py tests/test_dataloader.py tests/test_trainer.py
```

如果本地环境没有完整依赖，至少保证：

```bash
python -m py_compile train.py affinity_transformer/dataset/*.py affinity_transformer/dataset/pair_sampling/*.py
```

并在提交说明里写清楚未跑完整测试的原因。
