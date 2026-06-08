# AffinityTransformer v3

`AffinityTransformer` 是与 `AffinityMLP` 隔离的 v3 实验包。它可以复用 v1/v2 的数据思想和 ranking loss 思路，但不 import、不修改 v1/v2 代码。

## Core Policy

v3 的输入是：

```text
score = f(antibody, antigen_context)
```

抗原上下文按下面规则构建：

1. 如果官方提供了抗原序列，计算 `antigen_single_embedding`，并与 `antigen_msa_embedding` 拼接/融合。
2. 如果官方没有提供抗原序列，不伪造 single embedding，直接使用 `antigen_msa_embedding`。
3. 为了保持 batch 输入维度固定，没有官方序列时 single slot 用零向量占位，但 flags 会标记 `uses_msa_only_policy=1`，模型不能把它当成真实 embedding。
4. 抗体 encoder 可选 ESM2、IgBert、IgT5 或自定义 Hugging Face encoder。

## Main Modules

```text
config.py                    全局配置和 encoder presets
antigen_schema.py            registry 共享 schema；被多个子包复用，留在顶层
pipeline.py                  registry -> cache -> feature matrix 的高层调度入口

registry/                    antigen_registry 构建
  core.py                    FLAb CSV 直接构建 registry、读写、质检、合并
  sources.py                 TASKS/proteinbase/ANDD/SAbDab/FLAb CSV 解析
  workflow.py                多来源合并、质检、写出
  build.py                   registry 命令行入口

encoders/                    预训练序列模型包装
  sequence.py                Hugging Face ESM2/IgBert/IgT5 通用 encoder

embeddings/                  embedding 计算与 cache
  antibody.py                抗体 heavy/light 或 paired embedding cache
  antigen.py                 antigen single/MSA/ligand cache 入口
  msa.py                     ESM-MSA-1b antigen embedding

msa/                         同源序列与 MSA 构建
  homolog_search.py          FASTA 读写和 BLAST/MMseqs2 命令构造
  builder.py                 A3M 读写、MSA 采样、query 质检

data/                        训练特征矩阵
  context.py                 antibody + antigen_context feature matrix

models/                      torch 模型
  context.py                 AntigenContextMLP + AffinityTransformer
```

## Registry Stage

registry stage 先完成抗原识别，不做 MSA 搜索、不训练模型。它把下面来源整理成
`antigen_registry.csv`：

```text
FLAb/data/binding/*.csv
FLAb/data/flab_metadata.csv
references/task_docs/TASKS.md
competition_data/_Preliminary_SequenceData/22/proteinbase_all_data_28_01_2026.csv
competition_data/_Preliminary_NanobodyData/ANDD.xlsx
competition_data/_Preliminary_StructureData/sabdab_summary_all.tsv
```

运行：

```bash
python -m FLAb.AffinityTransformer.registry.build
```

输出：

```text
FLAb/results/v3/registry/antigen_registry.csv
FLAb/results/v3/registry/antigen_registry_issues.csv
FLAb/results/v3/auxiliary/task_controls.csv
FLAb/results/v3/auxiliary/proteinbase_targets.csv
```

调试时可以只读少量 binding 文件：

```bash
python -m FLAb.AffinityTransformer.registry.build --max_binding_files 5
```

注意：`ANDD.xlsx` 需要 `openpyxl`。如果集群环境缺少它，请重新安装
`FLAb/requirements.txt`。

## Encoder Strategy

默认配置：

```text
antibody encoder       = esm2_650m
antibody layout        = separate_chains
antigen single encoder = esm2_650m
antigen MSA encoder    = ESM-MSA-1b
```

抗体侧可以改为：

```text
igbert + separate_chains
igt5   + paired_chains
custom Hugging Face model name
```

## Cache Rule

训练阶段只读 cache，不做在线搜索和在线数据库访问。

推荐 cache：

```text
cache/v3/antibody_embeddings/
cache/v3/antigen_embeddings/single_esm2/
cache/v3/antigen_embeddings/msa_esm1b/
cache/v3/msa/
```

这些目录都是本地生成产物，不应提交到 GitHub。
