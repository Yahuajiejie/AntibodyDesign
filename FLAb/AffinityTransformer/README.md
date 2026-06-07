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
config.py                    v3 独立配置和 encoder presets
sequence_encoders.py         通用 Hugging Face 序列 encoder
antibody_embeddings.py       抗体 heavy/light 或 paired embedding cache
antigen_schema.py            antigen_registry schema 和字段工具
antigen_registry.py          registry 构建、读取、写出、质检
homolog_search.py            FASTA 读写和 BLAST/MMseqs2 命令构造
msa_builder.py               A3M 读写、MSA 采样、query 质检
msa_embeddings.py            ESM-MSA-1b antigen embedding
antigen_embeddings.py        antigen single/MSA/ligand cache 入口
antigen_context_dataset.py   v3 feature matrix 构造
antigen_context_model.py     MLP baseline + AffinityTransformer
pipeline.py                  registry -> cache -> feature matrix 调度层
```

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

