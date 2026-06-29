# AffinityMLPSimplified v1

这个目录保存 v1 的简化 MLP 基线，代码从 git commit `15c8c70` (`define our model`) 中的 `FLAb/affinity_model/` 恢复而来。

它和当前 v2/v3 隔离：

```text
AffinityMLPSimplified  v1: per-benchmark MLP baseline
AffinityMLP            v2: shared-head global MLP baseline
AffinityTransformer    v3: antigen-aware / MSA-aware route
```

## v1 Design

v1 的核心逻辑是：

- 用 ESM2-650M 提取 antibody sequence mean embedding；
- 每个 benchmark 单独划分 train/val/test；
- 每个 benchmark 单独训练一个 `AffinityMLP`；
- 支持 `mse`、`hinge`、`ranknet` 三种 loss；
- 用测试集 Spearman 评估排序效果。

这个版本适合作为历史基线和汇报对照，但不再作为主模型路线。主要原因是 per-benchmark head 会让模型利用数据集特异性，不能代表真正的通用亲和力排序能力。

## Restored Files

```text
__init__.py
config.py
data_loader.py
dataset.py
embeddings.py
losses.py
model.py
trainer.py
run_v1.py
```

## Run

从 `FLAb/` 目录运行：

```bash
python AffinityMLPSimplified/run_v1.py --mode all
```

或者分步：

```bash
python AffinityMLPSimplified/run_v1.py --mode embed
python AffinityMLPSimplified/run_v1.py --mode train
```

输出默认写入 `results/affinity_model`。如果要和 v2/v3 同时保存结果，建议通过 `--output_dir` 指到独立目录。

