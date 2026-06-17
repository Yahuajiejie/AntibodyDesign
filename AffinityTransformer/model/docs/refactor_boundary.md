# 拆分边界说明

这次拆分遵守一个原则：只改变阅读结构，不改变训练事实。

## 本次做了什么

1. 新建 `model/`。
2. 把 `affinity_transformer/dataset.py` 按职责复制拆分到 `model/dataset/`。
3. 把 pair sampler 的复杂内部逻辑继续拆到 `model/dataset/pair_sampling/`。
4. 新增 `model/docs/`，解释调用链和质检重点。
5. 保留 v0.6 采样器行为，不把 v0.7 的实验性采样器混进来。

## 本次没有做什么

1. 没有修改 `affinity_transformer/`。
2. 没有修改 `docs/programming_spec_v0.6.md`。
3. 没有修改训练脚本、SLURM 脚本、配置 YAML。
4. 没有把训练入口切换到 `model/`。

## 为什么先拆 dataset

`dataset.py` 当前超过 1200 行，里面同时放了：

1. schema 常量。
2. dataclass。
3. 标准表读取。
4. trainable 过滤。
5. pair sampler。
6. listwise group sampler。
7. torch Dataset 包装器。

这些逻辑互相有关，但不应该堆在一个文件里。按现在的拆法，组员可以先读 `records.py` 和 `schema.py`，再读 `pairs.py`，最后读 `datasets.py`，理解成本会低很多。

## 后续迁移建议

如果后面要让训练实际调用 `model/`，建议按这个顺序：

1. 给 `model/dataset/` 补行为一致性测试，和 `affinity_transformer.dataset` 对同一输入输出做比较。
2. 确认 pair sampler 的边界行为，包括 binary、two-label、continuous large group。
3. 把 import 从 `affinity_transformer.dataset` 切到 `model.dataset`。
4. 再拆 `trainer.py` 和 `config.py`。

不要在切换训练入口的同一个提交里修改采样算法。否则一旦结果变了，很难判断是拆分造成的，还是算法变化造成的。
