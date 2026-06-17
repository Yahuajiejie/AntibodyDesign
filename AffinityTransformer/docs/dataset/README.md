# Dataset 代码阅读说明

`affinity_transformer/dataset/` 是当前项目的标准表读取、训练记录过滤、pair/listwise 构造和 torch Dataset 包装层。

本目录由原来的单文件 `affinity_transformer/dataset.py` 拆分而来，目标是让训练代码保持可调用，同时让组员能按职责阅读和质检代码。当前约定：

1. `from affinity_transformer.dataset import ...` 是稳定公共入口。
2. 子文件只承担一个职责，便于按 spec v0.6 做代码质检。
3. 训练脚本、数据脚本和用户入口不应直接依赖 `pair_sampling/` 内部函数。
4. v0.7 新采样器可以新增为并列模块，不要偷偷改掉 v0.6 行为。

## 模块映射

| 原始位置 | 当前拆分位置 | 职责 |
| --- | --- | --- |
| `affinity_transformer/dataset.py` | `affinity_transformer/dataset/schema.py` | 标准表字段、pair/group 输出字段、采样默认参数 |
| `affinity_transformer/dataset.py` | `affinity_transformer/dataset/examples.py` | `AffinityExample`、pair/listwise example dataclass |
| `affinity_transformer/dataset.py` | `affinity_transformer/dataset/records.py` | 标准表读取、`keep_for_training` 和 `rank_label` 过滤 |
| `affinity_transformer/dataset.py` | `affinity_transformer/dataset/pairs.py` | pairwise ranking pair 生成入口，小 group 枚举 |
| `affinity_transformer/dataset.py` | `affinity_transformer/dataset/pair_sampling/` | v0.6 大 group 采样器、双标签采样器、分块采样器、参数校验 |
| `affinity_transformer/dataset.py` | `affinity_transformer/dataset/groups.py` | listwise group 生成 |
| `affinity_transformer/dataset.py` | `affinity_transformer/dataset/datasets.py` | torch `Dataset` 包装器 |

## 组长质检口径

读代码时优先检查三件事：

1. 数据是否只从标准表进入，不允许在 dataset 层解析原始 CSV。
2. pair/group 是否只在同一个 `group_id` 内构造，不允许跨 group。
3. 二分类或双标签 group 是否走专门逻辑，不应被强行切成 5 个连续分位块。

## 下一批可拆文件

`trainer.py`、`record_filter.py`、`config.py` 也已经偏长，但它们和训练命令、配置 YAML、实验输出关系更密。建议后续按同样规则拆：

1. 先把公共入口和内部 helper 分清楚。
2. 再按职责拆小文件。
3. 最后补调用链文档。
4. 不在同一轮里顺手改训练算法。
