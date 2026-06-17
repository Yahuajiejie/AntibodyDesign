# model/ 代码阅读说明

`model/` 是对当前 `affinity_transformer/` 核心代码的可读性拆分镜像。

当前训练脚本仍然调用原来的 `affinity_transformer/` 包；本目录的第一目标是让组员能读懂、评审和后续迁移代码。也就是说：

1. 原始生产代码不在本次拆分中修改。
2. `model/dataset/` 目前对应原始 `affinity_transformer/dataset.py`。
3. 每个子文件只承担一个职责，便于按 spec v0.6 做代码质检。
4. 如果后续确认拆分后的结构稳定，再把训练入口逐步切到 `model/`。

## 模块映射

| 原始文件 | 新拆分位置 | 职责 |
| --- | --- | --- |
| `affinity_transformer/dataset.py` | `model/dataset/schema.py` | 标准表字段、pair/group 输出字段、采样默认参数 |
| `affinity_transformer/dataset.py` | `model/dataset/examples.py` | `AffinityExample`、pair/listwise example dataclass |
| `affinity_transformer/dataset.py` | `model/dataset/records.py` | 标准表读取、`keep_for_training` 和 `rank_label` 过滤 |
| `affinity_transformer/dataset.py` | `model/dataset/pairs.py` | pairwise ranking pair 生成入口，小 group 枚举 |
| `affinity_transformer/dataset.py` | `model/dataset/pair_sampling/` | v0.6 大 group 采样器、双标签采样器、分块采样器、参数校验 |
| `affinity_transformer/dataset.py` | `model/dataset/groups.py` | listwise group 生成 |
| `affinity_transformer/dataset.py` | `model/dataset/datasets.py` | torch `Dataset` 包装器 |

## 组长质检口径

读代码时优先检查三件事：

1. 数据是否只从标准表进入，不允许在 dataset 层解析原始 CSV。
2. pair/group 是否只在同一个 `group_id` 内构造，不允许跨 group。
3. 二分类或双标签 group 是否走专门逻辑，不应被强行切成 5 个连续分位块。

## 下一批可拆文件

`trainer.py`、`record_filter.py`、`config.py` 也已经偏长，但它们和训练命令、配置 YAML、实验输出关系更密。建议在确认 `model/dataset/` 的拆分风格后，再按同样规则拆：

1. 先复制到 `model/`。
2. 再按职责拆小文件。
3. 最后补调用链文档。
4. 不在同一轮里顺手改算法。
