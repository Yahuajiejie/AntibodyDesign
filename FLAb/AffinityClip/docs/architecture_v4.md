# AffinityClip(v4) 技术方案

## 一、为什么要从 v3 走到 v4

v3 的方向是 attention + RankNet。它把抗体、抗原 single embedding、抗原 MSA embedding 和 flags 放进同一个 Transformer，最后输出一个 scalar score。这个方案合理，但它仍然是 pointwise/pairwise scorer：模型每次主要看一个抗原-抗体配对。

DrugCLIP 的启发在于把任务改写成检索问题：

```text
给定 target query，在候选库里检索最可能结合的 molecule。
```

迁移到本项目就是：

```text
给定 antigen query，在候选抗体库里检索亲和力更强的 antibody。
```

这更贴近比赛的 Spearman 排序目标。

## 二、不能照搬 DrugCLIP 的地方

DrugCLIP 的 batch 结构通常可以看作：

```text
(protein_i, molecule_i) 是正样本
(protein_i, molecule_j), i != j 是 in-batch 负样本
```

但 FLAb/比赛数据不是一一配对二分类。一个 `compatible_group` 里有多个抗体，它们都对应同一个或同类可比较抗原/实验体系，只是亲和力强弱不同。因此：

- 不能把 diagonal 以外的同组抗体都当负样本；
- 不能只用 one-hot diagonal target；
- 需要保留组内 ranking pair。

v4 的改法是 group-aware soft CLIP + RankNet。

## 三、模型输入

### 抗体侧

参数：

- `heavy_embedding`: `[batch, 1280]`
- `light_embedding`: `[batch, 1280]`

返回：

- `antibody_key`: `[batch, projection_dim]`

实现：

- heavy/light 分别投影成 token；
- 加一个 learned CLS token；
- 经过 antibody Transformer tower；
- 取 CLS 输出，线性投影并 L2 normalize。

### 抗原侧

参数：

- `antigen_single_embedding`: `[batch, 1280]`
- `antigen_msa_embedding`: `[batch, 768]`
- `antigen_flags`: `[batch, 20]`
- `antigen_available`: 可选，三个 `[batch]` bool mask

返回：

- `antigen_query`: `[batch, projection_dim]`

实现：

- single/MSA/flags 分别投影成 token；
- 缺失 single 或 MSA 时用 `available=False` mask 掉，而不是让全零向量直接参与注意力；
- 经过 antigen Transformer tower；
- 取 CLS 输出，线性投影并 L2 normalize。

## 四、模型输出

函数：

- `AffinityCLIP.similarity_matrix(...)`

参数：

- `antibody_features`: list of antibody tensors，默认 `[heavy, light]`
- `antigen_features`: list of antigen tensors，默认 `[single, msa, flags]`
- `antibody_available`: 可选 mask
- `antigen_available`: 可选 mask

返回：

- `logits`: `[num_antigens, num_antibodies]`

实现：

```text
q = antigen_tower(antigen_features)
k = antibody_tower(antibody_features)
logits = exp(logit_scale) * q @ k.T
```

因为 q/k 都做了 L2 normalize，所以 `q @ k.T` 是 cosine similarity。

## 五、Loss

### `group_soft_clip_loss`

参数：

- `logits`: `[batch, batch]`
- `labels`: `[batch]`，方向统一，越大亲和力越强
- `group_ids`: 每行的 `compatible_group`
- `label_temperature`: 把连续 label 转成 soft target 的温度
- `average_by_group`: 是否让每个 group 等权

返回：

- scalar loss

实现：

对第 `i` 个 antigen query，只允许同组抗体作为 target support：

```text
target_j = softmax(label_j / tau), if group_j == group_i
target_j = 0, otherwise
loss_i = -sum_j target_j * log_softmax(logits_i)_j
```

这相当于告诉模型：同组里亲和力越强的抗体，应该在这个 antigen query 下排得越靠前。

### `ranknet_loss_from_scores`

参数：

- `scores`: `[batch]`，通常取 `logits.diagonal()`
- `labels`: `[batch]`
- `group_ids`: compatible_group
- `min_label_diff`: 忽略差距太小的 pair

返回：

- scalar loss

实现：

只在同一个 group 内构造 pair。如果 `label_i > label_j`：

```text
loss_ij = softplus(score_j - score_i)
```

### `affinity_clip_loss`

返回：

- `{"loss": total, "clip": clip, "ranknet": ranknet}`

实现：

```text
total = clip_weight * group_soft_clip_loss
      + ranknet_weight * ranknet_loss_from_scores
```

## 六、预期优点与风险

优点：

- 输出相似度矩阵，天然适合排序；
- 抗原 query 和抗体 key 可预计算，便于之后做标准数据集排序；
- group-aware soft target 避免把同组高亲和力抗体误当作负样本；
- RankNet 保留了现有 v2/v3 中最可靠的相对排序逻辑。

风险：

- batch 采样很关键：每个 batch 应该包含多个 group，每个 group 至少两个样本；
- 如果某些 group 的 antigen context 缺失严重，antigen tower 可能只学到 flags；
- group-soft-CLIP 的 label temperature 需要调参，过小会接近只看组内第一名，过大会变成平均正样本。

## 七、下一步实现顺序

1. 先用 v3 生成的 heavy/light/single/MSA/flags cache 跑通 v4。
2. 先比较 `RankNet only` 和 `group-soft-CLIP + RankNet`。
3. 再做 antigen context 消融：single only / MSA only / single + MSA。
4. 最后替换 antibody encoder：ESM2 / IgBert / IgT5。
