# AffinityClip v4

AffinityClip(v4) 是一个独立于 v1/v2/v3 的新模型文件夹。它借鉴 DrugCLIP 的检索式建模思想：把目标侧编码成 query，把候选侧编码成 key，通过相似度矩阵完成排序。

## 文献依据

- DrugCLIP 的 NeurIPS 2023 版本把 virtual screening 重新表述为 dense retrieval：protein pocket 是 query，molecule library 是被检索对象，并用两个独立 encoder 对齐表示。论文明确写到 protein 和 molecule 分别经过 encoder，再用 dot product / cosine similarity 计算所有 pair 的相似度。
- DrugCLIP 的正式 Science 版本题为 *Deep contrastive learning enables genome-wide virtual screening*，DOI 为 `10.1126/science.ads9530`。我查到的正式发表信息是 Science 391(6781), 2026-01-08；很多新闻稿会按 2025/2026 在线节点混称。
- DrugCLIP 的原始 CLIP loss 假设一个 batch 中第 `i` 个 protein-pocket 只和第 `i` 个 molecule 是正样本，其它 in-batch pair 是负样本。FLAb 的亲和力排序不是这个数据形态：一个 `compatible_group` 内有多个抗体，且标签是连续亲和力排序。因此 v4 不能照搬 one-hot diagonal CLIP，必须改成 group-aware soft target + RankNet。

## v4 架构

```text
antigen_single / antigen_msa / antigen_flags
        -> Antigen Transformer Tower
        -> normalized query vector q

heavy_chain / light_chain
        -> Antibody Transformer Tower
        -> normalized key vector k

logit(antigen_i, antibody_j) = exp(logit_scale) * cosine(q_i, k_j)
```

默认 token：

- antibody tower: `heavy_embedding`, `light_embedding`
- antigen tower: `single_esm2`, `msa_esm1b`, `antigen_flags`

这里的 “double transformer” 不是把所有 token 丢进同一个 Transformer，而是两个塔各自做 self-attention：

- 抗原塔负责融合 single sequence embedding、MSA embedding、抗原类型/缺失标记；
- 抗体塔负责融合 heavy/light；
- 两塔输出在同一个 retrieval embedding space 中比较相似度。

## Loss

v4 默认使用两个损失相加：

1. `group_soft_clip_loss`
   - 输入整张 `[batch, batch]` 相似度矩阵；
   - 对每个 antigen query，只把同一个 `compatible_group` 的抗体当作候选正集合；
   - 同组抗体的 target 概率由统一方向的 affinity label 做 softmax 得到；
   - 其它 group 的抗体仍然通过 log-softmax 分母作为负候选。

2. `ranknet_loss_from_scores`
   - 只取相似度矩阵对角线，表示 batch 中真实配对的 antigen-antibody score；
   - 只在同一个 `compatible_group` 内构造 pair；
   - 若 `label_i > label_j`，优化 `softplus(score_j - score_i)`。

这个组合保留了 v3/v2 里比较可靠的 RankNet 排序逻辑，同时加入 DrugCLIP 式 query-key 检索压力。

## 与 v3 的区别

v3:

```text
[CLS], antibody, antigen_single, antigen_msa, flags
        -> one Transformer
        -> scalar score
```

v4:

```text
antigen tokens -> antigen tower -> query
antibody tokens -> antibody tower -> key
query x key -> similarity matrix -> soft CLIP + RankNet
```

直觉上，v3 是“给一个抗原-抗体 pair 打分”，v4 是“给一个抗原 query，从一批抗体 key 里检索排序”。比赛指标是 Spearman 排序，因此 v4 的输出形式更贴近最终使用方式。

## 文件说明

- `config.py`: v4 独立配置和默认维度。
- `features.py`: 把 v3 风格 concat feature 切成 v4 需要的 heavy/light/single/MSA/flags token。
- `model.py`: `TransformerTower` 和 `AffinityCLIP`。
- `losses.py`: group-aware soft CLIP loss、RankNet loss、组合 loss。

## 建议实验

1. `v3 attention + RankNet`
2. `v4 AffinityCLIP + RankNet only`
3. `v4 AffinityCLIP + group-soft-CLIP + RankNet`
4. `v4 AffinityCLIP` 使用不同 antibody encoder：ESM2 / IgBert / IgT5
5. 消融 antigen context：single only / MSA only / single + MSA

最重要的质检点：同一个 batch 里最好包含多个 `compatible_group`，且每个 group 至少有两个样本；否则 RankNet 没有 pair，CLIP 也会退化。
