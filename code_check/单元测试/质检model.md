# model/

## projection.py

```python
class TokenProjection(nn.Module):
    """Project one encoder's token embeddings into a shared hidden width."""
```
token projection forward 流程：
normalization(Layernorm)→ linear layer

## pooling.py
```python
class MaskedMeanPooling(nn.Module):
    """Module wrapper around :func:`masked_mean_pool`."""
```
**MaskedMeanPooling流程**：
1. 将 0/1的mask 转换为跟hidden相同dtype的mask_f，并在最后加入一个维度（B,L,1），方便与 （B,L,H）的hidden向量广播乘
2. 将mask_f与hidden进行广播乘，并沿着dim=1求和（即将不同氨基酸的向量取平均）
3. 计算有效token数
4. 返回summed/有效token数
```python
class AttentionPooling(nn.Module):
    """Learned-query pooling with explicit all-missing-row handling."""
```
**AttentionPooling流程：**
1. 先将hidden向量进行归一化
2. 然后**对每个 batch b、每个位置 l，把 normalized[b, l, :] 这个长度为 D 的向量，和 query 这个长度为 D 的向量，做逐元素相乘后求和**——这符合点积（dot product）的定义。这样的运算会得到尺寸为[b,l]的矩阵logits
3. 再检查是否存在某一条序列，它没有有效的token
- mask.any(dim=1)：沿着 dim=1（也就是序列长度 L 这一维）做"逻辑或"。对每个样本 b，只要这一行里**有任意一个** True，就代表存在有效token，结果就是 True；如果整行全是 False，结果就是 False。
- safe_mask = mask.clone() 然后复制一份mask
- (~has_tokens).any(): 只要存在一条不包含任何有效token的序列
- safe_mask[~has_tokens.squeeze(1), 0] = True safe_mask 就会把没有有效token那一条序列的第0个位置设为True
4. 然后我们沿着序列长度L这个维度，进行softmax运算，在运算前，我们需要将无效token处的数值设为 -NaN。所得结果矩阵是一个关联矩阵，且无效token处的取值为0
5. 之后，我们将weights * mask.to(weights.dtype)，这样做的目的是因为：
- 对于全无效token构成的序列，weights[0]=1，而这样的向量拿去做attention会得到错误的结果。此时weights *mask.to(weights.dtype)运算就会将weights变为全0向量
- 对于含有有效token的序列，weights * mask.to(weights.dtype)运算不会改变有效token的分数
6. 最后做attention运算
## heads.py
```python
class ScalarScoringHead(nn.Module):
    """Unbounded v0.65 scalar scorer shared by every ranking objective."""
```
representation [B, D]          ← 输入：每个抗体-抗原对的融合表征
     │
LayerNorm(D)                   ← 稳定训练，归一化特征分布
     │
Linear(D → hidden_dim)         ← 升/降维，提取非线性特征
     │
GELU()                         ← 激活函数
     │
Dropout(p)                     ← 正则化，防过拟合
     │
Linear(hidden_dim → 1)         ← 压缩到标量
     │
squeeze(-1)                    ← [B, 1] → [B]，去掉多余维度
     ▼
score [B]                      ← 每个样本一个无界实数分数

## attention.py

```python
def select_num_heads(d_model: int) -> int:
    """Return the largest supported head count that divides ``d_model``."""
```
这个模块功能很简单，multihead_attention要求d_model能被count head整除，我们就挑一个能整除d_model的最大头数
```python
def build_cross_attention(d_model: int) -> nn.MultiheadAttention:
    """Construct the legacy single cross-attention layer."""
```
根据上面的函数select_num_heads确定头数，建立一个attention层

## blocks.py

最主要的模块就是
```python
class InteractionBlock(nn.Module):
    """Pre-norm cross-attention and FFN updates for two token streams."""
```
该模块同时维护两条序列：
- **antibody_tokens** [B, L_ab, D]
- **antigen_tokens** [B, L_ag, D]
各自带有一个bool mask标记哪些位置是有效token。
模块整体来看是标准的 **Pre-LN Transformer** **风格双塔交互层**:每条流先做 cross-attention 去"看"另一条流,再做 FFN 做自身特征变换,残差连接贯穿全程。
```
antibody_cross_norm / antigen_cross_norm   → LayerNorm,在 attention 前归一化

antibody_to_antigen                         → MultiheadAttention(antibody 查询 antigen)

antigen_to_antibody (可选)                  → MultiheadAttention(antigen 查询 antibody)

antibody_ffn_norm / antigen_ffn_norm       → LayerNorm,在 FFN 前归一化

antibody_ffn / antigen_ffn                  → Linear→GELU→Dropout→Linear
```

**forward 的数据流(单向 antibody_to_antigen 为例)**

1. **校验 + 清零无效位置**:_validate_streams 检查 shape/dtype,然后 _zero_invalid 把 mask=False 的位置强制置零,防止 padding 残留脏数据。
2. **保存原始 antibody**(original_antibody),用于后面"无 antigen 则跳过整个交互块"的逻辑。
3. **Pre-Norm**:分别对两条流做 LayerNorm,归一化后再次清零无效位置(LayerNorm 的 bias 项会让 padding 位置变成非零,所以要再 mask 一次)。
4. **Cross-Attention**(_safe_cross_attention):
- antibody 做 query,antigen做 key/value;两者在进入attention前需要进行归一化
- key_padding_mask = ~antigen_mask,屏蔽 antigen 的 padding 位置;
- **特殊对待”整条 antigen 全是 padding"的样本**:如果某个 batch 行的 antigen mask 全都是 False,正常 attention 会因为所有 key 都被屏蔽导致 softmax 全都是 -inf → NaN。
	- 如何检测是否全为false呢，方法使用any函数，他会对某个维度做逻辑或运算，得到一个新的向量
	- 如果antigen mask确实全为0，attention模块会将对应抗体特征向量同样被强制替换为全零向量。
- 因此，attention模块的返回值有如下特点
	- 所有有效的 Query 位置（且有关键词上下文）：保留了经过交叉注意力计算后的真实特征向量。
	- 所有无效的 Query 位置（如 Padding 区域）：特征向量被强制替换为全零向量 [0.0, 0.0, ..., 0.0]。
	- 所有完全没有抗原上下文的样本：其抗体特征向量同样被强制替换为全零向量。
5. **残差更新 antibody**:antibody_tokens + dropout(antibody_delta),再清零无效位置。
6. **(若双向)同理算 antigen 的更新**。
7. **FFN 阶段**:对刚更新的 antibody/antigen 做 Pre-Norm → FFN → 残差相加 → 再清零无效位置,模式和 cross-attention 阶段完全一致。
8. **关键的 bypass 逻辑**(
   updated_antibody = torch.where(has_antigen, updated_antibody, original_antibody)
如果某条样本完全没有 antigen(has_antigen=False),那么不管前面算出了什么,antibody 输出都直接还原成**输入时的原始值**,等价于这条样本根本没经过这个交互块。
## embedding_ranker.py

**打分模型的数据流**：

**模式1：Antibody Only**
batch.antibody_embeddings -> antibody_projection -> antibody_pooling -> scoring_head

**模式2：Concat**
batch.antibody_embeddings & batch.antigen_embeddings -> 分别投影与池化 -> 拼接（Concat） -> scoring_head

**模式3：Deep_cross_attention**
batch.antibody_embeddings & batch.antigen_embeddings -> 分别投影 -> DeepCrossAttention -> 分别池化 -> 拼接（Concat） -> scoring_head

**质检**
1. 基础参数边界
- 不支持的融合模式：检查 fusion_kind 是否属于预设的合法集合（antibody_only, concat, deep_cross_attention）。如果传入未知的字符串，直接报错。
- 隐藏层维度非法：检查 d_model 是否小于 1。模型的核心维度必须是正整数。
2. antibody_only（仅抗体模式）的边界
- 冗余的抗原维度：既然仅使用抗体，antigen_input_dim 必须为 None。如果传入了具体数值，说明配置冲突，触发报错。
- 非法的交互层数：该模式不涉及抗原交互，因此 num_layers 必须严格等于 0。如果大于 0，触发报错。
3. 涉及抗原的通用边界
- 缺失抗原维度：当融合模式不是 antibody_only 时（即 concat 或 deep_cross_attention），antigen_input_dim 绝对不能为 None。如果缺失，说明缺少了必要的抗原输入配置。
4. concat（简单拼接模式）的边界
- 非法的交互层数：concat 模式仅做简单的特征拼接，不包含深度交互，因此 num_layers 必须严格等于 0。如果大于 0，触发报错。
5. deep_cross_attention（深度交叉注意力模式）的边界
- 交互层数不足：深度交叉注意力机制的核心就是多层交互，因此 num_layers 必须至少为 1。如果设为 0，该模块将失去意义，触发报错。
- 注意力头数非法：num_heads 必须至少为 1。
- 维度无法整除：在多头注意力机制中，d_model 必须能被 num_heads 整除（d_model % num_heads == 0），否则无法将特征均匀分配给各个注意力头，触发报错。

## losses.py

Pointwise 方法将排序问题简化为了一个普通的回归问题。它不关心样本之间的相对顺序，而是要求模型预测的分数（scores）尽可能逼近真实的排名标签（rank_targets）。

- 标签归一化：代码中要求 rank_targets 必须在 [0, 1] 之间（通常代表组内归一化后的排名，如第1名是1.0，最后一名是0.0）。
- 误差计算：根据 loss_type 参数，计算预测分数与目标标签之间的差异。
	- 如果选择 huber：使用 Huber Loss，它在误差较小时表现为平方误差，在误差较大时表现为线性误差，对异常值（如实验噪声）比 MSE 更鲁棒。
	- 如果选择 mse：使用均方误差（MSE），直接计算差值的平方。
- 运算本质：直接拉近预测值与真实排名的绝对距离。
- 数据质检：
	- 形状对齐质检：检查模型预测分数（scores）与真实排名标签（rank_targets）的张量形状是否完全一致。如果不一致，直接抛出异常。
	- 数值有效性质检：检查 scores 和 rank_targets 中是否包含 NaN（非数字）或 Inf（无穷大）等无效数值，确保所有参与运算的数据都是有限的（finite）。
	- 标签范围质检：严格检查 rank_targets 的值是否全部落在 [0, 1] 的区间内，确保输入的确实是经过组归一化处理的排名值。

  

Pairwise 并不预测具体分数，而是预测“A和B谁更好”。它的运算完全基于两个样本得分的**差值**。
- 分数差值计算：首先计算两个样本的分数差
- 缩放与映射：将差值乘以一个缩放因子 sigma（即代码中的 logits = sigma * (score_i - score_j)），将其映射为逻辑回归的 Logits。
- 交叉熵损失：使用带 Logits 的二元交叉熵（binary_cross_entropy_with_logits）计算损失。
- 运算本质：将排序问题转化为二分类问题，通过最小化错误排序的概率来优化模型。
如果在DataConfig里面开启了weight_pairs_by_group_size = True，那么在做cross_entropy_loss时还会根据 样本数量 和 组别大小 之间的关系进行重加权


ListNet 将视野扩大到整个列表（Group），通过比较预测的概率分布和真实的概率分布来优化。
- 构建真实目标分布（Target Softmax）：
	- 代码中的 _tie_aware_rank_targets 函数首先根据真实标签计算每个样本的“经验排名”（处理了相同分数的并列情况）。
	- 将排名除以 temperature（温度系数），然后进行 softmax 运算（target_probabilities），将排名转化为一个概率分布。排名越靠前的样本，其目标概率越高。
- 构建预测分布（Prediction Softmax）：
	- 将模型输出的原始分数（scores）同样除以 temperature，然后进行 log_softmax 运算（prediction_log_probabilities），得到模型预测的概率分布。
- 分布对齐（交叉熵）：
	- 计算目标概率与预测对数概率的乘积，并在有效成员（member_mask）上求和取负，得到每个组的交叉熵损失。
- 运算本质：$min - ∑P_{target}(x)(log P_{prediction}(x))$它强迫模型输出的分数排序（Softmax后的分布）与真实的标签排序分布高度一致。
数据质检：
第一层：基础输入质检（由 _validate_listwise_inputs 执行）
- 温度参数质检：检查 temperature（温度系数）是否大于 0，防止出现除以零或负数导致的数学异常。
- 张量维度与形状质检：确保 scores 和 labels 是二维张量且形状完全匹配 [G, M]（G为组数，M为成员数）。
- 掩码合法性质检：检查 member_mask 必须是布尔类型（BoolTensor），且形状与 scores 完全一致。
- 数据类型质检：确保 scores 和 labels 必须是浮点数类型（floating-point）。
- 有效区域数值质检：仅针对掩码标记为“有效”的区域（scores[member_mask]），检查其数值是否全部有限（finite）。
第二层：业务逻辑质检（由 listnet_valid_group_mask 执行）
- 有效组过滤质检：检查每一个组（行）是否满足排序的基本前提条件。只有同时满足以下两个条件的组才会被保留，不满足的组将被丢弃：
1. 该组内至少有 2 个有效成员（counts >= 2）。
2. 该组内至少存在 2 个不同的标签（maximum > minimum），即组内必须有排序的区分度。
全局可训练性质检：在执行完过滤后，还会检查当前批次（batch）中是否还剩下至少一个有效组。如果所有的组都被过滤掉了，说明当前批次没有任何可训练的样本，此时会抛出异常。

## ranker.py

**AffinityRanker

**数据输入**：

接收 RankBatch 格式的原始数据，包含抗体/抗原的原始 Token 序列及其对应的掩码（Mask）。

**模型结构**：

模型内部直接集成了基础预训练编码器（如 ESM2、IgBert 等）。在前向传播时，模型会先通过编码器提取隐藏层状态（Hidden States），随后经过 Masked-Mean 池化层、可选的单层交叉注意力（Cross-Attention）模块，最终由标量打分头（Scalar Head）输出结果。

**运行机制（Online 模式）**：

在“在线”模式（如 mode="frozen_online" 或未来的 LoRA 微调模式）下，模型在每个 Batch 的训练或推理过程中，都需要实时执行基础编码器的前向计算。基础编码器是否参与梯度更新（保持冻结或使用 LoRA 微调）取决于具体配置，但其前向计算过程始终存在于当前的计算图中。


**EmbeddingAffinityRanker**

**数据输入**：

接收 EmbeddingBatch 格式的数据。该数据包含预先计算并持久化缓存至磁盘的抗体/抗原 Embedding 向量，不包含原始 Token 序列，模型内部也不包含基础编码器。

**模型结构**：

模型仅包含轻量级的特征处理与打分模块，具体包括：

- Token 投影层（Token Projection）：将缓存的 Embedding 维度映射至模型的 d_model 维度。
- 池化层（Pooling）：支持多种池化策略，不再局限于 Masked-Mean。
- 深度交叉注意力模块（Deep Cross-Attention）：支持构建更深的交互网络（如 4/8/16 层），显著提升了特征交互能力。
- 标量打分头（Scalar Scoring Head）：用于输出最终的亲和力得分。

**运行机制（Frozen-Cached 模式）**：

对应 mode="frozen_cached" 模式。基础编码器的前向计算被前置处理，生成的 Embedding 向量被分片缓存至 processed/embeddings/... 目录。在训练阶段，模型仅读取缓存数据，并只针对投影层、交互层和打分头进行参数更新。这种机制有效避免了每个 Epoch 重复执行大模型前向计算，大幅降低了算力开销。