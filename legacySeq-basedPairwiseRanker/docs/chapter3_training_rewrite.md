## 第三章 排序模型设计与训练流程

本研究面向抗体-抗原亲和力排序任务，构建了一套以同质实验分组为基本监督单元的成对排序学习框架。整体训练链路包括预训练蛋白表征缓存、抗体-抗原交互建模、组内可靠样本对采样、RankNet 成对优化以及 group-level 排序指标验证。具体而言，抗体与抗原序列首先由冻结的预训练蛋白模型编码为 token 级嵌入；训练阶段在每个可比分组内部构造排序样本对，并通过噪声感知多尺度采样筛选具有可靠标签差异的比较关系；模型随后对 pair batch 中的左右样本分别打分，以分数差计算成对排序损失，并通过反向传播更新投影层、交互模块、池化层与标量打分头参数。

### 3.1 训练任务定义与同质分组

本研究将抗体亲和力预测任务建模为组内排序任务。经过数据预处理后，每条记录包含抗体序列、抗原序列、标准化后的排序标签 `rank_label`、标签类型 `label_kind` 与可比分组标识 `group_id`。训练前首先过滤 `keep_for_training=True` 且 `rank_label` 为有限数值的记录，确保进入排序学习的数据均具备有效监督信号。

排序关系只在同一 `group_id` 内构造。这样做的原因是，不同实验来源、不同抗原背景或不同测量指标之间的数值尺度并不天然可比；如果跨组构造样本对，模型可能学到实验体系差异，而不是抗原-抗体亲和力差异。因此训练器实际优化的不是全局绝对数值，而是同一可比实验背景下候选抗体之间的相对顺序。

给定第 \(g\) 个 group：

$$
\mathcal{G}_g=\{(x_i, y_i)\}_{i=1}^{n_g},
$$

其中 \(x_i\) 表示第 \(i\) 条抗体-抗原记录，\(y_i\) 为对应的 `rank_label`。若 \(y_i>y_j\)，则认为样本 \(i\) 的排序优先级高于样本 \(j\)。训练样本对定义为：

$$
(i,j,y_{ij}),\quad
y_{ij}=
\begin{cases}
1, & y_i>y_j \\
0, & y_i<y_j
\end{cases}
$$

标签完全相同的样本对不进入训练，因为它们不能提供明确的排序方向。

### 3.2 Frozen-cache 表征训练范式

当前实现采用冻结预训练编码器的缓存式训练范式。训练开始前，抗体侧与抗原侧序列分别通过预训练蛋白模型生成 token 级嵌入，并保存到 embedding cache 中。训练时，DataLoader 根据 record 的序列哈希从缓存中读取嵌入，对变长 token 序列进行 padding，并生成对应 mask。

该范式将大规模预训练编码与排序模型训练解耦，带来三点优势。第一，训练阶段无需反向传播到基础模型，显著降低显存占用和训练时间。第二，不同采样策略、交互结构和训练超参数共享同一输入表征，便于公平消融。第三，预训练模型不直接接触下游训练标签，降低小样本任务上过拟合和实体记忆的风险。

在当前主配置中，抗体编码器采用 IgBert 缓存表征，抗原编码器采用 ESM-2 缓存表征。训练阶段可学习参数主要包括：抗体/抗原 token 投影层、深度交叉注意力交互层、mask-aware 池化层以及最终标量打分头。

### 3.3 排序打分模型

模型输入为缓存得到的抗体 token 表征与抗原 token 表征。两侧表征首先经过独立的 LayerNorm 与线性投影，被映射到共同的隐藏维度 \(d\)。在 cross-attention 主模型中，投影后的两侧 token 序列进入多层交互模块；在 concat 或 antibody-only 消融模型中，则跳过或省略相应交互结构。

深度交叉注意力模块由多层 InteractionBlock 堆叠而成。每个 block 采用 pre-norm 结构，先对抗体 token 和抗原 token 进行归一化，再执行双向交叉注意力：

1. 以抗体 token 为 Query，抗原 token 为 Key/Value，更新抗体侧表示；
2. 以抗原 token 为 Query，抗体 token 为 Key/Value，更新抗原侧表示；
3. 两侧分别经过前馈网络、残差连接和 dropout；
4. 所有 padding 位置由 mask 屏蔽，缺失抗原的记录会绕过交互更新，避免无效 token 影响训练。

交互后的两侧 token 表征分别通过 mask-aware pooling 汇聚为定长向量。当前主配置使用 `masked_mean` 池化，即只对真实 token 位置求平均。随后将抗体向量与抗原向量拼接，输入两层标量打分头：

$$
s=f_\theta(x)\in\mathbb{R}.
$$

该分数不经过 Sigmoid 或 Softmax 约束，其数值本身不被解释为绝对亲和力；训练只使用分数差 \(s_i-s_j\) 表示样本 \(i\) 相对样本 \(j\) 的排序偏好。

### 3.4 Group 内样本对构造与采样

训练过程中，原始 record 不会直接进入 RankNet 损失。训练开始前，训练集首先由 `build_pairs` 在各个 group 内构造成 pair 表；随后每个 epoch 对这张 pair 表重新打乱并遍历。该过程是本研究训练方法的重要组成部分，直接决定模型看到的排序监督信号。

基础候选对来自组内所有标签不同的两两组合，理论规模为 \(O(n_g^2)\)。对于小 group，代码可枚举全部候选对，并根据配置进行随机采样；对于大 group，则避免完整枚举，采用更节省内存的采样结构。当前实验重点使用噪声感知多尺度采样策略。

#### 3.4.1 噪声阈值 tau

多源实验数据存在测量误差。若两个样本的标签差距过小，强行把它们构造成硬 0/1 排序对，模型可能学习到噪声而不是真实亲和力差异。因此噪声感知采样为每个 group 引入阈值 \(\tau_g\)，只有当：

$$
|y_i-y_j|\ge \tau_g
$$

时，样本对才被认为具有可分辨排序信号。当前实现不是使用统一全局 tau，而是根据 group 的 `antigen_key` 从 tau registry 中查找数据源相关阈值；未匹配的数据源使用保守默认值。

#### 3.4.2 Tau-separated anchor 骨架

对每个 group，采样器首先按 `rank_label` 升序排列记录，然后从低到高扫描标签序列。扫描时并不只比较相邻记录，而是比较当前记录与当前 anchor 的标签差距；一旦差距达到 \(\tau_g\)，就开启新的 anchor。这样每段内部标签跨度小于 tau，而相邻 anchor 至少相隔 tau，避免了“许多小差距连续累积后被错误视为同一簇”的链式问题。

随后采样器只在 anchor 节点上构建平衡树式比较骨架。该骨架用 \(O(m)\) 量级的边连接 \(m\) 个 anchor，使远距离排序信号能够通过较短路径传播，同时避免大 group 中 \(O(n^2)\) 全量配对带来的计算开销。

#### 3.4.3 覆盖边与多尺度增强边

骨架只覆盖 anchor，非 anchor 记录还需要进入训练监督。采样器会为每个具备可分辨伙伴的非 anchor 记录寻找至少一个 coverage partner。寻找伙伴时按标签差距划分为近距、中距、远距三个尺度：

$$
[\tau,2\tau),\quad [2\tau,4\tau),\quad [4\tau,+\infty).
$$

采样器通过二分定位候选区间，再从候选区间中探测少量样本，而不是物化所有候选对。伙伴选择优先考虑当前 degree 较低的记录，使比较边尽量均匀分布，避免少数样本成为高连接度 hub。

在完成 coverage 后，采样器还会为每条记录额外添加若干 enrichment edges。第一条增强边优先选择近距样本，后续增强边偏向中距或远距样本，从而同时提供局部精细区分和全局排序约束。若某个 group 的标签跨度整体小于 tau，则该 group 不产生训练 pair；这表示该 group 在当前噪声阈值下没有可靠硬排序监督，而不是采样失败。

### 3.5 Pair DataLoader 与训练 batch

pair 表生成后，训练集被封装为 PairwiseAffinityDataset。每个样本包含 left record、right record 和二元排序标签 \(y_{ij}\)。在 frozen-cache 模式下，collate 函数分别为 pair 的 left 和 right 两侧读取缓存嵌入，并独立 padding 成两个 embedding batch：

$$
B=\{(x_i^{L},x_i^{R},y_i)\}_{i=1}^{b}.
$$

DataLoader 使用 group-aware shuffle：先随机打乱 group 顺序，再在每个 group 内随机打乱 pair 顺序。这样既保留了同一 group 内抗原和缓存读取的局部性，也避免固定顺序训练带来的偏差。每个 epoch 中，采样得到的 pair 行会被遍历一次；如果启用 group-size reweighting，则每个 pair 的 loss 还会按其来源 group 的记录数与采样 pair 数进行缩放，使大 group 的训练贡献更接近其真实规模。

### 3.6 RankNet 优化过程

当前完整训练 runner 支持的主目标是 pairwise RankNet。对一个 pair batch，训练器执行如下过程：

1. 将 left/right 两侧 embedding batch 移动到训练设备；
2. 模型分别对 left 和 right 前向计算，得到 \(s_i=f_\theta(x_i^L)\) 与 \(s_j=f_\theta(x_i^R)\)；
3. 计算分数差的 logit：

   $$
   z_{ij}=\sigma(s_i-s_j),
   $$

   其中 \(\sigma\) 是 RankNet 分数差缩放系数；

4. 使用 binary cross entropy with logits 计算成对排序损失：

   $$
   \mathcal{L}_{ij}
   =
   -y_{ij}\log p_{ij}
   -(1-y_{ij})\log(1-p_{ij}),
   \quad
   p_{ij}=\frac{1}{1+\exp(-z_{ij})};
   $$

5. 对 batch 内 pair loss 求均值或加权均值；
6. 执行 `optimizer.zero_grad()`、`loss.backward()`、`optimizer.step()`。

因此，反向传播只是训练 batch 的最后一步；在它之前，训练器已经依次完成了 group 内 pair 采样、缓存嵌入读取、left/right 双路 batch 构造、两次模型前向和成对损失计算。Adam 优化器只接收 `requires_grad=True` 的模型参数，因此 frozen-cache 训练阶段不会更新预训练基础编码器。

为提升训练稳定性，CUDA 训练时前向与损失计算使用 bfloat16 autocast。若某个 batch 产生 NaN loss，训练会立即中止，并保存包含 epoch、global step、左右 record id、group id、模型分数和 pair 标签的错误上下文，便于定位异常 group 或异常样本对。

### 3.7 验证、模型选择与输出

验证阶段不再使用 pair，而是回到 record 粒度。模型对验证集每条记录单独打分，得到：

$$
\{(record\_id, group\_id, y_i, s_i)\}.
$$

随后在每个 group 内计算 `rank_label` 与模型分数的 Spearman 相关系数。最终指标同时报告 macro Spearman 和按 group 记录数加权的 weighted Spearman，并按 `label_kind` 给出分层统计。标签数不足或预测分数退化的 group 会被计入 skipped groups，而不会被强行纳入相关系数平均。

每个 epoch 结束后，训练器记录训练损失和验证指标，并保存 latest checkpoint。当验证指标优于历史最优时，保存 best checkpoint；训练结束后，公开用于后续验证和测试预测的模型状态会回滚到 best epoch，而不是简单使用最后一个 epoch。最终输出包括 checkpoint、history、metrics、验证/测试 predictions、group metrics、资源统计和运行配置副本。

### 3.8 本章小结

综上，当前排序模型的核心并不是单独的交叉注意力结构，而是“同质 group 内构造可靠比较关系，再用共享打分模型学习这些比较关系”的训练体系。模型架构负责把抗体和抗原嵌入映射为可比较标量分数；噪声感知多尺度采样负责决定哪些样本对能提供可靠监督；RankNet 优化负责把这些局部比较约束转化为模型参数更新；group-level Spearman 验证则保证模型选择标准与实际排序目标一致。

因此，本章后续实验分析应围绕三类变量展开：交互结构是否提升抗原-抗体特征建模能力，pair 采样策略是否提高监督信号质量，以及 group-level 验证指标是否反映模型在真实候选抗体筛选场景中的泛化排序能力。
