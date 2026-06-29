# 05 Wet Lab Validation

目标：快速验证计算候选是否能在体外结合靶蛋白，并尽早识别跨变体鲁棒性。

> 这里给的是实验设计框架，不替代所在实验室 SOP。具体表达体系、缓冲体系、温度、时间、浓度、仪器方法和安全要求按实验室或 CRO 的标准流程执行。

## 首轮验证样品

靶蛋白：

- 主序列重组蛋白。
- 2 到 5 个关键变体蛋白。
- 如果目标是构象表位，优先使用保持天然构象的结构域或全长胞外域。

候选 binder：

- VHH/scFv/Fab/mini-binder 按设计格式表达。
- 每个表位至少保留多个候选，避免单一表位失败导致整轮失败。
- 保留阴性对照和阳性对照，如果有已知 binder。

## 验证路径

### 快速筛选

适合首轮大批量：

- ELISA 或类似 plate-based binding assay。
- BLI/Octet single-concentration screen。
- SEC shift 或 pull-down 作为辅助。

输出：

- 是否有可检测结合。
- 粗略强弱排序。
- 是否对标签、载体或非目标蛋白有背景结合。

### 亲和力测定

对初筛阳性候选：

- BLI 或 SPR 做 kinetic assay。
- 记录 kon、koff、KD。
- 对主序列和关键变体都测。

输出：

- 命中候选的亲和力等级。
- 是否主要由慢解离驱动。
- 是否对高频突变敏感。

### 表位与特异性

对强阳性候选：

- competition/binning assay。
- alanine scan 或变体面板映射关键接触位点。
- 与同源蛋白或非目标蛋白做交叉反应检查。

## 赶实验优先级

若时间紧，推荐：

1. 先合成 top 8 到 24 个候选。
2. 覆盖 2 到 4 个表位，而不是只押一个表位。
3. 每条计算路线至少有候选进入实验。
4. 第一轮只验证 binding/no binding 和粗略强弱。
5. 命中后再做 affinity maturation、humanization、format conversion。

## 首轮结果解释

- 计算高分但无结合：可能 predictor hacking、构象错误、表达折叠失败或表位不可及。
- 弱结合但跨变体稳定：值得进入优化。
- 强结合但只认单一变体：可作为工具分子，但不适合易突变靶标的主线。
- 表达差但结构预测好：可以尝试换格式、换 framework、做 developability redesign。

