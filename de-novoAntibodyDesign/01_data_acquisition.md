# 01 Data Acquisition

目标：从公开数据库和文献中收集序列、结构、变异、同源蛋白和已知功能位点，为表位选择和设计约束提供输入。

## 数据来源

优先使用官方数据源：

- UniProt：目标蛋白 FASTA、domain、PTM、feature annotation。
- RCSB PDB：实验结构，优先 mmCIF。
- AlphaFold DB：无实验结构或结构覆盖不足时作为补充。
- NCBI/ENA/相关物种数据库：同源序列与自然变异。
- InterPro/Pfam：domain 和 motif。
- 文献与补充材料：DMS、escape mutations、功能残基、已知抗体/配体结合位点。

## 下载目录建议

```text
data/
  raw/
    sequence/
    structures/
    variants/
    literature/
  processed/
    cleaned_structures/
    alignments/
    conservation/
    epitope_candidates/
```

## 最小数据包

首轮至少准备：

1. 靶蛋白 canonical FASTA。
2. 至少一个实验结构或 AlphaFold 结构。
3. 同源序列集合，用于保守性和变异熵计算。
4. 突变或变异频率表。
5. 已知功能位点、结合位点、结构域边界。

## 下载模板

使用：

```bash
bash scripts/download_target_data.sh
```

先在脚本里填：

```bash
UNIPROT_ID="P00000"
PDB_IDS=("XXXX" "YYYY")
```

## 数据质控

下载后立即检查：

- FASTA 是否对应正确 isoform。
- PDB chain 是否对应目标蛋白，而不是结合抗体、配体或融合标签。
- PDB 中是否缺失关键 loop。
- AlphaFold 结构的低 pLDDT 区域是否覆盖候选表位。
- 是否有 glycan、metal、cofactor、membrane 或 oligomer 状态影响表位可及性。

## 结构预处理

1. 统一 residue numbering。
2. 去掉无关链、晶体接触、非目标融合标签。
3. 保留可能影响表位的 cofactor/PTM/glycan 信息。
4. 对多构象结构做 ensemble，而不是只使用单一结构。
5. 为每个结构记录来源、分辨率、chain、覆盖范围和备注。

