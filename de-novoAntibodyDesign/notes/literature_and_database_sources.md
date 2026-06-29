# Literature And Database Sources

## Official Data Sources

- RCSB PDB data API and file downloads: https://data.rcsb.org/
- RCSB file download pattern: `https://files.rcsb.org/download/{PDB_ID}.cif`
- UniProt programmatic access: https://www.uniprot.org/help/programmatic_access
- UniProt REST FASTA pattern: `https://rest.uniprot.org/uniprotkb/{UNIPROT_ID}.fasta`
- AlphaFold DB downloads: `https://alphafold.ebi.ac.uk/files/AF-{UNIPROT_ID}-F1-model_v4.cif`

## Design Tools To Consider

- RFantibody: https://github.com/RosettaCommons/RFantibody
- BindCraft: https://github.com/martinpacesa/BindCraft
- Germinal: https://github.com/SantiagoMille/germinal
- mBER: https://github.com/manifoldbio/mber-open
- BoltzGen: https://github.com/HannesStark/boltzgen
- ProteinMPNN: https://github.com/dauparas/ProteinMPNN
- AntiFold: https://github.com/oxpig/AntiFold
- ImmuneBuilder: https://github.com/oxpig/ImmuneBuilder

## Notes

- 设计阶段可以把多个工具输出统一到 `tables/candidate_scoring_table.csv`。
- 表位选择阶段优先把易突变数据、保守性和结构暴露放在一起看。
- 计算命中不代表实验命中，必须用独立模型和湿实验复核。

