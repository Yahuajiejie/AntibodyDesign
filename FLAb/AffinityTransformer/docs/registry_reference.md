# AffinityTransformer Registry 代码质检文档

registry stage 的范围很明确：先构建 `antigen_registry.csv`，不做 v3.2 的 MSA 搜索、
embedding 批处理和 attention 训练。

核心输入：

```text
FLAb/data/binding/*.csv
FLAb/data/flab_metadata.csv
references/task_docs/TASKS.md
competition_data/_Preliminary_SequenceData/22/proteinbase_all_data_28_01_2026.csv
competition_data/_Preliminary_NanobodyData/ANDD.xlsx
competition_data/_Preliminary_StructureData/sabdab_summary_all.tsv
```

核心输出：

```text
FLAb/results/v3/registry/antigen_registry.csv
FLAb/results/v3/registry/antigen_registry_issues.csv
FLAb/results/v3/auxiliary/task_controls.csv
FLAb/results/v3/auxiliary/proteinbase_targets.csv
```

## 一、新增模块：`registry/sources.py`

### `extract_markdown_fasta_records(path)`

参数：

- `path`: markdown 文档路径，当前主要是 `TASKS.md`。

返回：

- `list[MarkdownFastaRecord]`，每条包含 `header / sequence / line_start`。

功能：

- 从非标准 markdown 文档里抽取 FASTA-like 序列。

实现：

- 扫描以 `>` 开头的 header；
- 后续只收集看起来像蛋白序列的行；
- 中文说明、空行、普通段落会跳过。

改进思路：

- `TASKS.md` 不是标准 FASTA 文件，不能直接交给普通 FASTA parser；
- registry stage 用宽松 markdown parser，把官方抗原序列变成可复现输入。

### `parse_tasks_markdown(path)`

参数：

- `path`: `references/task_docs/TASKS.md`。

返回：

- `(task_antigen_registry, task_controls)`。

功能：

- 提取官方 Nipah G protein / 40980-V08H 序列；
- 提取 m102.4、n425、HENV-26 对照抗体序列。

实现：

- `40980-V08H` 记录写入 `task_antigen_registry`；
- 对照抗体写入 `task_controls`，不进入训练标签；
- Nipah G 标记为 `sequence_source=tasks:40980-V08H`、
  `sequence_confidence=high`。

改进思路：

- 旧方案只说 `TASKS.md` “可能有用”；
- registry stage 正式把它作为官方靶标来源；
- 对照抗体单独保存，供相似性过滤和 sanity check，不污染训练集。

### `read_csv_or_zip_tables(path, low_memory=False)`

参数：

- `path`: `.csv` 或 `.csv.zip`。
- `low_memory`: 传给 `pandas.read_csv`。

返回：

- `dict[str, DataFrame]`。

功能：

- 读取单个 CSV；
- 或读取 zip 包内所有 CSV。

实现：

- 普通 CSV 直接 `pd.read_csv`；
- `.csv.zip` 使用 `zipfile.ZipFile` 遍历所有 `.csv` member；
- 每张表增加 `source_file` 和 `source_member`。

改进思路：

- 旧读取逻辑只读 zip 中第一个 CSV；
- registry 阶段不能静默漏掉 zip 内其它表。

### `load_binding_tables(binding_dir, pattern="*.csv*", max_files=None)`

参数：

- `binding_dir`: FLAb binding 数据目录。
- `pattern`: glob pattern。
- `max_files`: 只读前 N 个文件，方便 debug。

返回：

- `dict[str, DataFrame]`。

功能：

- 批量读取 `FLAb/data/binding`。

### `load_flab_metadata(metadata_csv)`

参数：

- `metadata_csv`: `FLAb/data/flab_metadata.csv`。

返回：

- `dict[str, dict]`，按 `filename` 建索引。

功能：

- 给 `build_antigen_registry` 提供 dataset-level metadata。

### `build_proteinbase_target_index(path)`

参数：

- `path`: `proteinbase_all_data_28_01_2026.csv`。

返回：

- proteinbase target 统计表。

功能：

- 从 `evaluations` JSON 中提取 `target`、`kd`、`binding`、
  `binding_strength` 等信息。

实现：

- 不把 proteinbase 行直接当 FLAb 训练样本；
- 只先统计 target 出现次数、Kd 数量、binding 阳性/阴性等。

改进思路：

- proteinbase 是候选/设计序列库，不是和 FLAb 完全同构的 benchmark；
- registry stage 先做 target index，避免把不同数据形态硬混。

### `proteinbase_target_registry(path)`

参数：

- `path`: proteinbase CSV。

返回：

- REGISTRY_COLUMNS 形状的 DataFrame。

功能：

- 把 proteinbase target index 转成 external antigen registry。

实现：

- `compatible_group = proteinbase:{target_slug}`；
- 多数 target 只有名字，没有序列，因此 `sequence_confidence=none`。

### `andd_antigen_registry(path)`

参数：

- `path`: `ANDD.xlsx`。

返回：

- REGISTRY_COLUMNS 形状的 DataFrame。

功能：

- 从 ANDD 中提取 `Ag_Name / Ag_Seq`。

实现：

- 使用 `pandas.read_excel`；
- 自动寻找 `Ag_Name`、`Ag_Seq`、`Affinity_Kd` 等列；
- 每个 unique antigen name/sequence 一行。
- 如果名字里有 digoxigenin/PEG 等半抗原词，但同时存在 `Ag_Seq`，registry stage
  先按 protein-like antigen 登记，并在 notes 里记录混合抗原提示。

改进思路：

- ANDD 的纳米抗体样本暂时不直接并入主训练；
- registry stage 先登记抗原资产，后续可单独做 nanobody ablation。

注意：

- 需要 `openpyxl>=3.1.0`。

### `sabdab_antigen_registry(path)`

参数：

- `path`: SAbDab summary tsv。

返回：

- REGISTRY_COLUMNS 形状的 DataFrame。

功能：

- 从 SAbDab summary 提取结构抗原索引。

实现：

- 读取 `pdb / antigen_chain / antigen_type / antigen_name`；
- 支持 `a | b | c` 这种多抗原字段；
- 不从 PDB 文件提取序列，序列留给 v3.1/v3.2。

改进思路：

- SAbDab summary 是结构索引，不等于完整序列源；
- registry stage 只记录 `sequence_source=sabdab:summary` 和 PDB accession。

### `build_external_antigen_registry(...)`

参数：

- `tasks_md`
- `proteinbase_csv`
- `andd_xlsx`
- `sabdab_summary_tsv`

返回：

- `(external_registry, auxiliary_tables)`。

功能：

- 汇总 TASKS/proteinbase/ANDD/SAbDab 外部来源。

实现：

- registry 表统一成 `REGISTRY_COLUMNS`；
- 辅助表目前包括 `task_controls` 和 `proteinbase_targets`。

## 二、新增模块：`registry/workflow.py`

### `RegistryBuildResult`

参数：

- `registry`: 合并后的 antigen registry。
- `issues`: 质检报告。
- `auxiliary_tables`: 辅助表。

返回：

- dataclass 本身不执行计算，只承载结果。

### `build_registry(...)`

参数：

- `datasets`: FLAb binding DataFrame dict，可为空。
- `metadata`: FLAb metadata dict，可为空。
- `group_col`: 默认 `compatible_group`。
- `tasks_md / proteinbase_csv / andd_xlsx / sabdab_summary_tsv`: 外部来源。
- `include_external`: 是否合并外部来源。

返回：

- `RegistryBuildResult`。

功能：

- 构建完整 antigen registry。

实现：

- 先调用已有 `build_antigen_registry` 从 FLAb 表中抽取抗原；
- 再调用 `build_external_antigen_registry` 合并外部来源；
- 最后按 `compatible_group` 去重并调用 `validate_antigen_registry`。

改进思路：

- 旧 registry 只看 FLAb CSV/metadata；
- registry stage 把官方任务靶标和外部结构/纳米抗体/设计库信息纳入统一索引；
- 但用 `source` 和 `confidence` 字段保留来源边界。

### `build_registry_from_paths(...)`

参数：

- `binding_dir`
- `metadata_csv`
- `binding_pattern`
- `max_binding_files`
- `tasks_md`
- `proteinbase_csv`
- `andd_xlsx`
- `sabdab_summary_tsv`
- `include_external`

返回：

- `RegistryBuildResult`。

功能：

- CLI 友好的路径入口。

实现：

- 读取 binding tables；
- 读取 metadata；
- 调用 `build_registry`。

### `write_registry_result(result, registry_path, issues_path=None, auxiliary_dir=None)`

参数：

- `result`: registry 构建结果。
- `registry_path`: registry 输出路径。
- `issues_path`: 质检报告输出路径。
- `auxiliary_dir`: 辅助表目录。

返回：

- `dict[str, str]`，记录实际写出的文件。

功能：

- 统一写出 registry、issues 和辅助表。

## 三、新增模块：`registry/build.py`

### `build_arg_parser()`

输入：

- 无。

输出：

- `argparse.ArgumentParser`。

功能：

- 定义 registry CLI 参数。

### `main()`

输入：

- 命令行参数。

输出：

- 写出 registry、issues、auxiliary tables；
- 在 stdout 打印行数和质检摘要。

运行：

```bash
python -m FLAb.AffinityTransformer.registry.build
```

## 四、已有函数，仅说明输入/输出/功能

### `antigen_schema.clean_text(value)`

输入：任意值。

输出：字符串。

功能：把 None、NaN、unknown、空白等整理成空字符串，其它值去掉首尾空格。

### `antigen_schema.normalize_antigen_sequence(value)`

输入：原始序列文本。

输出：标准化大写序列。

功能：去掉空格、换行、数字和常见间隔符。

### `antigen_schema.infer_antigen_type(...)`

输入：抗原名、序列、SMILES、glycan 信息。

输出：`protein / glycoprotein / peptide / small_molecule / carbohydrate / unknown`。

功能：保守推断抗原类型。

### `antigen_schema.clean_registry_enum(value, default)`

输入：registry 枚举字段和默认值。

输出：小写枚举字符串。

功能：清理 `antigen_type`、`sequence_confidence` 这类枚举字段。

改进思路：

- `clean_text("unknown")` 和 `clean_text("none")` 会返回空字符串；
- 但 `antigen_type=unknown`、`sequence_confidence=none` 是合法枚举；
- registry stage 新增该函数，避免 registry 质检误报。

### `antigen_schema.stable_antigen_id(...)`

输入：compatible_group、抗原名、抗原序列、accession。

输出：`ag_xxxxxxxxxxxx`。

功能：生成本项目内部 cache key。

### `antigen_registry.build_antigen_registry(datasets, metadata=None, group_col="compatible_group")`

输入：FLAb 数据表 dict、metadata dict。

输出：REGISTRY_COLUMNS 形状的 DataFrame。

功能：从 FLAb CSV 行内字段和 metadata 生成基础 antigen registry。

### `antigen_registry.validate_antigen_registry(registry, strict=False)`

输入：registry DataFrame。

输出：质检报告 DataFrame。

功能：检查必需列、重复 group、source/confidence 合法性、sequence flag 是否一致。

改动说明：

- `antigen_type` 和 `sequence_confidence` 改用 `clean_registry_enum`；
- 因此 `unknown` 和 `none` 不会再被误判为空值。

### `antigen_registry.write_antigen_registry(registry, path)`

输入：registry DataFrame 和输出路径。

输出：写出 CSV，无返回值。

功能：补齐列、按标准列顺序输出。

### `antigen_schema.ordered_registry_dict(row)`

输入：任意 registry-like dict。

输出：按 `REGISTRY_COLUMNS` 排列的 dict。

功能：补齐缺失列，并规范 bool / enum 字段。

改动说明：

- `antigen_type` 缺失时补 `unknown`；
- `sequence_confidence` 缺失时补 `none`；
- `sequence_source` 缺失时补 `missing`；
- `sequence_source=tasks:40980-V08H` 会保留原始编号大小写，便于人工质检。

## 五、registry stage 质检重点

- `TASKS.md` 的 Nipah G protein 应该出现为 `tasks:nipah_g_40980_v08h`；
- 该行 `sequence_source` 应为 `tasks:40980-V08H`；
- 该行 `sequence_confidence` 应为 `high`；
- 对照抗体应在 `task_controls.csv`，不应混入训练标签；
- zip 包内多个 CSV 不应只读取第一个；
- proteinbase 输出的是 target index，不是直接训练样本；
- ANDD 需要 `openpyxl`；
- SAbDab summary 暂不提取 PDB chain sequence，只登记结构索引。
