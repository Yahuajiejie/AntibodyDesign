# AffinityTransformer v3 函数索引

这个文档只做函数级质检说明：**功能、输入、输出**。
实现细节先不展开，后续你问到某个函数时再单独解释。

## 目录结构

```text
AffinityTransformer/
  config.py                  全局配置和 encoder presets
  antigen_schema.py          registry 共享 schema，留在顶层供多个子包复用
  pipeline.py                高层调度入口，串起 registry/cache/feature matrix

  registry/                  antigen_registry 构建
    core.py                  FLAb CSV registry、读写、质检、合并
    sources.py               TASKS/proteinbase/ANDD/SAbDab/FLAb CSV 解析
    workflow.py              多来源合并、质检、写出
    build.py                 命令行入口

  encoders/                  预训练序列模型包装
    sequence.py              Hugging Face ESM2/IgBert/IgT5 encoder

  embeddings/                embedding 计算与 cache
    antibody.py              抗体 embedding
    antigen.py               抗原 single/MSA/ligand embedding cache
    msa.py                   ESM-MSA-1b embedding

  msa/                       同源序列与 MSA 构建
    homolog_search.py        FASTA 与 BLAST/MMseqs2 辅助函数
    builder.py               A3M 读写、抽样、质检

  data/                      训练特征矩阵
    context.py               antibody + antigen_context feature matrix

  models/                    torch 模型
    context.py               AntigenContextMLP 与 AffinityTransformer
```

放在顶层的文件只有三类：全局配置、共享 schema、高层 pipeline。其余按任务归入子包。

## `__init__.py`

### `__getattr__(name)`

功能：懒加载 v3 registry 相关 API。

输入：

- `name`: 要访问的属性名。

输出：

- 对应对象；如果名称不存在，抛出 `AttributeError`。

## `config.py`

### `SequenceEncoderSpec`

功能：保存一个序列编码模型的配置。

输入：

- `alias`: 项目内部短名。
- `model_name`: Hugging Face 或外部模型名。
- `embedding_dim`: 输出维度；未知时为 `None`。
- `tokenizer_style`: tokenizer 输入风格。
- `architecture`: 模型结构类型。
- `max_length`: 最大序列长度。
- `trust_remote_code`: 是否允许远程代码。
- `notes`: 说明文字。

输出：

- 一个配置对象。

### `V3Config`

功能：保存 v3 默认路径、模型、cache、训练参数。

输入：

- 类属性形式的配置项。

输出：

- 配置对象 `cfg`。

### `get_encoder_spec(alias_or_model_name)`

功能：根据 alias 或模型名取得 encoder 配置。

输入：

- `alias_or_model_name`: 预设 alias，或任意 Hugging Face model name。

输出：

- `SequenceEncoderSpec`。

## `antigen_schema.py`

### `AntigenRecord`

功能：表示 `antigen_registry.csv` 的一行。

输入：

- antigen ID、compatible group、抗原名、抗原类型、序列、来源、置信度、flags、备注等字段。

输出：

- 一个结构化 registry row 对象。

### `AntigenRecord.to_dict()`

功能：把 `AntigenRecord` 转为 dict。

输入：

- `self`。

输出：

- `dict[str, Any]`。

### `AntigenRecord.from_mapping(row)`

功能：从 dict 或 pandas row 创建 `AntigenRecord`。

输入：

- `row`: 含 registry 字段的映射对象。

输出：

- `AntigenRecord`。

### `clean_text(value)`

功能：把任意值清理成普通字符串。

输入：

- `value`: 可能是空值、数字、字符串等。

输出：

- 字符串；明显缺失值返回空字符串。

### `normalize_antigen_sequence(value)`

功能：标准化蛋白/肽抗原序列。

输入：

- `value`: 原始序列文本。

输出：

- 大写、去掉空格/数字/常见间隔符后的序列。

### `has_antigen_sequence(value)`

功能：判断一个字段是否含有效抗原序列。

输入：

- `value`: 原始序列字段。

输出：

- `bool`。

### `sequence_hash(sequence)`

功能：计算抗原序列 hash。

输入：

- `sequence`: 蛋白/肽序列。

输出：

- SHA1 字符串。

### `stable_antigen_id(compatible_group, antigen_name="", antigen_sequence="", sequence_accession="")`

功能：生成稳定的内部抗原 ID。

输入：

- `compatible_group`: 可比较组。
- `antigen_name`: 抗原名。
- `antigen_sequence`: 抗原序列。
- `sequence_accession`: 外部数据库 accession。

输出：

- 形如 `ag_xxxxxxxxxxxx` 的字符串。

### `infer_antigen_type(antigen_name="", antigen_sequence="", ligand_smiles="", glycan_info="")`

功能：粗略推断抗原类型。

输入：

- `antigen_name`: 抗原名。
- `antigen_sequence`: 抗原序列。
- `ligand_smiles`: 小分子 SMILES。
- `glycan_info`: 糖链信息。

输出：

- `protein / glycoprotein / peptide / small_molecule / carbohydrate / unknown`。

### `flags_from_antigen_type(antigen_type, antigen_sequence="")`

功能：根据抗原类型生成布尔 flags。

输入：

- `antigen_type`: 抗原类型。
- `antigen_sequence`: 抗原序列。

输出：

- dict，包含 `has_antigen_sequence`、`is_protein` 等 flags。

### `coerce_bool(value)`

功能：把常见文本 bool 转成 Python bool。

输入：

- `value`: bool、数字、字符串或空值。

输出：

- `bool`。

### `clean_registry_enum(value, default)`

功能：清理 registry 枚举字段，同时保留合法的 `unknown` / `none`。

输入：

- `value`: 原始枚举值。
- `default`: 缺失时的默认值。

输出：

- 枚举字符串。

### `ordered_registry_dict(row)`

功能：把 registry-like dict 整理成标准列顺序。

输入：

- `row`: 任意 registry-like dict。

输出：

- 按 `REGISTRY_COLUMNS` 排列的 dict。

## `registry/core.py`

### `_mode_text(series)`

功能：取一列文本中最常见的非空值。

输入：

- `series`: pandas Series。

输出：

- `(value, n_unique)`。

### `_mode_sequence(series)`

功能：取一列序列中最常见的非空标准化序列。

输入：

- `series`: pandas Series。

输出：

- `(sequence, n_unique)`。

### `_first_available_column(columns, candidates)`

功能：在候选列名中找到第一个实际存在的列。

输入：

- `columns`: 当前 DataFrame 列名。
- `candidates`: 候选列名。

输出：

- 列名字符串或 `None`。

### `_metadata_for_group(group_df, dataset_name, metadata)`

功能：找到某个 compatible group 对应的 metadata row。

输入：

- `group_df`: 一个 group 的 DataFrame。
- `dataset_name`: 数据集名。
- `metadata`: metadata dict。

输出：

- metadata row dict；找不到时返回空 dict。

### `_metadata_antigen_name(row)`

功能：从 metadata row 中取抗原名候选。

输入：

- `row`: metadata dict。

输出：

- 抗原名字符串；没有则返回空字符串。

### `_build_record_for_group(group_name, group_df, dataset_name, metadata_row)`

功能：为一个 compatible group 生成一条 `AntigenRecord`。

输入：

- `group_name`: compatible group 名。
- `group_df`: 该组样本表。
- `dataset_name`: 数据集名。
- `metadata_row`: 对应 metadata。

输出：

- `AntigenRecord`。

### `build_antigen_registry(datasets, metadata=None, group_col="compatible_group")`

功能：从 FLAb 数据表构建基础 antigen registry。

输入：

- `datasets`: 数据集名到 DataFrame 的 dict。
- `metadata`: metadata dict。
- `group_col`: 分组列名。

输出：

- registry DataFrame。

### `load_antigen_registry(path)`

功能：读取 registry CSV 并补齐标准列。

输入：

- `path`: registry CSV 路径。

输出：

- registry DataFrame。

### `write_antigen_registry(registry, path)`

功能：写出 registry CSV。

输入：

- `registry`: registry DataFrame。
- `path`: 输出路径。

输出：

- 无返回值。

### `validate_antigen_registry(registry, strict=False)`

功能：检查 registry 是否符合 v3 约定。

输入：

- `registry`: registry DataFrame。
- `strict`: 是否遇到 error 就抛异常。

输出：

- issues DataFrame。

### `validate_antigen_registry.add(severity, group, field, message)`

功能：在 `validate_antigen_registry` 内部追加一条质检问题。

输入：

- `severity`: 问题级别。
- `group`: compatible group。
- `field`: 字段名。
- `message`: 问题描述。

输出：

- 无显式返回；会修改外层 `issues` 列表。

### `merge_registry_updates(base_registry, updates, key="compatible_group", prefer_updates=True)`

功能：把人工补充表合并回基础 registry。

输入：

- `base_registry`: 基础 registry。
- `updates`: 更新表。
- `key`: 对齐字段。
- `prefer_updates`: 是否让更新表覆盖基础表。

输出：

- 合并后的 registry DataFrame。

### `load_like_registry(data)`

功能：把 DataFrame 或 list[dict] 整理成 registry 形状。

输入：

- `data`: DataFrame 或 list[dict]。

输出：

- registry DataFrame。

## `registry/sources.py`

### `MarkdownFastaRecord`

功能：保存从 markdown 中抽取的 FASTA-like 记录。

输入：

- `header`: FASTA header。
- `sequence`: 序列。
- `line_start`: header 所在行号。

输出：

- 一个记录对象。

### `TaskControlAntibody`

功能：保存 `TASKS.md` 里的对照抗体链。

输入：

- `control_name`: 对照名。
- `antibody_format`: Fv / VHH / unknown。
- `chain_role`: heavy / light / vhh / unknown。
- `sequence`: 抗体序列。
- `sequence_source`: 序列来源。
- `target_antigen`: 对应靶标。
- `notes`: 备注。

输出：

- 一个对照抗体对象。

### `slugify(value)`

功能：把文本变成适合做 ID 的 slug。

输入：

- `value`: 任意文本。

输出：

- 小写下划线 slug。

### `_letters_only(value)`

功能：只保留英文字母并转大写。

输入：

- `value`: 原始文本。

输出：

- 大写字母字符串。

### `_looks_like_protein_sequence_line(value)`

功能：判断一行是否像蛋白序列。

输入：

- `value`: 文本行。

输出：

- `bool`。

### `_clean_markdown_fasta_header(value)`

功能：清理 markdown FASTA header。

输入：

- `value`: 原始 header 行。

输出：

- 清理后的 header。

### `_remove_invisible_chars(value)`

功能：删除零宽字符。

输入：

- `value`: 原始文本。

输出：

- 删除零宽字符后的文本。

### `_is_markdown_fasta_header(value)`

功能：判断一行是否是 markdown FASTA header。

输入：

- `value`: 文本行。

输出：

- `bool`。

### `extract_markdown_fasta_records(path)`

功能：从 markdown 文档中抽取 FASTA-like 记录。

输入：

- `path`: markdown 文件路径。

输出：

- `list[MarkdownFastaRecord]`。

### `extract_markdown_fasta_records.flush()`

功能：在 markdown FASTA 解析过程中，把当前累积的 header/sequence 写入结果列表。

输入：

- 无显式参数；读取外层局部变量。

输出：

- 无显式返回；会修改外层 `records` 列表。

### `_task_header_kind(header)`

功能：判断 `TASKS.md` FASTA header 是靶标还是对照抗体。

输入：

- `header`: FASTA header。

输出：

- `antigen / control / unknown`。

### `_control_from_task_record(record)`

功能：把 TASKS 对照抗体 FASTA 记录转成结构化对象。

输入：

- `record`: `MarkdownFastaRecord`。

输出：

- `TaskControlAntibody`。

### `parse_tasks_markdown(path)`

功能：解析 `TASKS.md` 中的官方靶标和对照抗体。

输入：

- `path`: `TASKS.md` 路径。

输出：

- `(task_antigen_registry, task_controls)`。

### `read_csv_or_zip_tables(path, low_memory=False)`

功能：读取 CSV 或 CSV.zip 中的表。

输入：

- `path`: `.csv` 或 `.csv.zip` 路径。
- `low_memory`: pandas 读取参数。

输出：

- `dict[str, DataFrame]`。

### `_read_csv_path_with_fallback(path, low_memory=False)`

功能：用多种编码尝试读取 CSV 文件。

输入：

- `path`: CSV 路径。
- `low_memory`: pandas 读取参数。

输出：

- DataFrame。

### `_read_csv_bytes_with_fallback(data, low_memory=False)`

功能：用多种编码尝试读取 zip 内 CSV bytes。

输入：

- `data`: 原始 bytes。
- `low_memory`: pandas 读取参数。

输出：

- DataFrame。

### `load_binding_tables(binding_dir, pattern="*.csv*", max_files=None)`

功能：批量读取 FLAb binding 数据表。

输入：

- `binding_dir`: binding 目录。
- `pattern`: glob pattern。
- `max_files`: 最多读取文件数。

输出：

- `dict[str, DataFrame]`。

### `load_flab_metadata(metadata_csv)`

功能：读取 FLAb metadata。

输入：

- `metadata_csv`: metadata CSV 路径。

输出：

- 以 filename 为 key 的 dict。

### `_safe_json_loads(value)`

功能：宽松解析 JSON 字符串。

输入：

- `value`: JSON 字符串。

输出：

- Python 对象；失败时返回 `None`。

### `build_proteinbase_target_index(path)`

功能：从 proteinbase evaluations 中统计 target 信息。

输入：

- `path`: proteinbase CSV 路径。

输出：

- proteinbase target 统计 DataFrame。

### `proteinbase_target_registry(path)`

功能：把 proteinbase target 统计转成 registry。

输入：

- `path`: proteinbase CSV 路径。

输出：

- registry DataFrame。

### `_find_first_column(columns, candidates)`

功能：大小写不敏感地寻找候选列。

输入：

- `columns`: 实际列名。
- `candidates`: 候选列名。

输出：

- 找到的列名或 `None`。

### `andd_antigen_registry(path)`

功能：从 ANDD.xlsx 提取抗原名和抗原序列。

输入：

- `path`: ANDD Excel 路径。

输出：

- registry DataFrame。

### `_split_pipe_values(value)`

功能：拆分 SAbDab 的 `a | b | c` 字段。

输入：

- `value`: 原始文本。

输出：

- `list[str]`。

### `_normalize_sabdab_antigen_type(value, antigen_name)`

功能：把 SAbDab antigen type 转成本项目枚举。

输入：

- `value`: SAbDab 原始 antigen type。
- `antigen_name`: 抗原名。

输出：

- 本项目 antigen type。

### `sabdab_antigen_registry(path)`

功能：从 SAbDab summary 提取结构抗原索引。

输入：

- `path`: SAbDab summary TSV 路径。

输出：

- registry DataFrame。

### `build_external_antigen_registry(tasks_md=None, proteinbase_csv=None, andd_xlsx=None, sabdab_summary_tsv=None)`

功能：汇总 TASKS、proteinbase、ANDD、SAbDab 外部抗原来源。

输入：

- `tasks_md`: TASKS.md 路径。
- `proteinbase_csv`: proteinbase CSV 路径。
- `andd_xlsx`: ANDD Excel 路径。
- `sabdab_summary_tsv`: SAbDab summary 路径。

输出：

- `(external_registry, auxiliary_tables)`。

## `registry/workflow.py`

### `RegistryBuildResult`

功能：保存 registry 构建结果。

输入：

- `registry`: registry DataFrame。
- `issues`: 质检报告 DataFrame。
- `auxiliary_tables`: 辅助表 dict。

输出：

- 结果对象。

### `_deduplicate_compatible_groups(registry)`

功能：去除重复 compatible group。

输入：

- `registry`: registry DataFrame。

输出：

- 去重后的 registry DataFrame。

### `build_registry(...)`

功能：构建 antigen registry。

输入：

- `datasets`: FLAb binding DataFrame dict。
- `metadata`: FLAb metadata dict。
- `group_col`: 分组列。
- `tasks_md`: TASKS.md 路径。
- `proteinbase_csv`: proteinbase CSV 路径。
- `andd_xlsx`: ANDD Excel 路径。
- `sabdab_summary_tsv`: SAbDab summary 路径。
- `include_external`: 是否合并外部来源。

输出：

- `RegistryBuildResult`。

### `build_registry_from_paths(...)`

功能：从磁盘路径构建 antigen registry。

输入：

- `binding_dir`: binding 目录。
- `metadata_csv`: metadata CSV 路径。
- `binding_pattern`: binding 文件匹配规则。
- `max_binding_files`: 最多读取多少 binding 文件。
- `tasks_md`: TASKS.md 路径。
- `proteinbase_csv`: proteinbase CSV 路径。
- `andd_xlsx`: ANDD Excel 路径。
- `sabdab_summary_tsv`: SAbDab summary 路径。
- `include_external`: 是否合并外部来源。

输出：

- `RegistryBuildResult`。

### `write_registry_result(result, registry_path, issues_path=None, auxiliary_dir=None)`

功能：写出 registry、issues 和辅助表。

输入：

- `result`: registry 构建结果。
- `registry_path`: registry 输出路径。
- `issues_path`: 质检报告输出路径。
- `auxiliary_dir`: 辅助表输出目录。

输出：

- `dict[str, str]`，记录写出的文件路径。

## `registry/build.py`

### `build_arg_parser()`

功能：构建 registry CLI 参数解析器。

输入：

- 无。

输出：

- `argparse.ArgumentParser`。

### `_none_if_empty(value)`

功能：把空字符串参数转成 `None`。

输入：

- `value`: 字符串或 `None`。

输出：

- 字符串或 `None`。

### `main()`

功能：命令行入口，构建并写出 registry。

输入：

- 命令行参数。

输出：

- 无函数返回值；会写文件并打印摘要。

## `encoders/sequence.py`

### `sequence_hash(sequence)`

功能：计算单条序列 hash。

输入：

- `sequence`: 序列字符串。

输出：

- SHA1 字符串。

### `paired_sequence_hash(heavy, light)`

功能：计算 heavy/light 配对序列 hash。

输入：

- `heavy`: 重链序列。
- `light`: 轻链序列。

输出：

- SHA1 字符串。

### `format_single_sequence(sequence, tokenizer_style="raw")`

功能：按 tokenizer 风格格式化单条序列。

输入：

- `sequence`: 原始序列。
- `tokenizer_style`: `raw` 或 `space`。

输出：

- 格式化后的字符串。

### `format_paired_antibody_sequence(heavy, light, tokenizer_style="paired_t5")`

功能：格式化 heavy/light 配对输入。

输入：

- `heavy`: 重链序列。
- `light`: 轻链序列。
- `tokenizer_style`: tokenizer 风格。

输出：

- 格式化后的配对序列字符串。

### `HuggingFaceSequenceEncoder.__init__(spec, device=None)`

功能：创建 Hugging Face 序列 encoder 包装器。

输入：

- `spec`: `SequenceEncoderSpec`。
- `device`: 运行设备。

输出：

- encoder 对象。

### `HuggingFaceSequenceEncoder.load()`

功能：加载 tokenizer 和模型。

输入：

- `self`。

输出：

- 无显式返回；会设置对象内部 tokenizer/model。

### `HuggingFaceSequenceEncoder.encode_texts(texts, batch_size=16)`

功能：把已经格式化的文本编码成 embedding。

输入：

- `texts`: 文本列表。
- `batch_size`: batch 大小。

输出：

- `np.ndarray` embedding 矩阵。

### `HuggingFaceSequenceEncoder.encode_sequences(sequences, batch_size=16)`

功能：编码单链序列。

输入：

- `sequences`: 序列列表。
- `batch_size`: batch 大小。

输出：

- `np.ndarray` embedding 矩阵。

### `HuggingFaceSequenceEncoder.encode_paired_antibodies(heavy_sequences, light_sequences, batch_size=16)`

功能：编码 heavy/light 配对抗体序列。

输入：

- `heavy_sequences`: 重链序列列表。
- `light_sequences`: 轻链序列列表。
- `batch_size`: batch 大小。

输出：

- `np.ndarray` embedding 矩阵。

### `HuggingFaceSequenceEncoder.metadata()`

功能：返回 encoder 元信息。

输入：

- `self`。

输出：

- dict。

### `cache_paths(cache_dir, cache_key)`

功能：生成 embedding cache 路径和 metadata 路径。

输入：

- `cache_dir`: cache 目录。
- `cache_key`: cache key。

输出：

- `(embedding_path, metadata_path)`。

### `save_sequence_embedding(embedding, metadata, cache_dir, cache_key)`

功能：保存序列 embedding 和 metadata。

输入：

- `embedding`: embedding 向量。
- `metadata`: metadata dict。
- `cache_dir`: cache 目录。
- `cache_key`: cache key。

输出：

- 无显式返回。

### `load_cached_sequence_embedding(cache_dir, cache_key)`

功能：读取缓存的 sequence embedding。

输入：

- `cache_dir`: cache 目录。
- `cache_key`: cache key。

输出：

- `(embedding, metadata)`；不存在时返回 `(None, None)`。

### `get_or_compute_sequence_embeddings(sequences, encoder, cache_dir, sequence_kind, batch_size=16, force=False)`

功能：读取或计算一批序列 embedding。

输入：

- `sequences`: 序列列表。
- `encoder`: `HuggingFaceSequenceEncoder`。
- `cache_dir`: cache 目录。
- `sequence_kind`: 序列类型名。
- `batch_size`: batch 大小。
- `force`: 是否强制重算。

输出：

- `np.ndarray` embedding 矩阵。

## `embeddings/antibody.py`

### `_valid_sequence(value)`

功能：判断抗体序列是否有效。

输入：

- `value`: 序列字段。

输出：

- `bool`。

### `_zero(dim)`

功能：生成零向量。

输入：

- `dim`: 维度。

输出：

- `np.ndarray`。

### `embed_antibody_dataframe(df, cache_dir, encoder_alias="esm2_650m", layout="separate_chains", batch_size=16, force=False)`

功能：给 DataFrame 添加抗体 embedding 列。

输入：

- `df`: 含 heavy/light 或 paired 序列的 DataFrame。
- `cache_dir`: cache 目录。
- `encoder_alias`: encoder 名称。
- `layout`: `separate_chains` 或 `paired_chains`。
- `batch_size`: batch 大小。
- `force`: 是否强制重算。

输出：

- 新 DataFrame。

### `build_antibody_feature_matrix(df, layout="separate_chains")`

功能：把抗体 embedding 列拼成模型输入矩阵。

输入：

- `df`: 含 embedding 列的 DataFrame。
- `layout`: 抗体特征布局。

输出：

- `np.ndarray`。

## `embeddings/antigen.py`

### `AntigenEmbeddingManifest`

功能：保存一个抗原 embedding cache 的元信息。

输入：

- `antigen_id`
- `embedding_type`
- `model_name`
- `embedding_dim`
- `source_sequence_hash`
- `cache_path`
- `created_at`
- `notes`

输出：

- manifest 对象。

### `embedding_cache_paths(cache_root, antigen_id, embedding_type)`

功能：生成抗原 embedding 和 manifest 路径。

输入：

- `cache_root`: cache 根目录。
- `antigen_id`: 抗原 ID。
- `embedding_type`: embedding 类型。

输出：

- `(embedding_path, manifest_path)`。

### `_record_get(record, key, default="")`

功能：兼容 dict、pandas row、对象的取值。

输入：

- `record`: 数据记录。
- `key`: 字段名。
- `default`: 默认值。

输出：

- 字段值。

### `save_embedding_with_manifest(embedding, cache_root, antigen_id, embedding_type, model_name, source_sequence="", notes="")`

功能：保存抗原 embedding 和 manifest。

输入：

- `embedding`: embedding 向量。
- `cache_root`: cache 根目录。
- `antigen_id`: 抗原 ID。
- `embedding_type`: embedding 类型。
- `model_name`: 模型名。
- `source_sequence`: 来源序列。
- `notes`: 备注。

输出：

- `AntigenEmbeddingManifest`。

### `read_embedding_manifest(cache_root, antigen_id, embedding_type)`

功能：读取抗原 embedding manifest。

输入：

- `cache_root`: cache 根目录。
- `antigen_id`: 抗原 ID。
- `embedding_type`: embedding 类型。

输出：

- manifest dict。

### `load_antigen_embedding_cache(cache_root, antigen_id, embedding_type, expected_dim=None)`

功能：读取抗原 embedding cache。

输入：

- `cache_root`: cache 根目录。
- `antigen_id`: 抗原 ID。
- `embedding_type`: embedding 类型。
- `expected_dim`: 期望维度。

输出：

- `np.ndarray`。

### `has_cached_embedding(cache_root, antigen_id, embedding_type)`

功能：判断某个抗原 embedding 是否已缓存。

输入：

- `cache_root`: cache 根目录。
- `antigen_id`: 抗原 ID。
- `embedding_type`: embedding 类型。

输出：

- `bool`。

### `embed_antigen_single(record, cache_root, encoder_alias=cfg.ANTIGEN_SINGLE_ENCODER, force=False)`

功能：用单序列 encoder 计算蛋白/肽抗原 embedding。

输入：

- `record`: registry row。
- `cache_root`: cache 根目录。
- `encoder_alias`: encoder 名称。
- `force`: 是否强制重算。

输出：

- `AntigenEmbeddingManifest`。

### `embed_antigen_msa(record, cache_root, force=False)`

功能：用 MSA cache 计算抗原 MSA embedding。

输入：

- `record`: registry row。
- `cache_root`: cache 根目录。
- `force`: 是否强制重算。

输出：

- `AntigenEmbeddingManifest`。

### `embed_ligand(record, cache_root, force=False)`

功能：为小分子抗原预留 ligand embedding 入口。

输入：

- `record`: registry row。
- `cache_root`: cache 根目录。
- `force`: 是否强制重算。

输出：

- 当前实现通常抛出未实现异常或占位结果。

### `zero_embedding(dim)`

功能：生成指定维度零向量。

输入：

- `dim`: 维度。

输出：

- `np.ndarray`。

## `data/context.py`

### `_stack_embedding_column(df, column)`

功能：把 DataFrame 中的 embedding 列堆成矩阵。

输入：

- `df`: DataFrame。
- `column`: embedding 列名。

输出：

- `np.ndarray`。

### `build_antibody_feature_matrix(df, feature_mode="chain_concat")`

功能：构建抗体侧特征矩阵。

输入：

- `df`: 含 antibody embedding 的 DataFrame。
- `feature_mode`: 特征模式。

输出：

- `np.ndarray`。

### `antigen_type_one_hot(antigen_type)`

功能：把抗原类型转成 one-hot。

输入：

- `antigen_type`: 抗原类型。

输出：

- `np.ndarray`。

### `sequence_source_prefix(sequence_source)`

功能：取 sequence_source 的前缀。

输入：

- `sequence_source`: 来源字符串。

输出：

- 前缀字符串。

### `has_official_antigen_sequence(registry_row)`

功能：判断 registry 行是否有官方提供的抗原序列。

输入：

- `registry_row`: registry row。

输出：

- `bool`。

### `build_antigen_flags(registry_row, has_single_embedding_value, has_msa_embedding_value, official_sequence_value, msa_only_policy_value)`

功能：构建抗原上下文 flags。

输入：

- `registry_row`: registry row。
- `has_single_embedding_value`: 是否有 single embedding。
- `has_msa_embedding_value`: 是否有 MSA embedding。
- `official_sequence_value`: 是否有官方序列。
- `msa_only_policy_value`: 是否使用 MSA-only 策略。

输出：

- `np.ndarray`。

### `_registry_by_group(registry)`

功能：把 registry 转成 compatible_group 到 row 的映射。

输入：

- `registry`: registry DataFrame。

输出：

- `dict[str, Series]`。

### `_load_or_zero(cache_root, antigen_id, embedding_type, expected_dim, allow_missing)`

功能：读取 embedding；允许缺失时返回零向量。

输入：

- `cache_root`: cache 根目录。
- `antigen_id`: 抗原 ID。
- `embedding_type`: embedding 类型。
- `expected_dim`: 期望维度。
- `allow_missing`: 是否允许缺失。

输出：

- `(embedding, exists)`。

### `build_antigen_context_matrix(...)`

功能：为样本表构建 antigen context 矩阵。

输入：

- `df`: 样本 DataFrame。
- `registry`: antigen registry。
- `cache_root`: antigen cache 根目录。
- `group_col`: 分组列。
- `use_single`: 是否使用 single slot。
- `use_msa`: 是否使用 MSA slot。
- `include_type_flags`: 是否加入类型和 flags。
- `allow_missing`: 是否允许缺失 cache。
- `context_policy`: 上下文策略。

输出：

- `np.ndarray`。

### `build_antigen_context_feature_matrix(...)`

功能：拼接完整 v3 输入特征。

输入：

- `df`: 样本 DataFrame。
- `registry`: antigen registry。
- `cache_root`: antigen cache 根目录。
- `antibody_feature_mode`: 抗体特征模式。
- `group_col`: 分组列。
- `use_single`: 是否使用 single antigen。
- `use_msa`: 是否使用 MSA。
- `include_type_flags`: 是否使用 flags。
- `allow_missing`: 是否允许缺失。
- `context_policy`: 上下文策略。

输出：

- `np.ndarray`，顺序为 antibody + antigen context。

### `antigen_context_dim(use_single=True, use_msa=True, include_type_flags=True)`

功能：计算 antigen context 维度。

输入：

- `use_single`: 是否包含 single 维度。
- `use_msa`: 是否包含 MSA 维度。
- `include_type_flags`: 是否包含 flags。

输出：

- `int`。

## `models/context.py`

### `AntigenContextProjector.__init__(input_dim, output_dim=512, hidden_dim=None, dropout=0.2)`

功能：创建 antigen context 投影层。

输入：

- `input_dim`: 输入维度。
- `output_dim`: 输出维度。
- `hidden_dim`: 隐层维度。
- `dropout`: dropout。

输出：

- projector 模型对象。

### `AntigenContextProjector.forward(antigen_features)`

功能：投影 antigen context。

输入：

- `antigen_features`: `[batch, antigen_dim]`。

输出：

- `[batch, output_dim]`。

### `AntigenContextMLP.__init__(...)`

功能：创建抗体+抗原 MLP 打分模型。

输入：

- `antibody_dim`
- `antigen_dim`
- `antigen_projection_dim`
- `hidden_dim`
- `dropout`
- `project_antigen`

输出：

- MLP 模型对象。

### `AntigenContextMLP.forward(antibody_features, antigen_features)`

功能：计算抗体-抗原 pair 分数。

输入：

- `antibody_features`: `[batch, antibody_dim]`。
- `antigen_features`: `[batch, antigen_dim]`。

输出：

- `[batch]` score。

### `AntigenContextMLP.score_from_concat(features)`

功能：从拼接特征中切出 antibody/antigen 后打分。

输入：

- `features`: `[batch, antibody_dim + antigen_dim]`。

输出：

- `[batch]` score。

### `AffinityTransformer.__init__(...)`

功能：创建 modality-token Transformer 打分模型。

输入：

- `antibody_dim`
- `antigen_single_dim`
- `antigen_msa_dim`
- `flag_dim`
- `token_dim`
- `num_layers`
- `num_heads`
- `feedforward_dim`
- `dropout`
- `use_flags`

输出：

- Transformer 模型对象。

### `AffinityTransformer._infer_available(features)`

功能：根据特征是否全零推断 token 是否存在。

输入：

- `features`: `[batch, dim]`。

输出：

- `[batch]` bool tensor。

### `AffinityTransformer.forward(...)`

功能：计算抗体-抗原上下文 score。

输入：

- `antibody_features`
- `antigen_single_features`
- `antigen_msa_features`
- `flag_features`
- `single_available`
- `msa_available`

输出：

- `[batch]` score。

### `AffinityTransformer.score_from_concat(features)`

功能：从 v3 拼接特征中切出各部分并打分。

输入：

- `features`: `[batch, antibody + antigen_single + antigen_msa + flags]`。

输出：

- `[batch]` score。

## `msa/homolog_search.py`

### `FastaRecord`

功能：保存一条 FASTA 记录。

输入：

- `header`: FASTA header。
- `sequence`: 序列。

输出：

- FASTA 记录对象。

### `FastaRecord.normalized()`

功能：标准化 FASTA 记录。

输入：

- `self`。

输出：

- 新的 `FastaRecord`。

### `read_fasta(path)`

功能：读取 FASTA 文件。

输入：

- `path`: FASTA 路径。

输出：

- `list[FastaRecord]`。

### `write_fasta(records, path, line_width=80)`

功能：写出 FASTA 文件。

输入：

- `records`: FASTA 记录列表。
- `path`: 输出路径。
- `line_width`: 每行宽度。

输出：

- 无显式返回。

### `deduplicate_records(records)`

功能：按序列去重 FASTA 记录。

输入：

- `records`: FASTA 记录列表。

输出：

- 去重后的列表。

### `filter_homologs(records, query_sequence="", min_length=20, max_records=None)`

功能：过滤同源序列候选。

输入：

- `records`: FASTA 记录列表。
- `query_sequence`: query 序列。
- `min_length`: 最短长度。
- `max_records`: 最多保留数。

输出：

- 过滤后的 FASTA 记录列表。

### `write_homolog_fasta(query_record, homolog_records, path, max_records=None)`

功能：写出 query + homolog FASTA。

输入：

- `query_record`: query FASTA。
- `homolog_records`: 同源序列列表。
- `path`: 输出路径。
- `max_records`: 最多同源序列数。

输出：

- 写出的 FASTA 记录数。

### `build_blastp_command(query_fasta, database, output_path, max_target_seqs=256, evalue=1e-3, num_threads=8)`

功能：构造 BLASTP 命令。

输入：

- `query_fasta`
- `database`
- `output_path`
- `max_target_seqs`
- `evalue`
- `num_threads`

输出：

- 命令列表。

### `build_mmseqs_easy_search_command(query_fasta, database, output_dir, tmp_dir, max_seqs=256, threads=8)`

功能：构造 MMseqs easy-search 命令。

输入：

- `query_fasta`
- `database`
- `output_dir`
- `tmp_dir`
- `max_seqs`
- `threads`

输出：

- 命令列表。

### `run_search_command(command, cwd=None)`

功能：运行同源搜索命令。

输入：

- `command`: 命令列表。
- `cwd`: 工作目录。

输出：

- `subprocess.CompletedProcess`。

## `msa/builder.py`

### `strip_a3m_insertions(sequence)`

功能：去掉 A3M 中的小写插入字符。

输入：

- `sequence`: A3M 序列。

输出：

- 清理后的序列。

### `read_a3m(path)`

功能：读取 A3M 文件。

输入：

- `path`: A3M 路径。

输出：

- `list[FastaRecord]`。

### `write_a3m(records, path)`

功能：写出 A3M 文件。

输入：

- `records`: FASTA/A3M 记录。
- `path`: 输出路径。

输出：

- 无显式返回。

### `sample_msa_depth(records, max_depth=128, keep_query=True, seed=42)`

功能：抽样限制 MSA 深度。

输入：

- `records`: MSA 记录。
- `max_depth`: 最大深度。
- `keep_query`: 是否保留第一条 query。
- `seed`: 随机种子。

输出：

- 抽样后的记录列表。

### `validate_msa(records, query_sequence="", min_depth=2)`

功能：检查 MSA 是否可用。

输入：

- `records`: MSA 记录。
- `query_sequence`: query 序列。
- `min_depth`: 最小深度。

输出：

- issues 列表。

### `build_mafft_command(input_fasta, output_a3m, threads=8)`

功能：构造 MAFFT 命令。

输入：

- `input_fasta`
- `output_a3m`
- `threads`

输出：

- 命令列表。

## `embeddings/msa.py`

### `get_msa_model(model_name=cfg.ANTIGEN_MSA_MODEL_NAME)`

功能：加载 ESM-MSA 模型。

输入：

- `model_name`: 模型名。

输出：

- `(model, alphabet, batch_converter)`。

### `_pool_query_embedding(representations, query_index=0)`

功能：从 MSA 表征中取 query 序列平均 embedding。

输入：

- `representations`: 模型输出表示。
- `query_index`: query 所在行。

输出：

- `np.ndarray`。

### `embed_msa_file(msa_path, model_name=cfg.ANTIGEN_MSA_MODEL_NAME, max_depth=128)`

功能：读取 MSA 并计算 MSA-aware embedding。

输入：

- `msa_path`: A3M 路径。
- `model_name`: 模型名。
- `max_depth`: 最大 MSA 深度。

输出：

- `np.ndarray`。

## `pipeline.py`

### `AntigenEmbeddingPlanItem`

功能：保存一个抗原需要计算哪些 embedding。

输入：

- `antigen_id`
- `compatible_group`
- `needs_single`
- `needs_msa`
- `has_official_sequence`
- `reason`

输出：

- plan item 对象。

### `plan_antigen_embeddings(registry)`

功能：根据 registry 生成 antigen embedding 计划。

输入：

- `registry`: antigen registry DataFrame。

输出：

- `list[AntigenEmbeddingPlanItem]`。

### `embed_antigens_from_registry(registry, cache_root=cfg.ANTIGEN_CACHE_DIR, force=False, strict=True)`

功能：按计划生成 antigen embedding cache。

输入：

- `registry`: antigen registry。
- `cache_root`: cache 根目录。
- `force`: 是否强制重算。
- `strict`: 出错是否抛异常。

输出：

- `list[AntigenEmbeddingManifest]`。

### `validate_required_antigen_cache(registry, cache_root=cfg.ANTIGEN_CACHE_DIR)`

功能：检查训练所需 antigen cache 是否存在。

输入：

- `registry`: antigen registry。
- `cache_root`: cache 根目录。

输出：

- 缺失 cache 报告 DataFrame。

### `prepare_v3_dataframe(df, antibody_cache_dir=cfg.ANTIBODY_CACHE_DIR, antibody_encoder=cfg.ANTIBODY_ENCODER, antibody_layout=cfg.ANTIBODY_ENCODER_LAYOUT, batch_size=16)`

功能：为样本表生成抗体 embedding 列。

输入：

- `df`: 样本表。
- `antibody_cache_dir`: 抗体 cache 目录。
- `antibody_encoder`: encoder 名称。
- `antibody_layout`: heavy/light 布局。
- `batch_size`: batch 大小。

输出：

- 新 DataFrame。

### `build_v3_feature_matrix(df, registry, antigen_cache_dir=cfg.ANTIGEN_CACHE_DIR, antibody_layout=cfg.ANTIBODY_ENCODER_LAYOUT, allow_missing_antigen_context=cfg.ALLOW_MISSING_ANTIGEN_CONTEXT)`

功能：构建完整 v3 feature matrix。

输入：

- `df`: 样本表。
- `registry`: antigen registry。
- `antigen_cache_dir`: antigen cache 目录。
- `antibody_layout`: 抗体特征布局。
- `allow_missing_antigen_context`: 是否允许 antigen context 缺失。

输出：

- `np.ndarray`。
