# This_Is_Human_Reference

## `__init__.py`

```text
该文件主要是定义了可以被外部访问的接口
同时规定只有在使用指定函数或者模块时，才调用一些大型module
```

### `__getattr__(name)`

功能：懒加载 v3 registry 相关 API。

输入：

- `name`: 要访问的属性名。

输出：

- 对应对象；如果名称不存在，抛出 `AttributeError`。

## `antigen_schema.py`

```text
本模块主要定义了抗原类型枚举及抗原信息记录表的数据结构，并提供了通用函数，可以从任意源表格中提取并重构为目标格式的抗原信息表。
```

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



## `Registry/`

## `Registry/__init__.py`

## `Registry/core.py`

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

