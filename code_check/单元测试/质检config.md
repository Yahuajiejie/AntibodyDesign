# config.py

**1.模块入口**

`load_config(path)`

这是整个模块的“门面”，负责协调整个配置加载流程。

- 职责：读取 YAML 文件，解析顶层结构，并依次调用各个 _build_ 函数来构建完整的 Config 对象。
- 输入：YAML 文件路径 (Path)。
- 输出：完全填充的 Config 数据类实例。
- 边界检查：
	- 文件存在性：路径不存在则抛出 FileNotFoundError。
	- YAML 格式：确保文件内容是顶层的字典映射 (Mapping)，否则抛出 ValueError。
	- 必填章节：确保 YAML 中包含 data, model, train 三个顶级章节。

`_build_cross_validation_config(section, data_config)`

- 职责：专门处理可选的交叉验证配置。如果启用了交叉验证，会进行严格的逻辑检查，防止配置冲突。
- 输入：YAML 中的 cross_validation 字段（字典或 None），以及已构建的 data_config。
- 输出：CrossValidationConfig 对象。
- 边界检查：
	- 类型检查：确保 enabled 是布尔值，n_splits 是整数且 ≥2
	- 来源合法性：source 必须是 {"train", "train_valid"} 之一。
	- 业务逻辑互斥：
		- 若 enabled=True，则要求 data.split_strategy 必须为 "none"（即不能在自动切分数据的同时做交叉验证）。
		- 若 source="train_valid"，则要求 data.valid_path 必须存在。

**2. 配置构建与转换层**

`_build_data_config(section)`
- 职责：构建数据相关的配置，处理路径解析和采样策略。
- 输入：YAML 中的 data 字段。
- 输出：DataConfig 对象。
- 边界检查：
	- 必填字段：检查 train_path, valid_path, max_pairs_per_group, seed 是否存在。
	- 策略合法性：split_strategy 必须在 _VALID_SPLIT_STRATEGIES 集合中。
	- 路径依赖：
		- 如果不是 "none" 模式，必须提供 all_records_path 和 split_dir。
		- 如果是 "none" 模式，必须提供 train_path。
	- 数值范围：检查 valid_fraction + test_fraction 的和必须在 (0, 1) 之间。
`_build_model_config(section)`
- 职责：构建模型架构配置。它具有向后兼容性，能处理新旧两种 YAML 格式。
- 输入：YAML 中的 model 字段。
- 输出：ModelAbstractConfig 对象。
- 边界检查：
	- 格式判断：通过检测字段结构判断是新格式（包含 antibody_encoder 字典）还是旧格式（扁平结构）。
	- 组件一致性：调用 _validate_model_combination 检查编码器和交互层的逻辑一致性（例如：如果交互层是 antibody_only，则抗原编码器必须为 None）。

`_build_encoder_config(section, field_name)`

- 职责：构建单个编码器（抗体或抗原）的配置。
- 输入：编码器的字典配置及字段名（用于报错）。
- 输出：EncoderConfig 对象。
- 边界检查：
	- 必填字段：检查 name, revision, mode, embedding_layer, cache_dir, max_length, long_sequence_strategy。
	- 模式特定检查：
		- frozen_cached 模式：要求 cache_dir 必须存在且是目录，且 revision 不能是 main/latest（防止缓存失效），max_length 必须设置。
		- lora_online 模式：要求 lora_rank, lora_alpha 为正数，lora_dropout 在 [0,1) 之间，且 cache_dir 必须为 None。
	- 枚举值检查：mode 和 long_sequence_strategy 必须在允许的集合内。

`_build_interaction_config(raw)`
- 职责：构建交互层（如注意力机制）的配置。
- 输入：YAML 中的 interaction 字段。
- 输出：InteractionConfig 对象。
- 边界检查：
	- 维度合法性：d_model 必须能被 num_heads 整除。
	- 数值正负：ffn_multiplier 和 dropout 必须为正，且 dropout < 1。
	- 特定架构约束：
		- deep_cross_attention 要求 num_layers 必须是 {4, 8, 16} 之一。
		- 非 deep_cross_attention 架构要求 num_layers 必须为 0。

`_build_objective_config(raw)`

- 职责：构建训练目标（损失函数）的配置。
- 输入：YAML 中的 objective 字段。
- 输出：ObjectiveConfig 对象。
- 边界检查：
	- 目标类型：name 必须在 _OBJECTIVES 集合中（如 pairwise_ranknet）。
	- 物理意义：temperature 和 sigma 必须为正数。
	- 子项检查：pointwise_loss 必须是 "huber" 或 "mse"。

`_build_train_config(section)`

- 职责：构建训练循环的超参数。
- 输入：YAML 中的 train 字段。
- 输出：TrainConfig 对象。
- 边界检查：
	- 必填字段：检查 batch_size, lr, epochs, device。
	- 类型转换：确保数值类型正确（如 int, float）。

**3. 辅助与通用工具函数**

这些函数主要用于减少重复代码，提供统一的校验逻辑。

`_require_section(raw, name)`

- 职责：从顶层字典中提取子章节。
- 检查：确保子章节存在且为字典类型。

`_require_keys(section, keys, section_name)`

- 职责：批量检查字典中是否包含所有必需的键。
- 检查：如果缺失任何键，抛出包含缺失字段列表的异常。

`_require_existing_path(value, field_name) **&** _optional_existing_path`

- 职责：将字符串转换为 Path 对象并检查文件是否存在。
- 检查：路径必须指向一个存在的文件。

`_optional_path`

- 职责：将字符串转换为 Path 对象，但不检查文件是否存在（用于输出路径或可选路径）。

`_require_bool(value, field_name)`

- 职责：强制类型检查，确保值是布尔类型。

`_optional_int **/** _optional_float`

- 职责：安全地将值转换为数字，如果输入为 None 则返回 None。

`_legacy_encoder` **(内部调用)**

- 职责：仅用于旧版配置的兼容，将旧的字符串格式转换为默认的 EncoderConfig。