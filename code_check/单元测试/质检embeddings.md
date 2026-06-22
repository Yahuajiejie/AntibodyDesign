# embedding/

## schema.py

主要是定义了基本数据类型，目前来看鲁棒性不错

  

先是定义了 antibody输入格式，兼容不同的抗体

再定义了antibody antigen 通用的embedding请求格式

最后定义了embedding后的返回格式

## extractors.py

这个模块主要定义了Embedding提取器的协议（编程规范）

```python
@runtime_checkable

class EmbeddingExtractor(Protocol):

    """Interface implemented by ESM-2, IgBERT, AbLang, and future adapters."""

  
```

所有的embedding提取器必须具备该模块定义的函数

```python
ExtractorFactory = Callable[..., EmbeddingExtractor]

_EXTRACTOR_FACTORIES: dict[str, ExtractorFactory] = {} 
```

ExtractorFactory是一个函数，可以接受任意参数，返回一个符合EmbeddingExtractor规范的对象，它是一个构造函数（？）

而_EXTRACTOR_FACTORIES则存储了当前所有已注册的ExtractorFactory

```python
 def register_embedding_extractor(
    name: str,
    factory: ExtractorFactory,
    *,
    replace: bool = False,
) -> None:
    """Register one explicit base-model adapter factory."""
```

主要是向_EXTRACTOR_FACTORIES注册表里加入**构造对象的函数**ExtractorFactory（不是已经创建好的 extractor 对象）。

```python
def build_embedding_extractor(name: str, **kwargs: object) -> EmbeddingExtractor:

    """Build a registered adapter without silently falling back."""
```

从_EXTRACTOR_FACTORIES注册表里取出这个构造函数，再调用它，创建并返回一个新的 extractor 对象。

  

## huggingface.py

```python
class _HuggingFaceTokenExtractor:
    """Shared frozen-forward implementation; subclasses own input formatting."""
```

该类旨在基于从 Hugging Face 模型库下载的预训练模型，实现一个通用的 Token 提取器

```python
    def __init__(
        self,
        *,
        encoder_name: str,
        encoder_revision: str,
        model: torch.nn.Module,
        tokenizer: object,
        device: str | torch.device = "cpu",
        embedding_layer: int = -1,
        output_dtype: torch.dtype = torch.float16,
        max_length: int | None = None,
    ) -> None:
```

这个子函数做一些通用的检查，并设置好模型、冻结参数、指定设备

```python
    def encode(
        self,
        requests: Sequence[EmbeddingRequest],
    ) -> Mapping[str, EmbeddingItem]:
        """Encode unique requests and return unpadded, non-special token states."""
```

一个从Sequence[request]到Mapping[str, EmbeddingItem]的函数

具体步骤：

1. **请求去重** 相同 sequence_hash 只编码一次，避免重复计算。
2. **模型专用格式化** 父类不知道抗体应该怎样输入。所以子类需要提供抗体输入格式

ESM-2：

```
抗原       -> ["ANTIGEN"]
single     -> ["SCFV"]
heavy+light -> ["HEAVY", "LIGHT"]
```

IgBERT：

```
heavy+light -> ["Q V L ... [SEP] D I Q ..."]
```

3. **对模型的输入格式进行扁平化（二维变一维）记录扁平化前后的映射**

```python
sequences.extend(formatted)
spans.append((start, end))
```

例如：

```
request 1: ESM-2 抗体 -> ["HEAVY", "LIGHT"] -> span=(0,2)
request 2: 抗原       -> ["ANTIGEN"]        -> span=(2,3)

最终 tokenizer 输入：
["HEAVY", "LIGHT", "ANTIGEN"]
```

spans 用于模型前向后重新拼回原 request。

4. **批量 tokenize**

```python
encoded = self._tokenize(sequences)
```

要求 tokenizer 返回：

```
input_ids
attention_mask
special_tokens_mask
```

如果配置了 max_length，会执行截断。

5. **只把模型需要的 tensor 放到 GPU**

```python
model_inputs = {
    input_ids,
    attention_mask,
    token_type_ids,  # IgBERT 可能使用
}
```

special_tokens_mask 留在 CPU，因为它只在后处理时使用。

6. **冻结前向**

```python
with torch.inference_mode():
    outputs = model(...)
```

构造 extractor 时已经执行：

```python
model.eval()
parameter.requires_grad_(False)
```

所以没有 dropout，也不保存反向图。

7. **选择 embedding 层**

```python
embedding_layer == -1
    -> outputs.last_hidden_state
否则
    -> outputs.hidden_states[embedding_layer]
```

输出暂时是：

```
[N_flat, L_padded, D]
```

8. **移除 padding 和 special token**

```python
valid = attention_mask[index] & ~special_tokens_mask[index]
piece = hidden[index, valid]
```

因此 [CLS]、[SEP]、[PAD] 都不会写入最终 cache。

9. **重新组装 request**

```python
values = torch.cat(pieces, dim=0)
```

ESM-2 的 heavy/light 会在这里变成：

```
[heavy residue embeddings;
 light residue embeddings]
```

IgBERT 本来就是一次 paired forward，因此只有一个 piece。

10. **转 CPU 并写成缓存对象**

```python
values.to(dtype=output_dtype, device="cpu")

EmbeddingItem.from_values(values)
```

此时已经去掉 padding，所以初始 mask 全 True。训练 batch 的新 padding mask 由 collate_embedding_batch() 重新生成。

  

  

## pipeline.py

```
def collect_embedding_requests(
    examples: Iterable[AffinityExample],
    *,
    include_antibodies: bool = True,
    include_antigens: bool = True,
) -> list[EmbeddingRequest]:
    """Collect unique structured sequence requests in deterministic order."""
```

函数功能是收集所有的序列embedding请求，去除重复的请求，并对得到的列表进行排序

有点奇怪的是，它将抗体的embedding请求和抗原的embedding请求混为一谈。这有点不规范，不过实际应用时，应当会对两者分别调用一次函数，**否则这个函数就是错的**

**实现思路**

- 如果include_antibodies为真，那么他们就构造antibody input对象，并为此调用antibody编码器
- 如果include_antigen为真，他们就调用antigen_embedding_request

可以看出，examples这个容器里放的要不然都 只包含antibody，要不然都 只包含antigen

要不然两者必须同时包含，**不然代码得到的就是错误的结果**

得到：requests[(request.sequence_type, request.sequence_hash)] = request

```python
def write_embedding_cache(
    requests: Sequence[EmbeddingRequest],
    extractor: EmbeddingExtractor,
    output_dir: Path,
    *,
    shard_size: int = 256,
) -> Path:
    """Encode requests and write manifest-indexed tensor shards.
    Existing manifests are rejected so a cache cannot silently mix encoder
    revisions or extraction rules.
    """
```

**实现思路**

**第一步**：检查参数。 参数`shared_size`代表一个pt文件最多包含的记录数目

**第二步**：确认目录中没有旧缓存，否则就报错。

```python
manifest_path = output_dir / "manifest.parquet"
metadata_path = output_dir / "metadata.yaml"

if manifest_path.exists() or metadata_path.exists():
    raise FileExistsError(...)
```

至于为什么只检查这两个文件，不检查pt文件，我觉得是因为这两个文件当且仅当所有请求都被正确处理时才会生成。所以我们不希望任何正确的结果被覆盖

**第三步**：建立目录，并对embedding请求进行去重

```python
output_dir.mkdir(parents=True, exist_ok=True)

shard_dir = output_dir / "shards"
shard_dir.mkdir(exist_ok=True)

unique = _unique_requests(requests)
```

**第四步**：分块调用extractor

```python
for shard_index, start in enumerate(
range(0, len(unique), shard_size)
):

chunk = unique[start : start + shard_size]
encoded = extractor.encode(chunk)
```

**第五步：**验证extractor是否正确返回结果，检查方法是请求的 hash 集合 == extractor 返回的 hash 集合

**第六步：**构建并保存[shared_xxx.pt](http://shared_xxx.pt)

- 生成shared文件名  shard_name = f"shard_{shard_index:05d}.pt"
- 构造shared内容   shard_payload: dict[str, dict[str, torch.Tensor]] = {}
- 保存shared   torch.save(shard_payload, shard_dir / shard_name)

**第七步**：为每条 embedding 建立 manifest 索引

```python
manifest_rows.append({
    "sequence_hash": request.sequence_hash,
    "sequence_type": request.sequence_type,
    "encoder_name": extractor.encoder_name,
    "encoder_revision": extractor.encoder_revision,
    "shard_path": f"shards/{shard_name}",
    "item_key": item_key,
    "sequence_length": _request_sequence_length(request),
    "embedding_length": item.values.shape[0],
    "embedding_dim": item.values.shape[1],
    "dtype": str(item.values.dtype).removeprefix("torch."),
})
```

**第八步：**写出manifest.parquet及metadata.yaml

  

## store.py

  

该文件主要定义embeddings 存放在内存与存放在硬盘的读写规范

 

```python
class EmbeddingNotFoundError(KeyError):
    """Raised when a required sequence is absent from an embedding store."""


class EmbeddingStore(Protocol):
    """Minimal cache interface consumed by embedding dataloaders."""

    def get(self, sequence_hash: str, sequence_type: SequenceType) -> EmbeddingItem:
        """Return one cached item or raise ``EmbeddingNotFoundError``."""
        ...
```

上述两个声明定义了Embedding存储器的协议与报错类型

```python
class InMemoryEmbeddingStore:
    """Small store used by tests and programmatic callers."""
```

用于测试和程序调用的小型存储器

```python
class ShardedEmbeddingStore:
    """Lazy reader for manifest-indexed ``torch.save`` embedding shards."""
```

用于实际生产、加载共享文件[shared_xxx.pt](http://shared_xxx.pt)的大型存储器。

**它的主要功能是**

根据sequence_type + sequence_hash 从 manifest 中找到 embedding 所在的 shard，按需加载 shard，并缓存最近使用的几个 shard。

write_embedding_cache

↓ 写入

manifest.parquet + shards/*.pt

↓ 读取

ShardedEmbeddingStore

  

**初始化`__init__()`**

**1.** **方法签名与参数**

```python
def __init__(self, manifest_path: Path) -> None:
```

manifest_path: manifest.parquet 文件的路径。

**2. 初始化执行步骤**

**① 检查 Manifest 文件是否存在**

- 将传入的路径转换为 Path 对象并赋值给 self.manifest_path。
- 检查该路径是否存在，若不存在则抛出 FileNotFoundError。

```python
self.manifest_path = Path(manifest_path)
if not self.manifest_path.exists():
    raise FileNotFoundError(f"embedding manifest not found: {self.manifest_path}")
```

**② 读取 Manifest 文件**

- 调用内部方法读取 parquet 文件。

```python
manifest = _read_manifest(self.manifest_path)
```

**③ 检查 Manifest 必须列**

- 校验读取到的 DataFrame 是否包含所有必需的列（MANIFEST_COLUMNS），缺失则抛出 ValueError。

**④ 按 Shard 路径对 Manifest 行进行分组**

- 目标结构：将数据转换为以 Shard 文件路径为键，该 Shard 包含的所有行记录列表为值的字典。

```y'a'm'l
# 目标结构示例
{
    Path("shards/shard_00000.pt"): [
        {"sequence_type": "antibody", "sequence_hash": "abc123", "item_key": "abc123", ...},
        {"sequence_type": "antigen", "sequence_hash": "def456", "item_key": "def456", ...}
    ]
}
```

- 具体操作：
  1. 逐行遍历：使用 to_dict(orient="records") 将 DataFrame 转换为字典列表。
  2. 校验序列类型：检查 sequence_type 是否严格为 antibody 或 antigen。
  3. 处理相对路径：若 shard_path 为相对路径，则基于 manifest_path 的父目录拼接为绝对路径。
  4. 分组聚合：使用 setdefault 将属于同一个 Shard 文件的行记录聚合在一起。

**⑤ 全量加载 Shard 并构建内存字典**

- 目标结构：将所有嵌入数据加载到内存，构建以 (sequence_type, sequence_hash) 为键，EmbeddingItem 为值的字典。

  ```python
  # 目标结构示例
  self._items = {
      ("antibody", "abc123"): EmbeddingItem(values=tensor(...)),
      ("antigen", "def456"): EmbeddingItem(values=tensor(...))
  }
  ```

- 具体操作：
  1. 检查 Shard 文件存在性：遍历分组后的 Shard 路径，若文件不存在则抛出 FileNotFoundError。
  2. 加载 Shard 数据：使用 torch.load 将 Shard 加载到 CPU 内存。优先尝试 mmap=True，若不支持则降级重试。
  3. 校验数据类型：检查加载的 Shard 内容是否为 Mapping 类型。
  4. 逐行严格校验：对于每个 Shard 中的行记录：
     - 检查 item_key 是否存在于 Shard 中，不存在则抛出 EmbeddingNotFoundError。
     - 调用 _coerce_item 转换数据。
     - Shape 校验：检查嵌入向量的形状是否与 Manifest 中记录的 embedding_length 和 embedding_dim 完全一致。
     - Dtype 校验：检查嵌入向量的数据类型是否与 Manifest 中记录的 dtype 完全一致。
  5. 查重与入库：构建组合键 key = (sequence_type, sequence_hash)，检查是否重复，若无重复则存入 self._items 字典。

**⑥ 记录加载日志**

- 所有 Shard 加载和校验完成后，打印日志记录成功加载的嵌入项数量和 Shard 文件数量。

  ```python
  _logger.info(
      "ShardedEmbeddingStore: loaded %d embeddings from %d shards (%s)",
      len(self._items),
      len(shard_to_rows),
      self.manifest_path,
  )
  ```

  

**其他辅助函数**：

def _read_manifest(path: Path) -> pd.DataFrame:

很简单，就是读取manifest表格

  

def _coerce_item(raw: object) -> EmbeddingItem:

很简单，就是把任何可能数据格式的raw转为Embedding_item

​    

  

## collate.py

  

这个文件实现了一个**基于预计算缓存的 Embedding数据整理器（Collator）**。它的核心功能是在模型训练或推理时，将原始样本（如抗体、抗原序列）转换为经过填充（Padding）的批量张量（Batch Tensors），以便神经网络能够高效处理。

具体来说，它实现了以下几个核心功能：

**1. 定义批量数据结构（Dataclasses）**

- EmbeddingBatch: 定义了单个批次的结构，包含抗体和抗原的嵌入向量（Embeddings）、对应的掩码（Mask，用于标识有效长度）、标签（Labels）以及记录ID等。
- PairEmbeddingBatch: 定义了成对样本（Pairwise）的批次结构，包含左侧（Left）和右侧（Right）的 EmbeddingBatch，以及用于 RankNet 等排序算法的目标值（y_ij）。

**2. 核心数据整理函数（Collators）**

- collate_embedding_batch: 接收一批 AffinityExample 样本，从缓存存储（EmbeddingStore）中读取抗体和抗原的嵌入向量，将它们对齐并填充成统一的张量形状，最后打包成 EmbeddingBatch 返回。
- collate_pair_embedding_batch: 专门处理成对样本。它分别对左侧和右侧的样本调用 collate_embedding_batch，然后将它们和成对标签组合成 PairEmbeddingBatch。

**3. 底层填充与对齐逻辑（Padding Utilities）**

- _pad_items: 将一个 EmbeddingItem 列表打包成张量。它会计算这批样本中的最大长度，并用 0 填充较短的序列，同时生成一个布尔掩码（Mask）来标记哪些位置是真实数据、哪些是填充数据。
- _pad_optional_items: 处理抗原等可能为 None 的可选数据。它会过滤出非空项进行填充，并将空项保留为全零张量。