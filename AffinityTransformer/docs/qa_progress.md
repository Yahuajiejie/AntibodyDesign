# 质检进度跟踪

快照时间：2026-06-20，HEAD = `01dcab6`。按实际文件目录结构列出，不分优先级、不分层——勾上 = 这次会话里读过这个文件，没勾 = 还没看。

## affinity_transformer/

- [x] `config.py`
- [ ] `dataloader.py`
- [x] `metrics.py`
- [ ] `record_filter.py`
- [x] `splits.py`
- [ ] `trainer.py`
- [ ] `user_entry.py`
- [ ] `utils.py`

## affinity_transformer/dataset/

- [x] `__init__.py`
- [x] `datasets.py`
- [x] `examples.py`
- [x] `groups.py`
- [x] `pairs.py`
- [x] `records.py`
- [x] `schema.py`

## affinity_transformer/dataset/pair_sampling/

- [x] `__init__.py`
- [x] `blocks.py`
- [ ] `common.py`
- [ ] `heap_tree.py`
- [x] `labels.py`
- [x] `large_group.py`
- [ ] `randomized_tree.py`
- [ ] `tree.py`
- [x] `two_label.py`
- [ ] `validation.py`

## affinity_transformer/embeddings/

- [x] `__init__.py`
- [x] `collate.py`
- [x] `extractors.py`
- [x] `huggingface.py`
- [x] `pipeline.py`
- [x] `schema.py`
- [x] `store.py`
- [ ] `validation.py`

## affinity_transformer/model/

- [x] `__init__.py`
- [x] `attention.py`
- [x] `blocks.py`
- [x] `embedding_ranker.py`
- [x] `factory.py`
- [x] `heads.py`
- [x] `interaction.py`
- [x] `losses.py`
- [x] `pooling.py`
- [x] `projections.py`
- [x] `ranker.py`

## affinity_transformer/training/

- [ ] `__init__.py`
- [x] `artifacts.py`
- [ ] `cached.py`
- [ ] `cross_validation.py`
- [ ] `data.py`
- [ ] `evaluation.py`
- [ ] `loaders.py`
- [ ] `online.py`
- [x] `samplers.py`

## 根目录

- [ ] `train.py`
- [ ] `predict.py`
