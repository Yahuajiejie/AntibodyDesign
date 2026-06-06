# Bio-OS Antibody Affinity Modeling

本仓库是第四届 Bio-OS AI 开源大赛抗体设计赛道的工作目录，当前主线任务是构建抗原-抗体亲和力排序模型。仓库不是原版 FLAb 的完整镜像；FLAb benchmark、比赛官方数据、论文材料和我们的建模代码已经按用途重新整理。

当前实现重点是 `FLAb/affinity_model` 中的通用亲和力模型：使用冻结 ESM2 提取抗体序列 embedding，再训练一个跨数据集共享的 MLP head，对同一可比较组内的抗体亲和力进行排序。

## Repository Layout

```text
antibody/
  README.md
  .gitignore

  FLAb/
    train.py
    requirements.txt
    affinity_model/
      config.py
      data_loader.py
      embeddings.py
      dataset.py
      losses.py
      model.py
      trainer.py
      docs/
        code_reference.md
        code_reference_v1.md
        code_reference_v2_1.md
        code_reference_v3_plan.md
    data/
      flab_metadata.csv
      flab_metadata.json
    scripts/
      run.slurm
      run_zeroshot.slurm
    results/                 # local only

  competition_data/          # local only, not for GitHub
    _Preliminary_SequenceData/
    _Preliminary_StructureData/
    _Preliminary_NanobodyData/
    data/
    models/
    score/

  references/
    papers/
    task_docs/

  docs/
    中文解读.md
    lab.md
```

## What Is Tracked

`FLAb/affinity_model` is the active modeling code. It contains the data loader, ESM2 embedding cache logic, feature construction, losses, MLP model, training loop, split strategy, and evaluation logic.

`FLAb/data` should only keep lightweight metadata required by the loader, especially `flab_metadata.csv` and `flab_metadata.json`. Large raw benchmark CSVs, zipped datasets, generated embeddings, checkpoints, and result tables should stay local.

`competition_data` is a local archive of official competition files, moved FLAb official baseline scripts, and official score artifacts. It is useful for inspection and data mining, but it is too large and licensing-sensitive for normal GitHub commits.

`references` stores task documents and papers used during analysis. Large PDFs should be treated as local references unless redistribution is clearly allowed.

## Model Status

### v1

The first supervised affinity model used ESM2 embeddings and pairwise ranking losses, but it had a critical design problem: it trained separate task heads for different datasets. That made the validation score look better than it should and did not match the competition setting, where the model should produce a generally meaningful ranking rather than rely on per-dataset heads.

Other v1 issues included mixing `Kd`, `-logKd`, `IC50`, and other assay semantics too easily, creating tiny validation/test splits for small datasets, and merging datasets that were not physically comparable.

### v2.1

The current implementation trains one shared `AffinityMLP` head across eligible Kd-style data. The important constraints are:

- use only accepted Kd affinity data by default;
- keep each original compatible benchmark as a `compatible_group`;
- construct RankNet/Hinge pairs only within the same `compatible_group`;
- split train/validation/test by whole groups, not random rows;
- support MSE as a baseline using group-normalized `label_z`;
- use `chain_concat` by default: heavy-chain embedding concatenated with light-chain embedding;
- evaluate Spearman per group, then report aggregate metrics.

RankNet and hinge losses keep the original pair direction design. v2.1 does not add heavy-light difference vectors, product vectors, antigen attention, or MSA features.

### v3 Plan

v3 is currently a technical proposal, not implemented code. The planned direction is to add antigen context:

- build an antigen registry from FLAb, competition data, SAbDab/ANDD/proteinbase, and external identifiers where needed;
- add antigen protein embeddings when antigen sequence is available;
- use homolog search and MSA-aware embeddings for protein antigens with enough sequence evidence;
- handle non-protein antigens with separate molecular features and explicit type flags;
- keep missing-antigen handling explicit instead of silently pretending every sample has the same context.

See [code_reference_v3_plan.md](FLAb/affinity_model/docs/code_reference_v3_plan.md) for the detailed design.

## Running

Install dependencies in an environment with PyTorch, Transformers, pandas, NumPy, SciPy, scikit-learn, and tqdm. On the cluster, the intended environment name has been `esm2`.

From the `FLAb` directory:

```bash
cd FLAb
python train.py --mode embed
python train.py --mode train
```

Or run the full pipeline:

```bash
cd FLAb
python train.py --mode all
```

Useful options:

```bash
python train.py --mode all --loss ranknet
python train.py --mode all --loss mse hinge ranknet
python train.py --mode train --model_feature_mode chain_concat
python train.py --mode train --model_feature_mode scfv_mean
python train.py --mode train --checkpoint_metric val_weighted_spearman
python train.py --mode train --min_label_diff 0.1
```

Cluster scripts live in `FLAb/scripts`. They should be checked against the current cluster module/conda setup before submission.

## Documentation

The code quality documents are split to keep Markdown files manageable:

- [code_reference.md](FLAb/affinity_model/docs/code_reference.md): index page;
- [code_reference_v1.md](FLAb/affinity_model/docs/code_reference_v1.md): v1 logic and known problems;
- [code_reference_v2_1.md](FLAb/affinity_model/docs/code_reference_v2_1.md): current v2.1 implementation notes;
- [code_reference_v3_plan.md](FLAb/affinity_model/docs/code_reference_v3_plan.md): antigen-aware v3 proposal.

Task materials are under `references/task_docs`, including the moved `TASKS.md`.

## GitHub Hygiene

Do not commit local official data, raw benchmark archives, generated embeddings, model weights, or result folders. Before staging, check:

```bash
git status --short
git ls-files | rg "competition_data|score/|results/|\\.zip|\\.pdf|\\.csv\\.zip|\\.pt|\\.pth|\\.npy|\\.npz|\\.pkl"
```

Recommended local-only paths and file types:

```text
competition_data/
FLAb/results/
FLAb/score/
FLAb/cache/
*.csv
*.csv.zip
*.tsv
*.zip
*.gz
*.pt
*.pth
*.npy
*.npz
*.pkl
*.pdf
```

`FLAb/data/flab_metadata.csv` is an exception because the loader uses it to identify assay types and filter Kd-compatible datasets.

For clean history, keep commits separated:

```text
1. directory cleanup and ignore rules
2. affinity_model code changes
3. documentation updates
4. experiment results summaries
```

Avoid combining data movement, model changes, and documentation rewrites in one commit.
