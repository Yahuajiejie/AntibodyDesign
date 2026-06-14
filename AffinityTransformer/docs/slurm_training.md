# Slurm Training Guide

This guide assumes the project directory has been copied to the cluster and
commands are run from the project root on a login node.

Do not run training directly on the login node. Use the login node only for
environment setup, upload/download, and `sbatch`.

## 0. Partition choices

Available partitions from `sinfo`:

```text
GPUA800      GPU nodes, gpu:8 per node
GPUA40       GPU nodes, gpu:4 per node
C64M512G     CPU nodes, 512 GB memory class
C64M256G     CPU nodes, 256 GB memory class, cluster default
```

The Slurm templates use:

```text
g00_qc_and_splits.sbatch      C64M256G
warmup_esm2_cache.sbatch      C64M256G
smoke_test.sbatch             GPUA800
run_config.sbatch             GPUA800
run_group.sbatch              GPUA800
```

To use A40 instead of A800 for a GPU job, override at submit time:

```bash
sbatch -p GPUA40 \
  --export=ALL,CONFIG=configs/experiments/g01_maxctx_cross_attention.yaml \
  scripts/slurm/run_config.sbatch
```

To inspect concrete node hostnames rather than partition names:

```bash
sinfo -N -p GPUA800 -o "%N %t %G %m %c"
sinfo -N -p GPUA40  -o "%N %t %G %m %c"
scontrol show hostnames "$SLURM_NODELIST"   # inside a running job
```

## 1. Create the Conda environment

Recommended: create the environment through Slurm on a CPU node.

```bash
cd /gpfs/share/home/2300012116/path/to/AffinityTransformer
sbatch scripts/slurm/setup_affitest_env.sbatch
```

If your cluster policy allows package installation on the login node, the
same setup can also be run directly:

```bash
cd /gpfs/share/home/2300012116/path/to/AffinityTransformer
bash scripts/slurm/setup_affitest_env.sh
```

The default environment name is `affitest`. Override it only if needed:

```bash
AFFINITY_CONDA_ENV=affitest bash scripts/slurm/setup_affitest_env.sh
```

The setup script only loads:

```text
anaconda/3-2023.09-0
```

It installs `torch` from the CUDA 11.8 PyTorch wheel index, then installs
`requirements.txt`. It does not run tests on the login node.

The shared Anaconda installation is read-only. The script keeps conda's
writable package/env cache in the normal user location:

```text
$HOME/.conda/envs
$HOME/.conda/pkgs
```

The setup and training scripts support these overrides:

```text
AFFINITY_CONDA_ENV       default: affitest
AFFINITY_PROJECT_DIR     explicit project root if submitting outside the repo
CUDA_MODULE              default: cuda/11.8
MODULE_INIT              default: /gpfs/share/software/module/tools/modules/init/profile.sh
CONDA_INIT               default: /gpfs/share/software/anaconda/3-2023.09-0/etc/profile.d/conda.sh
HF_HOME                  default: <project>/cache/huggingface
TORCH_HOME               default: <project>/cache/torch
CONDA_ENVS_PATH          default: $HOME/.conda/envs
CONDA_PKGS_DIRS          default: $HOME/.conda/pkgs
```

## 2. Validate the environment on a compute node

```bash
sbatch scripts/slurm/smoke_test.sbatch
```

This checks `torch.cuda.is_available()` and runs `pytest` inside a Slurm job.

## 3. Optional: warm up the ESM2 cache

```bash
sbatch scripts/slurm/warmup_esm2_cache.sbatch
```

This downloads `facebook/esm2_t12_35M_UR50D` through Hugging Face inside a
Slurm job. If compute nodes cannot access the internet, download the cache on
an allowed node and set `HF_HOME` to that cache path before submitting jobs.

## 4. Build QC reports and fixed splits

```bash
sbatch scripts/slurm/g00_qc_and_splits.sbatch
```

Check progress:

```bash
squeue -u "$USER"
tail -f logs/slurm/aff-g00-qc-<jobid>.out
```

Expected outputs:

```text
reports/data/*.csv
processed/binding/filtered/*/all_records.parquet
processed/binding/splits/*/{train,valid,test}.parquet
processed/binding/splits/*/{split_summary,leakage_report}.csv
```

## 5. Submit one training config

```bash
sbatch \
  --job-name=aff-g01-cross \
  --export=ALL,CONFIG=configs/experiments/g01_maxctx_cross_attention.yaml \
  scripts/slurm/run_config.sbatch
```

To choose an explicit output directory:

```bash
sbatch \
  --job-name=aff-g01-cross \
  --export=ALL,CONFIG=configs/experiments/g01_maxctx_cross_attention.yaml,OUTPUT_DIR=outputs/slurm/g01_cross \
  scripts/slurm/run_config.sbatch
```

## 6. Submit a whole experiment group

```bash
sbatch \
  --job-name=aff-g01-core \
  --export=ALL,GROUP_SCRIPT=scripts/runs/g01_core_ablation.sh \
  scripts/slurm/run_group.sbatch
```

Other groups:

```bash
sbatch --job-name=aff-g02-label --export=ALL,GROUP_SCRIPT=scripts/runs/g02_label_source_ablation.sh scripts/slurm/run_group.sbatch
sbatch --job-name=aff-g03-pairs --export=ALL,GROUP_SCRIPT=scripts/runs/g03_pair_sampling_ablation.sh scripts/slurm/run_group.sbatch
sbatch --job-name=aff-g04-subset --export=ALL,GROUP_SCRIPT=scripts/runs/g04_antigen_subset_ablation.sh scripts/slurm/run_group.sbatch
```

## 7. Submit g00 then g01 as a dependency chain

```bash
bash scripts/slurm/submit_g00_g01_chain.sh
```

This submits `g01_core_ablation` only after `g00_qc_and_splits` finishes
successfully.

## 8. Common Slurm commands

```bash
squeue -u "$USER"
scancel <jobid>
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,MaxRSS,AllocGRES
tail -f logs/slurm/<job-name>-<jobid>.out
```

If the cluster requires a partition/account/QoS, add the corresponding
`#SBATCH --partition=...`, `#SBATCH --account=...`, or `#SBATCH --qos=...`
line to the `.sbatch` files, or pass them at submission time.
