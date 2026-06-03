#!/bin/bash
# ══════════════════════════════════════════════════════════════════
# SLURM 作业脚本 — ESM2 零样本亲和力打分
#
# 原理：逐 token mask，计算 pseudo-perplexity 作为亲和力代理
# 每条序列需要约 120 次 ESM2 forward pass，比训练慢约 120 倍
# 仅用于建立 baseline，对比有监督方法的提升幅度
#
# 提交：sbatch run_zeroshot.slurm
# ══════════════════════════════════════════════════════════════════

#SBATCH --job-name=flab_zeroshot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a800:1             # 1 张 A800 够用
#SBATCH --time=08:00:00               # 申请 8 小时（实际约 5-6 小时）
#SBATCH --partition=GPUA800            # 根据你的集群修改
#SBATCH --output=logs/zeroshot_%j.log
#SBATCH --error=logs/zeroshot_%j.log

mkdir -p logs

source activate esm2
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONIOENCODING=utf-8

echo "========================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "Start:     $(date)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "========================================"

cd $SLURM_SUBMIT_DIR

# 跑 ESM2-650M 和 ESM2-3B 两个尺寸，看 scaling 效果
# 默认跳过 >5000 条的大数据集（--max_size 5000）
echo ""
echo "[Step 1] ESM2-650M 零样本打分..."
python run_scoring_batch.py \
    --model esm2_650M \
    --data_dir data/binding \
    --max_size 5000

echo "[Step 1] 完成：$(date)"

echo ""
echo "[Step 2] ESM2-3B 零样本打分..."
python run_scoring_batch.py \
    --model esm2_3B \
    --data_dir data/binding \
    --max_size 5000

echo "[Step 2] 完成：$(date)"

echo ""
echo "========================================"
echo "全部完成：$(date)"
echo "结果位置: score_batch/esm2_650M/summary.csv"
echo "         score_batch/esm2_3B/summary.csv"
echo "========================================"
