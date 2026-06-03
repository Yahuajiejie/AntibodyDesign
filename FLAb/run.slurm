#!/bin/bash
# ══════════════════════════════════════════════════════════════════
# SLURM 作业脚本 — FLAb 抗体亲和力预测模型训练
#
# 提交方式：sbatch run.slurm
# 查看进度：tail -f logs/train_<jobid>.log
# 取消任务：scancel <jobid>
# ══════════════════════════════════════════════════════════════════

# ── 资源申请 ──────────────────────────────────────────────────────
#SBATCH --job-name=flab_train          # 任务名称（队列里显示）
#SBATCH --nodes=1                      # 1 个节点（MLP 太小，多节点没收益）
#SBATCH --ntasks=1                     # 1 个进程
#SBATCH --cpus-per-task=8             # 8 个 CPU（数据加载 + tokenizer 并行）
#SBATCH --mem=32G                      # 32GB 内存（ESM2-650M + 数据集）
#SBATCH --gres=gpu:a800:1             # 1 张 A800（ESM2-650M 只需 ~5GB 显存）
#SBATCH --time=02:00:00               # 申请 2 小时（实际跑 ~30 分钟，留 buffer）
#SBATCH --partition=GPUA800              # 根据你的集群修改分区名
#SBATCH --output=logs/train_%j.log    # 标准输出写入文件（%j = job id）
#SBATCH --error=logs/train_%j.log     # 错误输出同一文件

# ── 环境准备 ──────────────────────────────────────────────────────
mkdir -p logs                          # 确保日志目录存在

echo "========================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "Start:     $(date)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "========================================"

# 激活 conda 环境（根据你的环境名修改）
# conda activate flab
# 或者用 module：
# module load cuda/12.1 python/3.10

# 切换到项目目录（根据实际路径修改）
cd $SLURM_SUBMIT_DIR

# 确认 GPU 可用
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# ── 步骤1：提取 ESM2 Embedding（约 15~30 分钟）─────────────────────
echo ""
echo "[Step 1] 开始提取 ESM2 Embedding..."
echo "时间: $(date)"

python train.py --mode embed \
    --data_dir  data/binding \
    --cache_dir cache/embeddings \
    --output_dir results/affinity_model

echo "[Step 1] Embedding 完成：$(date)"

# ── 步骤2：训练 MLP + 消融实验（约 2~5 分钟）─────────────────────
echo ""
echo "[Step 2] 开始训练（3 种损失函数消融实验）..."
echo "时间: $(date)"

python train.py --mode train \
    --cache_dir  cache/embeddings \
    --output_dir results/affinity_model \
    --loss mse hinge ranknet

echo "[Step 2] 训练完成：$(date)"

# ── 完成 ──────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "全部完成！时间: $(date)"
echo "结果位置: results/affinity_model/summary_all_losses.csv"
echo "========================================"
