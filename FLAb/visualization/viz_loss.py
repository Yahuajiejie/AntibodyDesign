"""
viz_loss.py — 三种损失函数原理图

用数学曲线和示意图直观展示三种 loss 的差异：
  图1: 三种损失函数曲线（以分数差 Δ = score_A - score_B 为横轴）
  图2: Pairwise 训练示意图（正负样本对的构造原理）

用法：
  python visualization/viz_loss.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as FancyArrowPatch
from matplotlib.patches import FancyArrowPatch as FAP
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from visualization.style import apply_style, COLORS, LOSS_COLORS, LOSS_LABELS


# ── 图1：损失函数曲线 ───────────────────────────────────────────────────────────

def plot_loss_curves(save_path: str = None):
    """
    以 Δ = score_A - score_B 为横轴，绘制三种损失函数的值。

    直观展示：
      - 当 Δ > 0（A 分高于 B）时，三种 loss 都趋向 0
      - 当 Δ < 0（排序错误）时，三种 loss 都很大
      - 差别在于：收敛速度、曲线形状、"margin 区域"的处理方式
    """
    apply_style()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    margin = 0.1   # Hinge 的 margin 值

    # Delta 范围：从 "A 比 B 低 2 分" 到 "A 比 B 高 2 分"
    delta = np.linspace(-2, 2, 500)

    # 三种损失的计算公式
    loss_funcs = {
        "mse":     lambda d: np.maximum(0.5 - d, 0) ** 2,   # 简化的 MSE（目标 Δ=0.5）
        "hinge":   lambda d: np.maximum(margin - d, 0),      # Hinge
        "ranknet": lambda d: np.log1p(np.exp(-d)),           # softplus(-delta)
    }

    descriptions = {
        "mse":     (
            "MSE Loss",
            "假设：误差 ~ 正态分布\n目标：score_A - score_B = 0.5\n优化：最小化平方误差",
            "高斯噪声"
        ),
        "hinge":   (
            "Pairwise Hinge Loss",
            f"假设：SVM 几何间隔\n目标：score_A - score_B ≥ {margin}\n优化：最大化间隔",
            "无分布假设"
        ),
        "ranknet": (
            "RankNet Loss（Bradley-Terry）",
            "假设：误差 ~ Gumbel 分布\n目标：P(A≻B) = σ(score_A - score_B)\n优化：最大化似然",
            "Gumbel 噪声"
        ),
    }

    for ax, (loss_name, loss_fn) in zip(axes, loss_funcs.items()):
        color = LOSS_COLORS[loss_name]
        y = loss_fn(delta)

        # 绘制损失曲线
        ax.plot(delta, y, color=color, linewidth=3, zorder=3)

        # 填充"损失区域"（Δ < 某阈值时才有损失）
        if loss_name == "hinge":
            # Hinge：Δ < margin 才有损失
            mask = delta < margin
            ax.fill_between(delta[mask], y[mask], alpha=0.15, color=color)
            # 标注 margin
            ax.axvline(margin, color=color, linewidth=1.2, linestyle="--", alpha=0.7)
            ax.text(margin + 0.05, 0.8, f"margin={margin}", fontsize=8, color=color)
        else:
            ax.fill_between(delta, y, alpha=0.12, color=color)

        # 标注"正确排序区域"和"错误排序区域"
        ax.axvline(0, color=COLORS["gray"], linewidth=1, linestyle=":", alpha=0.6)
        ax.text( 0.8, ax.get_ylim()[1] * 0.88 if ax.get_ylim()[1] > 0 else 0.9,
                "排序正确 ✓", fontsize=8, color="#15803d", ha="center")
        ax.text(-0.8, 0.9, "排序错误 ✗", fontsize=8, color="#be123c", ha="center",
                transform=ax.get_xaxis_transform())

        title, desc, assumption = descriptions[loss_name]
        ax.set_title(title, fontsize=11, fontweight="bold", color=color, pad=8)
        ax.set_xlabel("Δ = score_A − score_B", fontsize=10)

        # 文字说明框
        ax.text(0.97, 0.97, desc, transform=ax.transAxes,
                fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=color, alpha=0.85))

        # 假设标签
        ax.text(0.03, 0.97, f"统计假设\n{assumption}", transform=ax.transAxes,
                fontsize=8, va="top", ha="left", color=color, alpha=0.8)

        ax.set_xlim(-2, 2)
        ax.set_ylim(-0.05, 2.2)

    axes[0].set_ylabel("Loss 值", fontsize=10)
    fig.suptitle("三种亲和力排序损失函数原理对比", fontsize=13, fontweight="bold", y=1.02)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"保存至: {save_path}")
    plt.show()


# ── 图2：Pairwise 训练示意图 ────────────────────────────────────────────────────

def plot_pairwise_concept(save_path: str = None):
    """
    可视化 Pairwise 训练的核心思想：
      - 左侧：同一抗原的多条抗体变体，按真实 Kd 排列
      - 右侧：模型为每条序列输出分数，应与 Kd 排序一致
      - 中间：箭头表示"正确对"（高亲和力 > 低亲和力）

    直观解释为什么用 pairwise 而不是直接回归 Kd 绝对值。
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # 五条抗体变体，按真实 Kd 排序（nM，越小结合越强）
    antibodies = [
        {"name": "Ab-1", "kd": 0.8,  "fitness": 9.1, "score": 8.7},
        {"name": "Ab-2", "kd": 2.1,  "fitness": 8.7, "score": 8.2},
        {"name": "Ab-3", "kd": 5.4,  "fitness": 8.3, "score": 8.5},   # score 有点排错
        {"name": "Ab-4", "kd": 18.0, "fitness": 7.7, "score": 7.5},
        {"name": "Ab-5", "kd": 85.0, "fitness": 7.1, "score": 6.9},
    ]

    # 左侧：真实 Kd 排序
    ax.text(1.5, 7.6, "真实亲和力（实验 Kd）", fontsize=11, fontweight="bold",
            ha="center", color=COLORS["gray"])
    for i, ab in enumerate(antibodies):
        y = 6.8 - i * 1.2
        # 序列方块
        color = plt.cm.RdYlGn(1 - i / 4)
        rect = plt.Rectangle((0.3, y - 0.25), 2.3, 0.55,
                              facecolor=color, alpha=0.8, edgecolor="white", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(1.45, y + 0.06, f"{ab['name']}  Kd = {ab['kd']} nM",
                ha="center", va="center", fontsize=9.5, fontweight="500")

    # Kd 轴标注
    ax.annotate("", xy=(0.3, 2.0), xytext=(0.3, 7.0),
                arrowprops=dict(arrowstyle="->", color=COLORS["gray"], lw=1.5))
    ax.text(0.12, 4.5, "Kd ↓\n更强", fontsize=8, ha="center", color=COLORS["gray"])

    # 中间：Pairwise 箭头（"A 比 B 亲和力更高"的训练对）
    ax.text(5.0, 7.6, "训练对构造\nfitness_A > fitness_B", fontsize=10,
            ha="center", fontweight="bold", color=COLORS["primary"])
    # 示例：Ab-1 vs Ab-3（Ab-1 Kd 更小 → 亲和力更高）
    for (ia, ib), alpha in [((0, 2), 0.9), ((0, 4), 0.6), ((1, 3), 0.7)]:
        ya = 6.8 - ia * 1.2
        yb = 6.8 - ib * 1.2
        ax.annotate("", xy=(4.5, (ya + yb) / 2 + 0.15), xytext=(2.65, ya),
                    arrowprops=dict(arrowstyle="->", color=COLORS["primary"],
                                   alpha=alpha, lw=1.5,
                                   connectionstyle="arc3,rad=0.25"))
        ax.annotate("", xy=(4.5, (ya + yb) / 2 - 0.15), xytext=(2.65, yb),
                    arrowprops=dict(arrowstyle="->", color=COLORS["red"],
                                   alpha=alpha, lw=1.5,
                                   connectionstyle="arc3,rad=-0.25"))

    ax.text(4.5, 4.0, "Loss: 要求\nscore_A > score_B + margin",
            fontsize=8.5, ha="center", color=COLORS["primary"],
            bbox=dict(boxstyle="round", fc="white", ec=COLORS["primary"], alpha=0.9))

    # 右侧：模型预测分数
    ax.text(8.2, 7.6, "模型预测分数", fontsize=11, fontweight="bold",
            ha="center", color=COLORS["secondary"])
    for i, ab in enumerate(antibodies):
        y = 6.8 - i * 1.2
        # 分数条形
        score_width = ab["score"] / 10 * 1.8
        bar_color = LOSS_COLORS["ranknet"] if ab["score"] >= ab["fitness"] - 0.3 else LOSS_COLORS["mse"]
        rect = plt.Rectangle((6.8, y - 0.2), score_width, 0.4,
                              facecolor=bar_color, alpha=0.7, edgecolor="white")
        ax.add_patch(rect)
        ax.text(6.8 + score_width + 0.1, y + 0.04, f"{ab['score']:.1f}",
                va="center", fontsize=9)

    # 连接线：真实排序 → 预测分数
    for i, ab in enumerate(antibodies):
        y = 6.8 - i * 1.2
        ax.plot([2.65, 6.8], [y, y], color=COLORS["grid"], linewidth=1, linestyle=":")

    # Ab-3 的错误标注（score 排序和 Kd 不完全一致）
    ax.text(8.5, 6.8 - 2 * 1.2, "← 排序偏差", fontsize=8, color=COLORS["amber"])

    # 底部说明
    ax.text(5.0, 0.5,
            "Pairwise Loss 只学习「谁比谁强」，不关心 Kd 绝对值，天然适应不同数据集的量纲差异",
            fontsize=9, ha="center", style="italic", color=COLORS["gray"])

    fig.suptitle("Pairwise 排序训练：从抗体变体中学习亲和力排序规律",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"保存至: {save_path}")
    plt.show()


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    plot_loss_curves(save_path="figures/fig4_loss_curves.pdf")
    plot_pairwise_concept(save_path="figures/fig5_pairwise_concept.pdf")
