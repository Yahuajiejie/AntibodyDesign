"""
viz_stats.py — 统计结果可视化

读取训练完成后生成的 summary_all_losses.csv，绘制：
  图1: 消融实验 — 三种损失函数的 Spearman 分布（箱线图）
  图2: 各 benchmark 的 Spearman 热力图（benchmark × loss）
  图3: 预测分数 vs 真实 fitness 散点图（选取最优 benchmark）

用法（从 FLAb/ 根目录运行）：
  python visualization/viz_stats.py --summary results/affinity_model/summary_all_losses.csv
  python visualization/viz_stats.py --demo   # 生成模拟数据演示（无需真实结果）
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# 导入共用样式
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from visualization.style import apply_style, COLORS, LOSS_COLORS, LOSS_LABELS


def make_demo_data() -> pd.DataFrame:
    """
    生成模拟数据用于演示（没有真实训练结果时调用）。
    模拟 20 个 benchmark × 3 种 loss 的 Spearman 结果。
    """
    rng = np.random.default_rng(42)
    benchmarks = [f"benchmark_{i:02d}" for i in range(20)]
    rows = []
    # 模拟三种 loss 的性能差异：ranknet > hinge > mse
    base = rng.uniform(-0.2, 0.9, size=20)
    for loss, boost in [("mse", 0.0), ("hinge", 0.05), ("ranknet", 0.10)]:
        for i, name in enumerate(benchmarks):
            rows.append({
                "name": name,
                "loss": loss,
                "n": rng.integers(10, 500),
                "spearman_test": float(np.clip(base[i] + boost + rng.normal(0, 0.05), -1, 1)),
            })
    return pd.DataFrame(rows)


# ── 图1：箱线图（消融实验）─────────────────────────────────────────────────────

def plot_ablation_boxplot(df: pd.DataFrame, save_path: str = None):
    """
    三种损失函数的 Spearman 分布箱线图。

    每个箱体代表所有 benchmark 的 Spearman 分布，
    方便直观比较三种 loss 的整体效果。
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    loss_names = ["mse", "hinge", "ranknet"]
    data = [
        df[df["loss"] == loss]["spearman_test"].dropna().values
        for loss in loss_names
    ]
    labels = [LOSS_LABELS[l] for l in loss_names]
    colors = [LOSS_COLORS[l] for l in loss_names]

    # 绘制箱线图
    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,    # 填充颜色
        notch=False,
        widths=0.5,
        medianprops=dict(color="white", linewidth=2.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker="o", markersize=5, alpha=0.5),
    )
    # 给每个箱体上色
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor(color)

    # 在箱体上标注中位数值
    for i, d in enumerate(data):
        median = np.median(d)
        ax.text(i + 1, median + 0.03, f"{median:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=colors[i])

    # 添加零线（Spearman=0 表示无相关性）
    ax.axhline(0, color=COLORS["gray"], linewidth=1, linestyle="--", alpha=0.5)
    ax.text(3.45, 0.02, "random baseline", fontsize=8, color=COLORS["gray"])

    ax.set_ylabel("Spearman 相关系数（test set）")
    ax.set_title("消融实验：三种损失函数的亲和力排序效果", pad=12)
    ax.set_ylim(-1.05, 1.15)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"保存至: {save_path}")
    plt.show()


# ── 图2：热力图（benchmark × loss）────────────────────────────────────────────

def plot_heatmap(df: pd.DataFrame, save_path: str = None, top_n: int = 25):
    """
    各 benchmark 在三种 loss 下的 Spearman 热力图。

    行 = benchmark，列 = 损失函数
    颜色越深绿表示 Spearman 越高，越红表示越低（负相关）。
    只展示样本量最多的 top_n 个 benchmark。
    """
    apply_style()

    # 取样本量最多的 top_n 个 benchmark（小数据集 Spearman 不稳定）
    top_names = (
        df.groupby("name")["n"].max()
          .nlargest(top_n)
          .index.tolist()
    )
    sub = df[df["name"].isin(top_names)]

    # 转为矩阵形式：行=benchmark，列=loss
    pivot = sub.pivot(index="name", columns="loss", values="spearman_test")
    pivot = pivot[["mse", "hinge", "ranknet"]]  # 固定列顺序

    # 按 ranknet 列的 Spearman 排序，最优的在上方
    pivot = pivot.sort_values("ranknet", ascending=False)

    fig, ax = plt.subplots(figsize=(7, max(6, len(pivot) * 0.35)))

    import matplotlib.colors as mcolors
    # 自定义色图：红 → 白 → 绿
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rg", ["#f43f5e", "#ffffff", "#10b981"], N=256
    )

    im = ax.imshow(pivot.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    # 坐标轴标签
    ax.set_xticks(range(3))
    ax.set_xticklabels([LOSS_LABELS[c] for c in pivot.columns], fontsize=9)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(
        [n.replace("_kd", "").replace("_Kd", "")[:30] for n in pivot.index],
        fontsize=7.5
    )

    # 在每个格子里写数值
    for i in range(len(pivot)):
        for j in range(3):
            val = pivot.values[i, j]
            if not np.isnan(val):
                text_color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=text_color)

    # 颜色条
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman 相关系数", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(f"各 Benchmark × 损失函数 Spearman（Top {top_n}）", pad=12)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"保存至: {save_path}")
    plt.show()


# ── 图3：预测分数 vs 真实 fitness 散点图 ────────────────────────────────────────

def plot_scatter(
    scores: np.ndarray,
    fitness: np.ndarray,
    benchmark_name: str,
    loss_name: str = "ranknet",
    save_path: str = None
):
    """
    在给定 benchmark 上，用散点图展示模型预测分数与真实 fitness 的相关性。

    参数：
      scores:  模型输出的亲和力分数数组
      fitness: 对应的真实 fitness 值数组
      benchmark_name: 数据集名称（用于标题）
      loss_name: 使用的损失函数名称
    """
    from scipy.stats import spearmanr, pearsonr
    apply_style()

    fig, ax = plt.subplots(figsize=(5.5, 5))

    # 计算相关系数
    spearman_r, _ = spearmanr(scores, fitness)
    pearson_r,  _ = pearsonr(scores, fitness)

    # 散点（按 fitness 高低着色）
    sc = ax.scatter(
        fitness, scores,
        c=fitness,
        cmap="RdYlGn",          # 红（低亲和力）→ 绿（高亲和力）
        alpha=0.7,
        s=40,
        edgecolors="white",
        linewidths=0.5,
    )
    plt.colorbar(sc, ax=ax, label="真实 fitness", fraction=0.046)

    # 趋势线
    z = np.polyfit(fitness, scores, 1)
    x_line = np.linspace(fitness.min(), fitness.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line),
            color=LOSS_COLORS[loss_name], linewidth=2, alpha=0.8, label="趋势线")

    # 标注相关系数
    ax.text(0.05, 0.92,
            f"Spearman ρ = {spearman_r:.3f}\nPearson r = {pearson_r:.3f}",
            transform=ax.transAxes,
            fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=LOSS_COLORS[loss_name], alpha=0.9))

    ax.set_xlabel("真实 fitness（实验亲和力）")
    ax.set_ylabel("模型预测分数")
    short_name = benchmark_name.replace("_kd", "").replace("_Kd", "")
    ax.set_title(f"{short_name}\n{LOSS_LABELS[loss_name]}", pad=10)
    ax.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"保存至: {save_path}")
    plt.show()


# ── 主函数 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default=None,
                        help="summary_all_losses.csv 的路径")
    parser.add_argument("--demo",   action="store_true",
                        help="使用模拟数据演示（无需真实结果）")
    parser.add_argument("--outdir", default="figures",
                        help="图片输出目录")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 加载数据
    if args.demo or args.summary is None:
        print("[Demo 模式] 使用模拟数据")
        df = make_demo_data()
    else:
        df = pd.read_csv(args.summary)

    # 绘制并保存
    plot_ablation_boxplot(df, save_path=f"{args.outdir}/fig1_ablation_boxplot.pdf")
    plot_heatmap(df, save_path=f"{args.outdir}/fig2_heatmap.pdf")

    # 散点图：用模拟数据演示
    rng = np.random.default_rng(0)
    fitness = rng.uniform(5, 12, 50)
    scores  = fitness + rng.normal(0, 1.2, 50)
    plot_scatter(scores, fitness,
                 benchmark_name="hie2023efficient_CoV2_S309",
                 loss_name="ranknet",
                 save_path=f"{args.outdir}/fig3_scatter.pdf")

    print(f"\n全部图片已保存至 {args.outdir}/")


if __name__ == "__main__":
    main()
