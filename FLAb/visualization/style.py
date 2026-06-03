"""
style.py — 统一绘图风格配置

所有图共用这套配置，保证论文级别的视觉一致性。
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

# ── 颜色方案 ────────────────────────────────────────────────────────────────────
# 以深蓝-紫色为主色，红色为对比色，绿色为正向指标
COLORS = {
    "primary":   "#6366f1",   # 主色（ESM2 / 模型相关）
    "secondary": "#a855f7",   # 辅色（MLP head）
    "green":     "#10b981",   # 正向指标（Spearman 高）
    "amber":     "#f59e0b",   # 警示（中等结果）
    "red":       "#f43f5e",   # 负向（Spearman 低）
    "gray":      "#6b7280",   # 中性
    "bg":        "#ffffff",   # 白底
    "grid":      "#f3f4f6",   # 网格线
}

# 三种损失函数对应的颜色（消融实验用）
LOSS_COLORS = {
    "mse":     "#f59e0b",   # 橙黄，MSE 是最基础的
    "hinge":   "#6366f1",   # 蓝紫，Hinge 是经典 SVM
    "ranknet": "#10b981",   # 绿，RankNet 是概率框架
}

LOSS_LABELS = {
    "mse":     "MSE Loss",
    "hinge":   "Pairwise Hinge",
    "ranknet": "RankNet (BT)",
}


def apply_style():
    """
    应用全局 matplotlib 样式。
    在每个绘图文件的开头调用一次即可。
    """
    mpl.rcParams.update({
        # 字体
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.labelsize":    11,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,

        # 线条
        "axes.linewidth":    1.2,
        "grid.linewidth":    0.8,
        "lines.linewidth":   2.0,

        # 背景与网格
        "axes.facecolor":    COLORS["bg"],
        "figure.facecolor":  COLORS["bg"],
        "axes.grid":         True,
        "grid.color":        COLORS["grid"],
        "grid.alpha":        1.0,

        # 去掉右侧和上侧边框（更简洁）
        "axes.spines.right": False,
        "axes.spines.top":   False,

        # 分辨率（保存时用）
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.facecolor": COLORS["bg"],
    })
