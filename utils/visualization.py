# utils/visualization.py
"""
Visualization utilities - Unified wrapper for experiment plots

Supported plot types:
1. Efficiency comparison (runtime + iterations)
2. Attack test comparison (grouped bar chart)
3. Convergence curve
4. Improvement comparison (dual line chart)

Style guidelines:
- Chinese font: SimHei / Microsoft YaHei
- English font: Times New Roman
- Clear legends, light grid lines (alpha=0.3)
- Save PNG (300 DPI)
"""

import os
import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams, font_manager

    def _setup_chinese_font():
        """显式加载可用中文字体，避免 matplotlib 回退到不含中文字形的 DejaVu。"""
        font_paths = [
            '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
            '/root/.trae-cn-server/bin/stable-2a1926816d08d71538a3de294bf301b26ac911e0-debian10/extensions/ai-completion/resource/aiserver/resources/font/HeiTi.ttf',
            '/root/.trae-cn-server/bin/stable-18a1ac5cc57753e4f248b4095f404def974ec17b-debian10/extensions/ai-completion/resource/aiserver/resources/font/HeiTi.ttf',
            '/root/.trae-cn-server/bin/stable-f82b86c16a5840826df774b852a21b31c4a68a60-debian10/extensions/ai-completion/resource/aiserver/resources/font/HeiTi.ttf',
            '/root/.trae-cn-server/bin/stable-73aef0ec203906061850122b18258e8f2a85744a-debian10/extensions/ai-completion/resource/aiserver/resources/font/HeiTi.ttf',
        ]
        loaded_fonts = []
        for font_path in font_paths:
            if os.path.exists(font_path):
                font_manager.fontManager.addfont(font_path)
                loaded_fonts.append(font_manager.FontProperties(fname=font_path).get_name())
        fallback_fonts = ['WenQuanYi Zen Hei', 'SimHei', 'Microsoft YaHei', 'SimSun', 'Noto Sans CJK SC', 'DejaVu Sans']
        rcParams['font.sans-serif'] = loaded_fonts + fallback_fonts
        rcParams['font.family'] = 'sans-serif'

    _setup_chinese_font()
    rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    rcParams['axes.unicode_minus'] = False

    rcParams['figure.dpi'] = 100
    rcParams['savefig.dpi'] = 300
    rcParams['grid.alpha'] = 0.3
    rcParams['legend.framealpha'] = 0.8

except ImportError:
    plt = None
    print("Warning: matplotlib not installed, visualization disabled")


# 固定 8 色调色板（图表统一配色，跨图表保持一致）
PALETTE = [
    '#f57c6e',  # 珊瑚红
    '#f2b56f',  # 橙黄
    '#fae69e',  # 浅黄
    '#84c3b7',  # 青绿
    '#88d8db',  # 青色
    '#71b7ed',  # 蓝色
    '#b8aeeb',  # 淡紫
    '#f2a7da',  # 粉色
]

# 固定算法配色：每个算法在所有图表中使用同一颜色
# 红色 (f57c6e) 留给本文提出的改进算法，便于在对比图中突出
ALGO_COLORS = {
    'EigenTrust':         '#71b7ed',  # 蓝色
    'EigenTrust(改进)':   '#f57c6e',  # 珊瑚红（本文方法）
    'ImprovedEigenTrust': '#f57c6e',  # 兼容英文标签
    'PeerTrust':          '#84c3b7',  # 青绿
    'PageRank':           '#f2b56f',  # 橙黄
}


def _color_for(name, idx=0):
    """根据算法名称返回固定颜色；未知名称回退到 PALETTE 循环。"""
    return ALGO_COLORS.get(name, PALETTE[idx % len(PALETTE)])


def ensure_dir(path):
    """Ensure directory exists"""
    os.makedirs(path, exist_ok=True)


def plot_efficiency_comparison(results, output_path):
    """
    Plot efficiency comparison (dual bar chart: runtime + iterations)

    Args:
        results: list of dict, each contains {'algorithm', 'time', 'iterations'}
        output_path: str, output directory path
    """
    if plt is None:
        print("Warning: matplotlib not installed, skipping plot")
        return

    ensure_dir(output_path)

    algorithms = [r['algorithm'] for r in results]
    times = [r['time'] for r in results]
    iterations = [r['iterations'] for r in results]

    # 两块面板分别用调色板中的两种颜色，整体仍与全图风格一致
    time_color = PALETTE[5]        # #71b7ed 蓝
    iter_color = PALETTE[0]        # #f57c6e 珊瑚红

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bars1 = ax1.bar(algorithms, times, color=time_color, alpha=0.8)
    ax1.set_xlabel('Algorithm', fontsize=12)
    ax1.set_ylabel('Runtime (seconds)', fontsize=12)
    ax1.set_title('Algorithm Runtime Comparison', fontsize=14)
    ax1.grid(True, axis='y')
    ax1.tick_params(axis='x', labelsize=10)

    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=10)

    bars2 = ax2.bar(algorithms, iterations, color=iter_color, alpha=0.8)
    ax2.set_xlabel('Algorithm', fontsize=12)
    ax2.set_ylabel('Iterations', fontsize=12)
    ax2.set_title('Algorithm Iterations Comparison', fontsize=14)
    ax2.grid(True, axis='y')
    ax2.tick_params(axis='x', labelsize=10)

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    png_path = os.path.join(output_path, 'efficiency_comparison.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    plt.close()
    print(f"Efficiency comparison saved: {png_path}")


def plot_attack_comparison(results, output_path, attack_type='Sybil', attack_params=None, metric_name='ROC-AUC'):
    """
    Plot attack test comparison (grouped bar chart)

    Args:
        results:       dict, format {'EigenTrust': [...], 'ImprovedEigenTrust': [...], ...}
        output_path:   str, output directory path
        attack_type:   str, attack type name (for title), 'Sybil' or 'Whitewashing'
        attack_params: list of actual attack parameter values used in the experiment.
        metric_name:   str, metric label shown on Y axis and file name.
    """
    if plt is None:
        print("Warning: matplotlib not installed, skipping plot")
        return

    ensure_dir(output_path)

    algorithms = list(results.keys())
    num_points = len(next(iter(results.values())))

    # 根据传入的真实攻击参数生成 x 轴标签
    if attack_params is not None and len(attack_params) == num_points:
        if attack_type.lower() == 'whitewashing':
            # 白洗率以百分比显示
            x_labels = [f'{int(round(v * 100))}%' for v in attack_params]
            x_title  = 'Whitewash Rate'
        else:
            # Sybil 团伙规模显示整数
            x_labels = [str(v) for v in attack_params]
            x_title  = 'Sybil Group Size'
    else:
        # 兜底：顺序编号
        x_labels = [str(i + 1) for i in range(num_points)]
        x_title  = 'Attack Parameter'

    x     = np.arange(len(x_labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    # 使用固定算法配色，未知算法按 PALETTE 顺序回退
    for i, (algo_name, values) in enumerate(results.items()):
        bars = ax.bar(x + i * width, values, width=width, label=algo_name,
                      color=_color_for(algo_name, i), alpha=0.8)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel(x_title, fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f'Algorithm Performance Under {attack_type} Attack ({metric_name})', fontsize=14)
    ax.set_xticks(x + width * (len(algorithms) - 1) / 2)
    ax.set_xticklabels(x_labels)
    ax.legend(fontsize=12)
    ax.grid(True, axis='y')
    ax.set_ylim(0, 1)

    plt.tight_layout()

    safe_metric_name = metric_name.lower().replace('-', '_').replace(' ', '_')
    png_path = os.path.join(output_path, f'attack_{attack_type.lower()}_{safe_metric_name}_comparison.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{attack_type} attack comparison saved: {png_path}")


def plot_convergence(residuals, labels=None, output_path='results/figures'):
    """
    Plot convergence curve

    Args:
        residuals: list of lists or single list, residual history data
        labels: list of str, label for each curve
        output_path: str, output directory path
    """
    if plt is None:
        print("Warning: matplotlib not installed, skipping plot")
        return

    ensure_dir(output_path)

    fig, ax = plt.subplots(figsize=(10, 6))

    if not isinstance(residuals[0], list):
        residuals = [residuals]

    if labels is None:
        labels = [f'Algorithm {i+1}' for i in range(len(residuals))]

    # 使用固定算法配色，未知算法按 PALETTE 顺序回退
    for i, (res, label) in enumerate(zip(residuals, labels)):
        ax.plot(range(1, len(res)+1), res, label=label,
                color=_color_for(label, i), linewidth=2)

    ax.set_xlabel('Iterations', fontsize=12)
    ax.set_ylabel('Residual (L1 norm)', fontsize=12)
    ax.set_title('Algorithm Convergence Curve', fontsize=14)
    ax.set_yscale('log')
    ax.legend(fontsize=12)
    ax.grid(True)

    plt.tight_layout()

    png_path = os.path.join(output_path, 'convergence_plot.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    plt.close()
    print(f"Convergence curve saved: {png_path}")


def plot_improvement_comparison(baseline, improved, output_path='results/figures'):
    """
    Plot improvement comparison (dual line chart)

    Args:
        baseline: dict, contains {'name', 'accuracy', 'f1', 'efficiency'} or residuals list
        improved: dict, same as above
        output_path: str, output directory path
    """
    if plt is None:
        print("Warning: matplotlib not installed, skipping plot")
        return

    ensure_dir(output_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    if 'residuals' in baseline and 'residuals' in improved:
        ax1.plot(range(1, len(baseline['residuals'])+1), baseline['residuals'],
                 label=baseline['name'], color=_color_for(baseline['name']), linewidth=2, linestyle='--')
        ax1.plot(range(1, len(improved['residuals'])+1), improved['residuals'],
                 label=improved['name'], color=_color_for(improved['name']), linewidth=2)
        ax1.set_xlabel('Iterations', fontsize=12)
        ax1.set_ylabel('Residual (L1 norm)', fontsize=12)
        ax1.set_title('Convergence Speed Comparison', fontsize=14)
        ax1.set_yscale('log')
        ax1.legend(fontsize=12)
        ax1.grid(True)

    if 'accuracy' in baseline and 'accuracy' in improved:
        labels = ['Accuracy', 'F1 Score']
        baseline_vals = [baseline['accuracy'], baseline.get('f1', 0)]
        improved_vals = [improved['accuracy'], improved.get('f1', 0)]

        x = np.arange(len(labels))
        width = 0.35

        bars1 = ax2.bar(x - width/2, baseline_vals, width,
                        label=baseline['name'], color=_color_for(baseline['name']), alpha=0.8)
        bars2 = ax2.bar(x + width/2, improved_vals, width,
                        label=improved['name'], color=_color_for(improved['name']), alpha=0.8)

        ax2.set_xlabel('Metrics', fontsize=12)
        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_title('Performance Metrics Comparison', fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.legend(fontsize=12)
        ax2.grid(True, axis='y')
        ax2.set_ylim(0, 1)

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    png_path = os.path.join(output_path, 'improvement_comparison.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    plt.close()
    print(f"Improvement comparison saved: {png_path}")


def plot_accuracy_comparison(results, output_path='results/figures'):
    """
    Plot accuracy comparison (grouped bar chart)

    Args:
        results: dict, format {'EigenTrust': [...], 'PeerTrust': [...], 'PageRank': [...]}
        output_path: str, output directory path
    """
    if plt is None:
        print("Warning: matplotlib not installed, skipping plot")
        return

    ensure_dir(output_path)

    algorithms = list(results.keys())
    num_points = len(next(iter(results.values())))
    from config import ATTACK_RATIOS
    # 防御性处理：若 results 长度与 ATTACK_RATIOS 不一致，按下标生成兜底标签
    if len(ATTACK_RATIOS) == num_points:
        x_labels = [f'{int(r * 100)}%' for r in ATTACK_RATIOS]
    else:
        x_labels = [f'{int(i*100/num_points)}%' for i in range(num_points)]
    x = np.arange(len(x_labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (algo_name, values) in enumerate(results.items()):
        bars = ax.bar(x + i*width, values, width=width, label=algo_name,
                      color=_color_for(algo_name, i), alpha=0.8)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Attack Ratio', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Algorithm Accuracy Under Different Attack Ratios', fontsize=14)
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(x_labels)
    ax.legend(fontsize=12)
    ax.grid(True, axis='y')
    ax.set_ylim(0.5, 1.0)

    plt.tight_layout()

    png_path = os.path.join(output_path, 'accuracy_comparison.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    plt.close()
    print(f"Accuracy comparison saved: {png_path}")


def plot_f1_comparison(results, output_path='results/figures'):
    """
    Plot F1 score comparison (grouped bar chart)

    Args:
        results: dict, format {'EigenTrust': [...], 'PeerTrust': [...], 'PageRank': [...]}
        output_path: str, output directory path
    """
    if plt is None:
        print("Warning: matplotlib not installed, skipping plot")
        return

    ensure_dir(output_path)

    algorithms = list(results.keys())
    num_points = len(next(iter(results.values())))
    from config import ATTACK_RATIOS
    # 防御性处理：若 results 长度与 ATTACK_RATIOS 不一致，按下标生成兜底标签
    if len(ATTACK_RATIOS) == num_points:
        x_labels = [f'{int(r * 100)}%' for r in ATTACK_RATIOS]
    else:
        x_labels = [f'{int(i*100/num_points)}%' for i in range(num_points)]
    x = np.arange(len(x_labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (algo_name, values) in enumerate(results.items()):
        bars = ax.bar(x + i*width, values, width=width, label=algo_name,
                      color=_color_for(algo_name, i), alpha=0.8)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Attack Ratio', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Algorithm F1 Score Under Different Attack Ratios', fontsize=14)
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(x_labels)
    ax.legend(fontsize=12)
    ax.grid(True, axis='y')
    ax.set_ylim(0, 1.0)

    plt.tight_layout()

    png_path = os.path.join(output_path, 'f1_comparison.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    plt.close()
    print(f"F1 comparison saved: {png_path}")


def plot_ablation_comparison(ablation_results, output_path='results/figures'):
    """
    Plot ablation study comparison (line chart over attack ratios).

    Args:
        ablation_results: dict mapping variant name -> list of accuracies per attack ratio
        output_path: str, output file path (png)
    """
    if plt is None:
        print("Warning: matplotlib not installed, skipping plot")
        return

    output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else '.'
    ensure_dir(output_dir)

    from config import ATTACK_RATIOS
    x = [int(r * 100) for r in ATTACK_RATIOS]

    # 消融变体不是固定算法，使用 8 色调色板循环
    markers = ['o', 's', '^', 'D']
    linestyles = ['--', '-.', ':', '-']

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, (name, accs) in enumerate(ablation_results.items()):
        ax.plot(x, accs,
                label=name,
                color=PALETTE[idx % len(PALETTE)],
                marker=markers[idx % len(markers)],
                linestyle=linestyles[idx % len(linestyles)],
                linewidth=2, markersize=10,
                markeredgecolor='white', markeredgewidth=0.8)

    ax.set_xlabel('攻击节点比例 (%)', fontsize=12)
    ax.set_ylabel('准确率', fontsize=12)
    ax.set_title('消融实验 - 各变体准确率对比', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    # Y 轴按数据动态缩放：上限 1.05，下限略低于最小值留出留白
    all_vals = [v for accs in ablation_results.values() for v in accs]
    if all_vals:
        ymin = min(all_vals)
        ax.set_ylim(max(0.0, ymin - 0.05), 1.05)
    else:
        ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Ablation comparison saved: {output_path}")


def plot_ablation_contribution(contribution, output_path='results/figures'):
    """
    Plot ablation contribution as grouped bar chart over attack ratios.

    7 根柱：4 个模块独立贡献 + 协同效应 + 总提升(关TD) + 完整版TD边际
    """
    if plt is None:
        print("Warning: matplotlib not installed, skipping plot")
        return

    output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else '.'
    ensure_dir(output_dir)

    ratios = [int(r * 100) for r in contribution['attack_ratios']]
    x = np.arange(len(ratios))
    width = 0.12

    td   = contribution['time_decay_contribution']
    cf   = contribution['confidence_contribution']
    sy   = contribution['sybil_defense_contribution']
    ap   = contribution['adaptive_pretrust_contribution']
    syn  = contribution['synergy_effect']
    tot  = contribution['total_improvement']
    marg = contribution.get('time_decay_marginal_in_full',
                            [0.0] * len(ratios))  # 兼容老 contribution

    fig, ax = plt.subplots(figsize=(16, 6))

    # 7 个模块柱分别使用调色板中的 7 种颜色，保持风格统一
    # 顺序：PALETTE[0..6]，最后一个 PALETTE[7] 备用
    bars_td  = ax.bar(x - 3.0*width, td,   width, label='时间衰减贡献',     color=PALETTE[0], alpha=0.85)
    bars_cf  = ax.bar(x - 2.0*width, cf,   width, label='置信度折扣贡献',   color=PALETTE[1], alpha=0.85)
    bars_sy  = ax.bar(x - 1.0*width, sy,   width, label='Sybil 防御贡献',   color=PALETTE[3], alpha=0.85)
    bars_ap  = ax.bar(x + 0.0*width, ap,   width, label='自适应预信任贡献', color=PALETTE[6], alpha=0.85)
    bars_syn = ax.bar(x + 1.0*width, syn,  width, label='协同效应',         color=PALETTE[4], alpha=0.85)
    bars_tot = ax.bar(x + 2.0*width, tot,  width, label='总提升(关TD)',     color=PALETTE[7], alpha=0.85)
    # ⭐ 反例消融核心证据：完整版中加入时间衰减的边际贡献
    bars_marg = ax.bar(x + 3.0*width, marg, width, label='完整版TD边际',     color=PALETTE[2], alpha=0.85)

    # 在每根柱顶标注数值（保留 2 位小数，允许负值显示在 0 线下）
    for bars in (bars_td, bars_cf, bars_sy, bars_ap, bars_syn, bars_tot, bars_marg):
        for bar in bars:
            h = bar.get_height()
            offset = 0.3 if h >= 0 else -0.3
            va = 'bottom' if h >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                    f'{h:+.2f}', ha='center', va=va, fontsize=7)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('攻击节点比例 (%)', fontsize=12)
    ax.set_ylabel('准确率变化 (%)', fontsize=12)
    ax.set_title('消融实验 - 各模块贡献度分析 (4 模块 + 完整版TD边际)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{r}%' for r in ratios])
    ax.legend(fontsize=9, ncol=4)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Ablation contribution saved: {output_path}")
