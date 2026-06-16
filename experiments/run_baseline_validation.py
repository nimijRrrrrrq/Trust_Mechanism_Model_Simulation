# experiments/run_baseline_validation.py
"""
基线算法验证实验脚本

本脚本对 EigenTrust、PeerTrust、PageRank 三个基线算法进行系统性验证，
覆盖题目要求的四项测试内容：

  测试1 - 功能测试：
      在 10 节点小网络上手动验证算法输出的合法性：
      - 信任值无 NaN / Inf
      - 信任值在 [0, 1] 范围内
      - 信任值之和等于 1（概率分布性质）
      - 打印每个节点的具体信任值，便于人工核查

  测试2 - 准确率测试：
      在 200 节点网络上，遍历攻击比例 0%~60%，
      使用排序法（rank-based）评估恶意节点识别准确率，
      输出准确率表格并生成折线图和柱状图。

  测试3 - 效率测试：
      记录各算法在不同攻击比例下的纯计算时间（秒），
      输出时间对比表格并生成折线图。

  测试4 - 收敛测试：
      记录各算法在 20% 攻击比例下的迭代次数和每步残差，
      输出迭代次数汇总并生成对数坐标收敛曲线图。

输出文件（保存到 results/figures/）：
  baseline_accuracy.png      - 准确率折线图
  baseline_accuracy_bar.png  - 准确率柱状图
  baseline_efficiency.png    - 计算时间折线图
  baseline_convergence.png   - 收敛曲线图（对数坐标）

运行方式：
  python experiments/run_baseline_validation.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from core.network import TrustNetwork
from core.simulation import Simulation
from models.eigentrust import EigenTrust
from models.eigentrust_improved import ImprovedEigenTrust
from models.peertrust import PeerTrust
from models.pagerank import PageRankTrust
from config import RANDOM_SEED, TRANSACTION_ROUNDS, NUM_NODES, REPEAT_TIMES as CONFIG_REPEAT_TIMES, EIGENTRUST_MAX_ITER


# ─────────────────────────────────────────────────────────────────────────────
# 全局配置
# ─────────────────────────────────────────────────────────────────────────────

# 攻击比例列表（与基线实验保持一致）
ATTACK_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

# 每组参数重复次数（从 config.py 读取）
REPEAT_TIMES = CONFIG_REPEAT_TIMES

# 图表输出目录
OUTPUT_DIR = '/root/autodl-tmp/trust_simulation/results/figures'

# 参与验证的算法
ALGORITHMS = {
    'EigenTrust':      EigenTrust,
    'EigenTrust(改进)': ImprovedEigenTrust,
    'PeerTrust':       PeerTrust,
    'PageRank':        PageRankTrust,
}


# ─────────────────────────────────────────────────────────────────────────────
# 中文字体配置（Windows 环境）
# ─────────────────────────────────────────────────────────────────────────────

def _setup_chinese_font():
    """尝试加载 Windows 中文字体，失败则静默跳过"""
    candidates = ['Microsoft YaHei', 'SimHei', 'SimSun', 'FangSong']
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams['font.family'] = name
            plt.rcParams['axes.unicode_minus'] = False
            return name
    plt.rcParams['axes.unicode_minus'] = False
    return None


_setup_chinese_font()


# ─────────────────────────────────────────────────────────────────────────────
# 内部工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _seed(repeat: int):
    """同时设置 numpy 和 Python random 的随机种子，保证可复现"""
    np.random.seed(RANDOM_SEED + repeat)
    random.seed(RANDOM_SEED + repeat)


def _build_network(num_nodes: int, attack_ratio: float, repeat: int) -> TrustNetwork:
    """
    构建并运行仿真网络。
    每次调用前重置随机种子，确保不同 repeat 产生不同但可复现的网络。
    """
    _seed(repeat)
    net = TrustNetwork(num_nodes, attack_ratio)
    sim = Simulation(net)
    sim.run(TRANSACTION_ROUNDS)
    return net


def _compute(algo_class, network, track_convergence: bool = False):
    """
    统一调用各算法的 compute 接口，屏蔽返回值格式差异。

    各算法返回格式：
      EigenTrust : 始终返回 3-tuple (trust_vector, conv_history_or_None, compute_time)
      PeerTrust  : track_convergence=True  → 3-tuple (trust_vector, conv_history, compute_time)
                   track_convergence=False → 2-tuple (trust_vector, compute_time)
      PageRank   : 同 PeerTrust

    本函数统一返回: (trust_vector, conv_history_or_None, compute_time)
      - trust_vector : np.ndarray，节点信任值（已写回 node.trust_value）
      - conv_history : list[float] 或 None
      - compute_time : float，纯算法计算时间（秒）

    注意：_evaluate 依赖 compute 写回 node.trust_value 的副作用，
    调用本函数后无需再单独赋值。
    """
    kwargs = {'track_convergence': track_convergence}
    if algo_class in (EigenTrust, ImprovedEigenTrust):
        kwargs['max_iter'] = EIGENTRUST_MAX_ITER
    ret = algo_class.compute(network, **kwargs)

    # 改进版 EigenTrust 返回 EigenTrustResult dataclass
    if hasattr(ret, 'trust_vector'):
        return ret.trust_vector, ret.convergence_history, ret.compute_time

    # EigenTrust 始终返回 3-tuple；PeerTrust/PageRank 在 track_convergence=False
    # 时返回 2-tuple，True 时返回 3-tuple。
    if len(ret) == 3:
        return ret[0], ret[1], ret[2]
    else:
        # 2-tuple: (trust_vector, compute_time)
        return ret[0], None, ret[1]


def _evaluate(network) -> tuple:
    """
    排序法评估（rank-based），不依赖固定阈值。

    将节点按信任值降序排列，信任值最低的 num_malicious 个节点判为恶意，
    其余判为正常，计算 accuracy / precision / recall / f1。

    返回: (accuracy, precision, recall, f1)
    """
    n       = len(network.nodes)
    num_mal = network.num_malicious
    num_hon = network.num_honest

    if num_mal == 0:
        return 1.0, 1.0, 1.0, 1.0,1.0,1.0

    sorted_nodes = sorted(network.nodes, key=lambda x: x.trust_value, reverse=True)

    tp = fp = fn = 0
    for i, node in enumerate(sorted_nodes):
        if i < num_hon:          # 预测为正常
            if node.is_malicious:
                fn += 1
        else:                    # 预测为恶意
            if node.is_malicious:
                tp += 1
            else:
                fp += 1

    # rank-based 评估中 fp==fn 恒成立（两者都等于 num_mal - tp），
    # 因此 tn = num_hon - fp 与 num_hon - fn 等价，取前者与 fp 对称。
    tn        = num_hon - fp
    accuracy  = (tp + tn) / n
    precision = tp / (tp + fp + 1e-10)
    recall    = tp / (tp + fn + 1e-10)
    f1        = 2 * precision * recall / (precision + recall + 1e-10)

    # AUC evaluation
    y_true = []
    y_score = []

    for node in network.nodes:
        y_true.append(1 if node.is_malicious else 0)

        # trust 越低越恶意
        y_score.append(-node.trust_value)

    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    return accuracy, precision, recall, f1, roc_auc, pr_auc



def _save_figure(fig, filename: str):
    """保存图表到 OUTPUT_DIR，自动创建目录"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 测试 1：功能测试（10 节点小网络手动验证）
# ─────────────────────────────────────────────────────────────────────────────

def test_functional(num_nodes: int = 10, attack_ratio: float = 0.3) -> bool:
    """
    功能测试：在小网络上验证算法输出的基本合法性。

    检查项：
      - 无 NaN / Inf
      - 所有信任值在 [0, 1] 范围内
      - 信任值之和 ≈ 1（允许 1e-3 误差）
      - 打印每个节点的信任值，便于人工核查

    参数：
      num_nodes    : 节点数（建议 10，方便人工阅读）
      attack_ratio : 恶意节点比例

    返回：所有算法均通过时返回 True
    """
    print("\n" + "=" * 60)
    print("测试 1：功能测试（小网络手动验证）")
    print(f"  网络规模: {num_nodes} 节点  攻击比例: {attack_ratio*100:.0f}%")
    print("=" * 60)

    all_passed = True

    for algo_name, algo_class in ALGORITHMS.items():
        net = _build_network(num_nodes, attack_ratio, repeat=0)
        t_vec, _, _ = _compute(algo_class, net)

        has_nan  = bool(np.any(np.isnan(t_vec)))
        has_inf  = bool(np.any(np.isinf(t_vec)))
        t_sum    = float(np.sum(t_vec))
        in_range = bool(np.all(t_vec >= -1e-9) and np.all(t_vec <= 1 + 1e-9))
        # PeerTrust 输出的是加权平均信任值，不是概率分布，不要求 sum=1
        check_sum = algo_name != 'PeerTrust'
        sum_ok    = (abs(t_sum - 1.0) < 1e-3) if check_sum else True

        passed = not has_nan and not has_inf and in_range and sum_ok
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        sum_note = f"{t_sum:.6f}({'OK' if abs(t_sum-1.0)<1e-3 else 'N/A (PeerTrust)' if not check_sum else 'FAIL'})"
        print(f"\n  [{algo_name}] {status}")
        print(f"    NaN={has_nan}  Inf={has_inf}  "
              f"范围合法={in_range}  和={sum_note}")

        # 打印每个节点的信任值（小网络可读）
        node_lines = []
        for i, node in enumerate(net.nodes):
            label = "恶意" if node.is_malicious else "正常"
            node_lines.append(f"节点{node.id}({label})={t_vec[i]:.4f}")
        print("    " + "  ".join(node_lines))

    return all_passed


# ─────────────────────────────────────────────────────────────────────────────
# 测试 2：准确率测试（0%~60% 攻击比例）
# ─────────────────────────────────────────────────────────────────────────────

def test_accuracy(num_nodes: int = 200) -> dict:
    """
    准确率测试：在不同攻击比例下评估各算法识别恶意节点的性能。

    每个攻击比例重复 REPEAT_TIMES 次取平均，减少随机波动。
    使用排序法评估，不依赖固定阈值。

    参数：
      num_nodes : 网络节点数

    返回：
      dict，键为指标名，值为各算法在不同攻击比例下的平均值列表
      指标包括：accuracy, precision, recall, f1, roc_auc, pr_auc
    """
    print("\n" + "=" * 60)
    print("测试 2：准确率测试（不同攻击比例）")
    print(f"  网络规模: {num_nodes} 节点  重复次数: {REPEAT_TIMES}")
    print("=" * 60)

    # 存储所有指标
    results = {
        'accuracy':  {name: [] for name in ALGORITHMS},
        'precision': {name: [] for name in ALGORITHMS},
        'recall':    {name: [] for name in ALGORITHMS},
        'f1':        {name: [] for name in ALGORITHMS},
        'roc_auc':   {name: [] for name in ALGORITHMS},
        'pr_auc':    {name: [] for name in ALGORITHMS},
    }

    # 打印表头（只显示主要指标）
    header = f"  {'攻击比例':<10}" + "".join(f"{n:<14}" for n in ALGORITHMS)
    print(f"\n[准确率]")
    print(header)
    print("  " + "-" * (10 + 14 * len(ALGORITHMS)))

    for ratio in ATTACK_RATIOS:
        row = f"  {int(ratio*100):>4}%      "
        for algo_name, algo_class in ALGORITHMS.items():
            acc_sum = prec_sum = rec_sum = f1_sum = roc_sum = pr_sum = 0.0
            for rep in range(REPEAT_TIMES):
                net = _build_network(num_nodes, ratio, rep)
                _compute(algo_class, net)
                acc, prec, rec, f1, roc, pr = _evaluate(net)
                acc_sum += acc
                prec_sum += prec
                rec_sum += rec
                f1_sum += f1
                roc_sum += roc
                pr_sum += pr
            
            # 计算平均值
            avg_acc  = acc_sum / REPEAT_TIMES
            avg_prec = prec_sum / REPEAT_TIMES
            avg_rec  = rec_sum / REPEAT_TIMES
            avg_f1   = f1_sum / REPEAT_TIMES
            avg_roc  = roc_sum / REPEAT_TIMES
            avg_pr   = pr_sum / REPEAT_TIMES
            
            # 存储结果
            results['accuracy'][algo_name].append(avg_acc)
            results['precision'][algo_name].append(avg_prec)
            results['recall'][algo_name].append(avg_rec)
            results['f1'][algo_name].append(avg_f1)
            results['roc_auc'][algo_name].append(avg_roc)
            results['pr_auc'][algo_name].append(avg_pr)
            
            row += f"{avg_acc:<14.4f}"
        print(row)

    # 打印其他指标摘要
    print("\n[F1 分数]")
    print(header)
    print("  " + "-" * (10 + 14 * len(ALGORITHMS)))
    for i, ratio in enumerate(ATTACK_RATIOS):
        row = f"  {int(ratio*100):>4}%      "
        for algo_name in ALGORITHMS:
            row += f"{results['f1'][algo_name][i]:<14.4f}"
        print(row)

    print("\n[ROC AUC]")
    print(header)
    print("  " + "-" * (10 + 14 * len(ALGORITHMS)))
    for i, ratio in enumerate(ATTACK_RATIOS):
        row = f"  {int(ratio*100):>4}%      "
        for algo_name in ALGORITHMS:
            row += f"{results['roc_auc'][algo_name][i]:<14.4f}"
        print(row)

    print("\n[PR AUC]")
    print(header)
    print("  " + "-" * (10 + 14 * len(ALGORITHMS)))
    for i, ratio in enumerate(ATTACK_RATIOS):
        row = f"  {int(ratio*100):>4}%      "
        for algo_name in ALGORITHMS:
            row += f"{results['pr_auc'][algo_name][i]:<14.4f}"
        print(row)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 测试 3：效率测试（计算时间）
# ─────────────────────────────────────────────────────────────────────────────

def test_efficiency(num_nodes: int = 200) -> dict:
    """
    效率测试：记录各算法在不同攻击比例下的纯计算时间。

    计时范围仅包含算法本身（不含网络构建和仿真），
    每组重复 REPEAT_TIMES 次取平均。

    参数：
      num_nodes : 网络节点数

    返回：
      dict，键为算法名，值为各攻击比例下的中位数计算时间列表（秒）
    """
    print("\n" + "=" * 60)
    print("测试 3：效率测试（计算时间，单位：秒）")
    print(f"  网络规模: {num_nodes} 节点  重复次数: {REPEAT_TIMES}")
    print("=" * 60)

    results = {name: [] for name in ALGORITHMS}

    header = f"  {'攻击比例':<10}" + "".join(f"{n:<16}" for n in ALGORITHMS)
    print(f"\n{header}")
    print("  " + "-" * (10 + 16 * len(ALGORITHMS)))

    for ratio in ATTACK_RATIOS:
        row = f"  {int(ratio*100):>4}%      "
        for algo_name, algo_class in ALGORITHMS.items():
            samples = []
            for rep in range(REPEAT_TIMES):
                net = _build_network(num_nodes, ratio, rep)
                _, _, t = _compute(algo_class, net)
                samples.append(t)
            median = float(np.median(samples))
            results[algo_name].append(median)
            row += f"{median:<16.4f}"
        print(row)

    # 打印各算法全局中位数时间
    print("\n  各算法全局中位数计算时间：")
    for algo_name, times in results.items():
        print(f"    {algo_name:<12}: {np.median(times)*1000:.2f} ms")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 测试 4：收敛测试（迭代次数与收敛曲线）
# ─────────────────────────────────────────────────────────────────────────────

def test_convergence(num_nodes: int = 200, attack_ratio: float = 0.2) -> dict:
    """
    收敛测试：记录各算法的迭代次数和每步 L1 残差。

    取 REPEAT_TIMES 次重复的平均收敛曲线，
    用于绘制对数坐标收敛曲线图。

    参数：
      num_nodes    : 网络节点数
      attack_ratio : 固定攻击比例（默认 20%）

    返回：
      dict，键为算法名，值为平均收敛历史（list[float]）
    """
    print("\n" + "=" * 60)
    print("测试 4：收敛测试（迭代次数与收敛曲线）")
    print(f"  网络规模: {num_nodes} 节点  攻击比例: {attack_ratio*100:.0f}%")
    print("=" * 60)

    histories = {}

    for algo_name, algo_class in ALGORITHMS.items():
        all_histories = []
        for rep in range(REPEAT_TIMES):
            net = _build_network(num_nodes, attack_ratio, rep)
            _, hist, _ = _compute(algo_class, net, track_convergence=True)
            if hist and len(hist) > 0:
                all_histories.append(hist)

        if not all_histories:
            print(f"  {algo_name:<12}: 无收敛历史（算法不支持 track_convergence）")
            continue

        # 对齐长度后取平均；clamp 到 1e-8 防止 semilogy 遇到 0 崩溃
        min_len  = min(len(h) for h in all_histories)
        avg_hist = np.mean([h[:min_len] for h in all_histories], axis=0)
        avg_hist = np.maximum(avg_hist, 1e-8).tolist()
        histories[algo_name] = avg_hist

        iters = len(avg_hist)
        final = avg_hist[-1]
        print(f"  {algo_name:<12}: 迭代 {iters:>3} 次  最终残差 {final:.2e}")

    return histories


# ─────────────────────────────────────────────────────────────────────────────
# 图表生成函数
# ─────────────────────────────────────────────────────────────────────────────

def plot_accuracy(results: dict, metric_name: str = 'Accuracy', filename: str = 'baseline_accuracy.png'):
    """
    绘制指标折线图。
    X 轴：攻击比例（%），Y 轴：指标值，每条线代表一个算法。

    参数：
      results     : dict，键为算法名，值为各攻击比例下的指标值列表
      metric_name : 指标名称，用于图表标题和 Y 轴标签
      filename    : 输出文件名
    """
    from utils.visualization import _color_for
    fig, ax = plt.subplots(figsize=(8, 5))
    x       = [int(r * 100) for r in ATTACK_RATIOS]
    markers = ['o', 's', '^', 'D']

    for (name, vals), marker in zip(results.items(), markers):
        ax.plot(x, vals, marker=marker, label=name, color=_color_for(name),
                linewidth=2, markersize=6)

    ax.set_xlabel('Attack Ratio (%)')
    ax.set_ylabel(metric_name)
    ax.set_title(f'Baseline Algorithm {metric_name} Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    _save_figure(fig, filename)


def plot_accuracy_bar(results: dict):
    """
    绘制准确率分组柱状图，便于直观比较各攻击比例下的差异。
    """
    from utils.visualization import _color_for
    fig, ax = plt.subplots(figsize=(10, 5))
    x      = np.arange(len(ATTACK_RATIOS))
    names  = list(results.keys())
    n_algo = len(names)
    width  = 0.8 / n_algo
    offsets = np.linspace(-(n_algo - 1) / 2,
                           (n_algo - 1) / 2,
                           n_algo) * width

    for name, offset in zip(names, offsets):
        bars = ax.bar(x + offset, results[name], width, label=name,
                      color=_color_for(name), alpha=0.8)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(r*100)}%" for r in ATTACK_RATIOS])
    ax.set_xlabel('Attack Ratio')
    ax.set_ylabel('Accuracy')
    ax.set_title('Baseline Algorithm Accuracy Comparison (Bar Chart)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)
    _save_figure(fig, 'baseline_accuracy_bar.png')


def plot_efficiency(results: dict):
    """
    绘制计算时间折线图。
    X 轴：攻击比例（%），Y 轴：计算时间（秒）。
    """
    from utils.visualization import _color_for
    fig, ax = plt.subplots(figsize=(8, 5))
    x       = [int(r * 100) for r in ATTACK_RATIOS]
    markers = ['o', 's', '^', 'D']

    for (name, vals), marker in zip(results.items(), markers):
        ax.plot(x, vals, marker=marker, label=name, color=_color_for(name),
                linewidth=2, markersize=6)

    ax.set_xlabel('Attack Ratio (%)')
    ax.set_ylabel('Computation Time - Median (seconds)')
    ax.set_title('Baseline Algorithm Efficiency Comparison (Median)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_figure(fig, 'baseline_efficiency.png')


def plot_convergence(histories: dict):
    """
    绘制收敛曲线图（对数坐标）。
    X 轴：迭代次数，Y 轴：L1 残差（对数刻度）。
    每条线代表一个算法在 20% 攻击比例下的平均收敛过程。
    """
    from utils.visualization import _color_for
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ['o', 's', '^', 'D']

    for (name, hist), marker in zip(histories.items(), markers):
        iters = range(1, len(hist) + 1)
        ax.semilogy(iters, hist,
                    marker=marker,
                    markevery=max(1, len(hist) // 10),
                    label=name,
                    color=_color_for(name),
                    linewidth=2)

    ax.set_xlabel('Iterations')
    ax.set_ylabel('Residual (L1 norm, log scale)')
    ax.set_title('Baseline Algorithm Convergence Curve (Attack Ratio 20%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_figure(fig, 'baseline_convergence.png')


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def run_all_validations():
    """
    依次运行四项验证测试并生成图表。
    返回 True 表示功能测试全部通过。
    """
    total_start = time.time()

    print("=" * 60)
    print("基线算法验证实验")
    print("=" * 60)

    # 测试 1：功能测试
    func_ok = test_functional(num_nodes=10, attack_ratio=0.3)

    # 测试 2：准确率测试（返回多指标结果）
    all_results = test_accuracy(num_nodes=200)

    # 测试 3：效率测试
    eff_results = test_efficiency(num_nodes=200)

    # 测试 4：收敛测试
    conv_histories = test_convergence(num_nodes=200, attack_ratio=0.2)

    # 生成图表
    print("\n" + "=" * 60)
    print("生成图表")
    print("=" * 60)
    plot_accuracy(all_results['accuracy'])
    plot_accuracy_bar(all_results['accuracy'])
    plot_accuracy(all_results['f1'], metric_name='F1 Score', filename='baseline_f1.png')
    plot_accuracy(all_results['roc_auc'], metric_name='ROC AUC', filename='baseline_roc_auc.png')
    plot_accuracy(all_results['pr_auc'], metric_name='PR AUC', filename='baseline_pr_auc.png')
    plot_efficiency(eff_results)
    if conv_histories:
        plot_convergence(conv_histories)

    elapsed = time.time() - total_start
    print(f"\n验证完成，总耗时 {elapsed:.1f}s")
    print(f"图表保存在: {os.path.abspath(OUTPUT_DIR)}/")

    return func_ok


if __name__ == "__main__":
    success = run_all_validations()
    sys.exit(0 if success else 1)
