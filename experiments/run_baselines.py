# experiments/run_baselines.py
"""
基线算法综合实验 - P2P 仿真 + Epinions 大数据集
为 main.py 提供以下接口：
  run_p2p_experiment(verbose)      -> {algo_name: [acc_per_ratio]}
  run_epinions_experiment(verbose) -> {algo_name: [time_seconds]}
  print_results_table(results)
  print_efficiency_table()
全局缓存变量：
  _experiment_results, _convergence_results, _metric_results, _efficiency_results
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import random
import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from core.network import TrustNetwork
from core.simulation import Simulation
from models.eigentrust import EigenTrust
from models.eigentrust_improved import ImprovedEigenTrust
from models.peertrust import PeerTrust
from models.pagerank import PageRankTrust
from config import (
    RANDOM_SEED, TRANSACTION_ROUNDS, NUM_NODES,
    REPEAT_TIMES, EIGENTRUST_MAX_ITER, ATTACK_RATIOS
)

# ─────────────────────────────────────────────────────────────
# 全局缓存（被 main.py 直接导入访问）
# ─────────────────────────────────────────────────────────────
_experiment_results = {}   # {algo_name: [acc per attack_ratio]}
_convergence_results = {}  # {algo_name: [convergence_history (first repeat)]}
_metric_results = {}       # {algo_name: {'f1': [...], 'roc_auc': [...], 'pr_auc': [...]}}
_efficiency_results = {}   # {algo_name: [median_time per attack_ratio]}

# 参与对比的算法
_ALGORITHMS = {
    'EigenTrust':       EigenTrust,
    'EigenTrust(改进)': ImprovedEigenTrust,
    'PeerTrust':        PeerTrust,
    'PageRank':         PageRankTrust,
}


# ─────────────────────────────────────────────────────────────
# 内部工具函数
# ─────────────────────────────────────────────────────────────

def _seed(repeat: int):
    np.random.seed(RANDOM_SEED + repeat)
    random.seed(RANDOM_SEED + repeat)


def _build_network(num_nodes: int, attack_ratio: float, repeat: int) -> TrustNetwork:
    _seed(repeat)
    net = TrustNetwork(num_nodes, attack_ratio)
    Simulation(net).run(TRANSACTION_ROUNDS)
    return net


def _compute(algo_class, network, track_convergence: bool = False):
    """
    统一调用接口，屏蔽各算法返回值差异。
    返回: (trust_vector, conv_history_or_None, compute_time)
    """
    kwargs = {'track_convergence': track_convergence}
    if algo_class in (EigenTrust, ImprovedEigenTrust):
        kwargs['max_iter'] = EIGENTRUST_MAX_ITER
    # [TIME_DECAY_DISABLED] P2P 主实验统一使用 ImprovedEigenTrust 调强参数
    if algo_class is ImprovedEigenTrust:
        kwargs.update({
            'adaptive_trust_rate': 0.2,
            'min_transactions': 5,
            'sybil_internal_ratio': 0.5,
            'sybil_penalty_factor': 0.1,
        })

    ret = algo_class.compute(network, **kwargs)

    # ImprovedEigenTrust 返回 EigenTrustResult dataclass
    if hasattr(ret, 'trust_vector'):
        return ret.trust_vector, ret.convergence_history, ret.compute_time

    # EigenTrust 始终返回 3-tuple
    # PeerTrust / PageRank: track_convergence=True → 3-tuple, False → 2-tuple
    if len(ret) == 3:
        return ret[0], ret[1], ret[2]
    return ret[0], None, ret[1]


def _evaluate(network):
    """
    排序法评估（rank-based）。
    返回: (accuracy, precision, recall, f1, roc_auc, pr_auc)
    """
    n = len(network.nodes)
    num_mal = network.num_malicious
    num_hon = network.num_honest

    if num_mal == 0:
        return 1.0, 1.0, 1.0, 1.0, 1.0, 1.0

    sorted_nodes = sorted(network.nodes, key=lambda x: x.trust_value, reverse=True)

    tp = fp = fn = 0
    for i, node in enumerate(sorted_nodes):
        if i < num_hon:
            if node.is_malicious:
                fn += 1
        else:
            if node.is_malicious:
                tp += 1
            else:
                fp += 1

    tn = num_hon - fp
    accuracy  = (tp + tn) / n
    precision = tp / (tp + fp + 1e-10)
    recall    = tp / (tp + fn + 1e-10)
    f1        = 2 * precision * recall / (precision + recall + 1e-10)

    y_true  = [1 if nd.is_malicious else 0 for nd in network.nodes]
    y_score = [-nd.trust_value for nd in network.nodes]

    try:
        roc_auc = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else 0.5
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = average_precision_score(y_true, y_score)
    except Exception:
        pr_auc = 0.0

    return accuracy, precision, recall, f1, roc_auc, pr_auc


# ─────────────────────────────────────────────────────────────
# run_p2p_experiment
# ─────────────────────────────────────────────────────────────

def run_p2p_experiment(verbose: bool = True) -> dict:
    """
    在 200 节点 P2P 网络上，遍历攻击比例，对4个算法进行仿真→计算→评估。
    写入全局缓存，返回 {algo_name: [acc_per_ratio]} 字典。
    """
    global _experiment_results, _convergence_results, _metric_results, _efficiency_results

    # 清空缓存
    for d in (_experiment_results, _convergence_results, _metric_results, _efficiency_results):
        d.clear()

    for algo_name in _ALGORITHMS:
        _experiment_results[algo_name] = []
        _convergence_results[algo_name] = []
        _metric_results[algo_name] = {'f1': [], 'roc_auc': [], 'pr_auc': []}
        _efficiency_results[algo_name] = []

    if verbose:
        print(f"\n  算法: {list(_ALGORITHMS.keys())}")
        print(f"  攻击比例: {ATTACK_RATIOS}")
        print(f"  每组重复: {REPEAT_TIMES} 次\n")

    header = f"  {'攻击比例':<10}" + "".join(f"{n:<16}" for n in _ALGORITHMS)
    if verbose:
        print(header)
        print("  " + "-" * (10 + 16 * len(_ALGORITHMS)))

    for ratio in ATTACK_RATIOS:
        row = f"  {int(ratio * 100):>4}%      "

        for algo_name, algo_class in _ALGORITHMS.items():
            acc_sum = f1_sum = roc_sum = pr_sum = 0.0
            time_samples = []
            conv_first = None

            for rep in range(REPEAT_TIMES):
                try:
                    net = _build_network(NUM_NODES, ratio, rep)
                    track = (rep == 0)
                    t_vec, conv_hist, elapsed = _compute(algo_class, net, track_convergence=track)

                    time_samples.append(elapsed)
                    if track and conv_hist:
                        conv_first = conv_hist

                    acc, _, _, f1, roc, pr = _evaluate(net)
                    acc_sum += acc
                    f1_sum  += f1
                    roc_sum += roc
                    pr_sum  += pr
                except Exception as e:
                    if verbose:
                        print(f"\n    警告: {algo_name} @ ratio={ratio} rep={rep} 失败: {e}")
                    time_samples.append(0.0)
                    acc_sum += 0.5
                    f1_sum  += 0.0
                    roc_sum += 0.5
                    pr_sum  += 0.0

            avg_acc = acc_sum / REPEAT_TIMES
            avg_f1  = f1_sum  / REPEAT_TIMES
            avg_roc = roc_sum / REPEAT_TIMES
            avg_pr  = pr_sum  / REPEAT_TIMES
            med_t   = float(np.median(time_samples)) if time_samples else 0.0

            _experiment_results[algo_name].append(avg_acc)
            _metric_results[algo_name]['f1'].append(avg_f1)
            _metric_results[algo_name]['roc_auc'].append(avg_roc)
            _metric_results[algo_name]['pr_auc'].append(avg_pr)
            _efficiency_results[algo_name].append(med_t)
            if conv_first:
                _convergence_results[algo_name].append(conv_first)

            row += f"{avg_acc:<16.4f}"

        if verbose:
            print(row)

    return dict(_experiment_results)


# ─────────────────────────────────────────────────────────────
# run_epinions_experiment
# ─────────────────────────────────────────────────────────────

def run_epinions_experiment(verbose: bool = True) -> dict:
    """
    在 Epinions 大数据集上运行信任计算（高效模式）。
    EigenTrust: 直接矩阵输入；PageRank: 直接图输入。
    PeerTrust / ImprovedEigenTrust 需要 TrustNetwork，跳过（记录 N/A）。
    返回 {algo_name: [time_seconds]} 字典。
    """
    from data_loader import load_epinions, build_eigentrust_matrix, build_pagerank_graph
    from config import DATA_PATHS

    results = {name: [] for name in _ALGORITHMS}

    epinions_path = DATA_PATHS['epinions']
    if not os.path.exists(epinions_path):
        if verbose:
            print(f"  警告: Epinions 数据集不存在 {epinions_path}，跳过")
        return results

    if verbose:
        print(f"  加载 Epinions 数据集: {epinions_path}")

    try:
        edges = load_epinions(epinions_path)
        if verbose:
            print(f"  加载边数: {len(edges)}")
    except Exception as e:
        if verbose:
            print(f"  警告: 加载失败 {e}")
        return results

    # 采样节点以控制内存 (max 2000)
    MAX_NODES = 2000
    all_nodes = set()
    for t, te, _ in edges:
        all_nodes.add(t)
        all_nodes.add(te)

    if len(all_nodes) > MAX_NODES:
        from collections import Counter
        degree = Counter()
        for t, te, _ in edges:
            degree[t] += 1
        top_nodes = set(n for n, _ in degree.most_common(MAX_NODES))
        edges = [(t, te, v) for t, te, v in edges if t in top_nodes and te in top_nodes]
        all_nodes = top_nodes

    node_list = sorted(all_nodes)
    node_idx  = {n: i for i, n in enumerate(node_list)}
    n = len(node_list)

    if verbose:
        print(f"  高效模式节点数: {n}")

    # EigenTrust（矩阵模式）
    try:
        if verbose:
            print("  构建 EigenTrust 矩阵...")
        C = build_eigentrust_matrix(edges, node_idx, n)
        t0 = time.time()
        EigenTrust.compute(C, max_iter=EIGENTRUST_MAX_ITER, verbose=False)
        elapsed = time.time() - t0
        results['EigenTrust'].append(elapsed)
        if verbose:
            print(f"  EigenTrust 完成，耗时 {elapsed:.4f}s")
    except Exception as e:
        if verbose:
            print(f"  EigenTrust 失败: {e}")

    # PageRank（图模式）
    try:
        if verbose:
            print("  构建 PageRank 图...")
        G = build_pagerank_graph(edges, max_nodes=MAX_NODES)
        t0 = time.time()
        PageRankTrust.compute(G)
        elapsed = time.time() - t0
        results['PageRank'].append(elapsed)
        if verbose:
            print(f"  PageRank 完成，耗时 {elapsed:.4f}s")
    except Exception as e:
        if verbose:
            print(f"  PageRank 失败: {e}")

    # PeerTrust / ImprovedEigenTrust: 需要 TrustNetwork，跳过
    for name in ('PeerTrust', 'EigenTrust(改进)'):
        if verbose:
            print(f"  {name}: Epinions 高效模式下跳过（需要 TrustNetwork 对象）")

    return results


# ─────────────────────────────────────────────────────────────
# 打印函数
# ─────────────────────────────────────────────────────────────

def print_results_table(results: dict):
    """格式化打印准确率结果表格"""
    if not results:
        print("  （无结果）")
        return

    algo_names = list(results.keys())
    header = f"  {'攻击比例':<10}" + "".join(f"{n:<16}" for n in algo_names)
    print(header)
    print("  " + "-" * (10 + 16 * len(algo_names)))

    num_ratios = max(len(v) for v in results.values()) if results else 0
    for i in range(num_ratios):
        ratio = ATTACK_RATIOS[i] if i < len(ATTACK_RATIOS) else i
        row = f"  {int(ratio * 100):>4}%      "
        for name in algo_names:
            vals = results.get(name, [])
            val = vals[i] if i < len(vals) else float('nan')
            row += f"{val:<16.4f}"
        print(row)


def print_efficiency_table():
    """格式化打印效率（计算时间）结果表格"""
    if not _efficiency_results:
        print("  （无效率数据）")
        return

    algo_names = list(_efficiency_results.keys())
    header = f"  {'攻击比例':<10}" + "".join(f"{n:<16}" for n in algo_names)
    print(header)
    print("  " + "-" * (10 + 16 * len(algo_names)))

    num_ratios = max(len(v) for v in _efficiency_results.values()) if _efficiency_results else 0
    for i in range(num_ratios):
        ratio = ATTACK_RATIOS[i] if i < len(ATTACK_RATIOS) else i
        row = f"  {int(ratio * 100):>4}%      "
        for name in algo_names:
            vals = _efficiency_results.get(name, [])
            val = vals[i] if i < len(vals) else float('nan')
            row += f"{val:<16.4f}"
        print(row)

    print("\n  各算法全局中位数计算时间：")
    for name, times in _efficiency_results.items():
        if times:
            print(f"    {name:<16}: {np.median(times) * 1000:.2f} ms")


# ─────────────────────────────────────────────────────────────
# 独立运行入口
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("P2P 基线实验")
    print("=" * 60)

    p2p_results = run_p2p_experiment(verbose=True)

    print("\n" + "-" * 60)
    print("准确率汇总")
    print("-" * 60)
    print_results_table(p2p_results)

    print("\n" + "-" * 60)
    print("效率汇总")
    print("-" * 60)
    print_efficiency_table()

    print("\n" + "=" * 60)
    print("Epinions 大数据集实验")
    print("=" * 60)
    epinions_results = run_epinions_experiment(verbose=True)
