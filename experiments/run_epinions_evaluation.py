"""
Epinions 四算法统一评估实验

实验目的：
- 使用 Epinions 的真实正/负信任边构造统一训练网络
- 在同一测试边集合上评估 EigenTrust、ImprovedEigenTrust、PeerTrust、PageRank
- 不修改各算法实现，只做数据适配与指标计算

运行方式：
    python experiments/run_epinions_evaluation.py

当前你的 Epinions 文件里读取到的子图没有显式负边，所以脚本在测试集里用“未观测节点对”采样为负样本
训练集仍只使用真实 Epinions 边
这种做法适合做链接/信任预测评估，后续论文里需要说明。

PeerTrust 的核心优势依赖：
- 多次交易记录；
- 评价者可信度；
- 交易上下文；
- 反馈密度；
- 卖家收到反馈的比例；
- 交易金额或交易场景差异。
但当前 Epinions 实验中：
- 一条信任边基本只对应一次静态评价；
- 没有真实交易金额；
- 没有交易上下文；
- 没有完整交易序列；
- 当前读取到的子图没有显式负边；
- 测试负样本来自“未观测节点对”。
这会让 PeerTrust 的交易反馈聚合机制发挥不足。

结论：在 Epinions 静态社交信任网络评估中，基于全局结构传播的 EigenTrust/PageRank 更适配该数据形式；
PeerTrust 由于缺少交易上下文和多轮反馈，其优势未能充分体现。
在真实社交信任拓扑下，ImprovedEigenTrust 保持较好排序性能。

Epinions 是静态社交信任网络，缺少交易上下文和多轮交易反馈，因此对依赖交易上下文的 PeerTrust 存在一定限制。
为避免单一数据集造成偏差，本文同时在 P2P 仿真网络和 Bitcoin OTC 数据集上进行补充验证
"""

import os
import sys
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import DATA_PATHS, EIGENTRUST_MAX_ITER
from core.node import Node
from core.network import TrustNetwork
from data_loader import load_epinions
from models.eigentrust import EigenTrust
from models.eigentrust_improved import ImprovedEigenTrust
from models.pagerank import PageRankTrust
from models.peertrust import PeerTrust

Edge = Tuple[int, int, int]


@dataclass
class EpinionsNetworkBundle:
    network: TrustNetwork
    id_to_idx: Dict[int, int]
    idx_to_id: Dict[int, int]
    train_edges: List[Edge]
    test_edges: List[Edge]


class EpinionsTrustNetwork(TrustNetwork):
    """轻量 TrustNetwork，仅用于承载 Epinions 子图和交易历史。"""

    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.attack_ratio = 0.0
        self.dataset_name = "epinions"
        self.nodes = [Node(i, False) for i in range(num_nodes)]
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(range(num_nodes))


def _make_seed(seed: Optional[int]) -> int:
    return random.SystemRandom().randint(0, 2**32 - 1) if seed is None else seed


def _sample_balanced_edges(edges: List[Edge], max_edges: int, seed: int) -> List[Edge]:
    if max_edges <= 0 or len(edges) <= max_edges:
        return list(edges)

    rng = random.Random(seed)
    positives = [edge for edge in edges if edge[2] > 0]
    negatives = [edge for edge in edges if edge[2] < 0]

    if not positives or not negatives:
        sampled = list(edges)
        rng.shuffle(sampled)
        return sampled[:max_edges]

    pos_target = max_edges // 2
    neg_target = max_edges - pos_target
    pos_sample = rng.sample(positives, min(pos_target, len(positives)))
    neg_sample = rng.sample(negatives, min(neg_target, len(negatives)))

    remaining = max_edges - len(pos_sample) - len(neg_sample)
    if remaining > 0:
        used = set(pos_sample + neg_sample)
        candidates = [edge for edge in edges if edge not in used]
        pos_sample.extend(rng.sample(candidates, min(remaining, len(candidates))))

    sampled = pos_sample + neg_sample
    rng.shuffle(sampled)
    return sampled


def _select_subgraph_edges(edges: List[Edge], max_nodes: int, seed: int) -> List[Edge]:
    """每次随机选择一个包含边的 Epinions 子图。"""
    rng = random.Random(seed)
    shuffled = list(edges)
    rng.shuffle(shuffled)

    selected_nodes = set()
    for src, dst, _ in shuffled:
        if src in selected_nodes and dst in selected_nodes:
            continue
        if len(selected_nodes) < max_nodes:
            selected_nodes.add(src)
            selected_nodes.add(dst)
        if len(selected_nodes) >= max_nodes:
            break

    return [edge for edge in edges if edge[0] in selected_nodes and edge[1] in selected_nodes]


def _split_edges(edges: List[Edge], test_ratio: float, seed: int) -> Tuple[List[Edge], List[Edge]]:
    rng = random.Random(seed)
    positives = [edge for edge in edges if edge[2] > 0]
    negatives = [edge for edge in edges if edge[2] < 0]

    def split_group(group: List[Edge]) -> Tuple[List[Edge], List[Edge]]:
        group = list(group)
        rng.shuffle(group)
        n_test = max(1, int(len(group) * test_ratio)) if len(group) > 1 else 0
        return group[n_test:], group[:n_test]

    pos_train, pos_test = split_group(positives)
    neg_train, neg_test = split_group(negatives)
    train_edges = pos_train + neg_train
    test_edges = pos_test + neg_test
    rng.shuffle(train_edges)
    rng.shuffle(test_edges)
    return train_edges, test_edges


def _sample_unknown_negative_edges(
    train_edges: List[Edge],
    test_edges: List[Edge],
    nodes: List[int],
    count: int,
    seed: int,
) -> List[Edge]:
    """当数据集没有显式负边时，从未观测节点对中采样负测试边。"""
    rng = random.Random(seed)
    observed = {(src, dst) for src, dst, _ in train_edges + test_edges}
    negatives = []
    max_attempts = max(count * 50, 1000)

    for _ in range(max_attempts):
        if len(negatives) >= count:
            break
        src = rng.choice(nodes)
        dst = rng.choice(nodes)
        if src == dst or (src, dst) in observed:
            continue
        observed.add((src, dst))
        negatives.append((src, dst, -1))

    return negatives


def build_epinions_network(
    max_nodes: int = 1000,
    max_edges: int = 20000,
    test_ratio: float = 0.2,
    seed: Optional[int] = None,
) -> EpinionsNetworkBundle:
    seed = _make_seed(seed)
    epinions_path = DATA_PATHS["epinions"]
    if not os.path.exists(epinions_path):
        raise FileNotFoundError(f"Epinions 数据集不存在: {epinions_path}")

    edges = load_epinions(epinions_path)
    edges = [edge for edge in edges if edge[0] != edge[1] and edge[2] != 0]
    edges = _select_subgraph_edges(edges, max_nodes=max_nodes, seed=seed)
    edges = _sample_balanced_edges(edges, max_edges=max_edges, seed=seed)

    train_edges, test_edges = _split_edges(edges, test_ratio=test_ratio, seed=seed)
    if not train_edges or not test_edges:
        raise ValueError("Epinions 训练边或测试边为空，请调大 max_nodes/max_edges")

    nodes = sorted({src for src, _, _ in train_edges + test_edges} | {dst for _, dst, _ in train_edges + test_edges})
    if len({1 if edge[2] > 0 else 0 for edge in test_edges}) < 2:
        positive_count = sum(1 for _, _, value in test_edges if value > 0)
        needed = positive_count if positive_count > 0 else max(1, len(test_edges))
        test_edges.extend(_sample_unknown_negative_edges(train_edges, test_edges, nodes, needed, seed))

    if len({1 if edge[2] > 0 else 0 for edge in test_edges}) < 2:
        raise ValueError("Epinions 测试集缺少正负两类样本，无法计算有效 AUC")

    nodes = sorted({src for src, _, _ in train_edges + test_edges} | {dst for _, dst, _ in train_edges + test_edges})
    id_to_idx = {node_id: idx for idx, node_id in enumerate(nodes)}
    idx_to_id = {idx: node_id for node_id, idx in id_to_idx.items()}

    network = EpinionsTrustNetwork(len(nodes))

    for timestamp, (src, dst, trust_value) in enumerate(train_edges):
        src_idx = id_to_idx[src]
        dst_idx = id_to_idx[dst]
        feedback = 1.0 if trust_value > 0 else -1.0

        network.nodes[src_idx].add_transaction({
            "other": dst_idx,
            "buyer": src_idx,
            "seller": dst_idx,
            "feedback": feedback,
            "true_feedback": feedback,
            "amount": 1.0,
            "timestamp": timestamp,
        })

        if trust_value > 0:
            network.graph.add_edge(src_idx, dst_idx)

    return EpinionsNetworkBundle(network, id_to_idx, idx_to_id, train_edges, test_edges)


def _extract_result(algo_name: str, result):
    if algo_name == "EigenTrust(改进)":
        return result.trust_vector, result.convergence_history, result.compute_time
    if isinstance(result, tuple):
        if len(result) == 3:
            return result[0], result[1], result[2]
        if len(result) == 2:
            return result[0], None, result[1]
    raise ValueError(f"无法解析 {algo_name} 的返回结果")


def _evaluate_scores(trust_vector: np.ndarray, test_edges: List[Edge], id_to_idx: Dict[int, int]) -> Dict[str, float]:
    y_true = []
    y_score = []

    for _, dst, trust_value in test_edges:
        if dst not in id_to_idx:
            continue
        y_true.append(1 if trust_value > 0 else 0)
        y_score.append(float(trust_vector[id_to_idx[dst]]))

    if not y_true:
        raise ValueError("测试集中没有可评估边")

    threshold = float(np.median(y_score))
    y_pred = [1 if score >= threshold else 0 for score in y_score]

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else 0.5,
        "pr_auc": average_precision_score(y_true, y_score),
    }


def run_epinions_evaluation(
    max_nodes: int = 1000,
    max_edges: int = 20000,
    test_ratio: float = 0.2,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """运行 Epinions 四算法统一评估。"""
    bundle = build_epinions_network(
        max_nodes=max_nodes,
        max_edges=max_edges,
        test_ratio=test_ratio,
        seed=seed,
    )

    if verbose:
        pos_train = sum(1 for _, _, v in bundle.train_edges if v > 0)
        neg_train = len(bundle.train_edges) - pos_train
        pos_test = sum(1 for _, _, v in bundle.test_edges if v > 0)
        neg_test = len(bundle.test_edges) - pos_test
        print("Epinions 四算法评估")
        print(f"  节点数: {len(bundle.network.nodes)}")
        print(f"  训练边: {len(bundle.train_edges)} (正={pos_train}, 负={neg_train})")
        print(f"  测试边: {len(bundle.test_edges)} (正={pos_test}, 负={neg_test})")

    algorithms = {
        "EigenTrust": lambda net: EigenTrust.compute(net, max_iter=EIGENTRUST_MAX_ITER, track_convergence=True),
        "EigenTrust(改进)": lambda net: ImprovedEigenTrust.compute(net, max_iter=EIGENTRUST_MAX_ITER, track_convergence=True),
        "PeerTrust": lambda net: PeerTrust.compute(net, track_convergence=True),
        "PageRank": lambda net: PageRankTrust.compute(net, track_convergence=True),
    }

    results = {}
    for algo_name, runner in algorithms.items():
        if verbose:
            print(f"\n运行 {algo_name}...")
        start = time.time()
        raw_result = runner(bundle.network)
        trust_vector, convergence, compute_time = _extract_result(algo_name, raw_result)
        metrics = _evaluate_scores(trust_vector, bundle.test_edges, bundle.id_to_idx)
        metrics["time"] = compute_time if compute_time is not None else time.time() - start
        metrics["iterations"] = len(convergence) if convergence is not None else 0
        results[algo_name] = metrics

        if verbose:
            print(
                f"  Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, "
                f"ROC-AUC={metrics['roc_auc']:.4f}, PR-AUC={metrics['pr_auc']:.4f}, "
                f"Time={metrics['time']:.4f}s"
            )

    return results


def plot_epinions_results(results: Dict[str, Dict[str, float]], output_dir: str = "results/figures"):
    os.makedirs(output_dir, exist_ok=True)

    algorithms = list(results.keys())
    display_algorithms = [algo.replace("EigenTrust(改进)", "ImprovedEigenTrust") for algo in algorithms]
    metrics = ["accuracy", "f1", "roc_auc", "pr_auc"]
    metric_labels = ["Accuracy", "F1", "ROC-AUC", "PR-AUC"]
    # 使用 utils.visualization 中定义的固定算法配色，跨图表保持一致
    from utils.visualization import _color_for
    x = np.arange(len(metric_labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (algo_name, display_name) in enumerate(zip(algorithms, display_algorithms)):
        values = [results[algo_name][metric] for metric in metrics]
        bars = ax.bar(x + (i - 1.5) * width, values, width,
                      label=display_name, color=_color_for(algo_name, i), alpha=0.85)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                    f"{height:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Epinions Dataset - Four Metrics Comparison", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    metrics_path = os.path.join(output_dir, "epinions_metrics_comparison.png")
    plt.savefig(metrics_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Epinions metrics comparison saved: {metrics_path}")


def print_results_table(results: Dict[str, Dict[str, float]]):
    print("\nEpinions 评估结果")
    print("-" * 86)
    print(f"{'Algorithm':<20} {'Accuracy':>10} {'F1':>10} {'ROC-AUC':>10} {'PR-AUC':>10} {'Time(s)':>10} {'Iter':>8}")
    print("-" * 86)
    for algo_name, metrics in results.items():
        print(
            f"{algo_name:<20} "
            f"{metrics['accuracy']:>10.4f} "
            f"{metrics['f1']:>10.4f} "
            f"{metrics['roc_auc']:>10.4f} "
            f"{metrics['pr_auc']:>10.4f} "
            f"{metrics['time']:>10.4f} "
            f"{metrics['iterations']:>8}"
        )


if __name__ == "__main__":
    results = run_epinions_evaluation()
    print_results_table(results)
    plot_epinions_results(results)
