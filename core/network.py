"""
network.py - 信任网络
负责：节点管理、网络拓扑构建、交易执行

Epinions/Bitcoin OTC 数据集模式：
  数据集的信任边只决定"谁和谁可能交易"，
  交易的具体好评/差评由攻击模型动态生成。
"""

import random
import networkx as nx
from typing import List, Tuple, Optional
from config import RANDOM_SEED, GRAPH_TYPE, DEGREE, DATA_PATHS
from core.node import Node
from core.transaction import Transaction
from data_loader import load_epinions, load_bitcoin_otc

# =========================================================
# 模块级全局缓存：数据集只加载一次
# =========================================================
_dataset_cache = None
_dataset_loaded_print_done = False


def _load_dataset(dataset_name: str = 'epinions'):
    """加载数据集（全局单例，只执行一次磁盘IO）"""
    global _dataset_cache, _dataset_loaded_print_done

    if _dataset_cache is not None:
        return _dataset_cache

    import os

    dataset_path = DATA_PATHS.get(dataset_name, DATA_PATHS['epinions'])
    if not os.path.exists(dataset_path):
        print(f"警告: 数据集文件不存在 {dataset_path}")
        return None

    adj = {}
    if dataset_name == 'bitcoin_otc':
        # 使用 Bitcoin OTC 数据集
        tx_data = load_bitcoin_otc(dataset_path)
        for source, target, rating, time in tx_data:
            if source not in adj:
                adj[source] = []
            adj[source].append(target)
        print(f"从 Bitcoin OTC 数据集加载了 {len(tx_data)} 条交易，{len(adj)} 个节点")
    else:
        # 使用 Epinions 数据集（默认）
        edges_data = load_epinions(dataset_path)
        for trustor, trustee, _ in edges_data:
            if trustor not in adj:
                adj[trustor] = []
            adj[trustor].append(trustee)
        print(f"从 Epinions 数据集加载了 {len(edges_data)} 条边，{len(adj)} 个节点")

    _dataset_cache = adj
    return _dataset_cache


class TrustNetwork:
    """信任网络"""

    def __init__(self, num_nodes: int, attack_ratio: float, dataset_name: str = 'epinions'):
        """
        初始化网络

        参数:
            num_nodes: 节点总数
            attack_ratio: 恶意节点比例
            dataset_name: 数据集名称 ('epinions' 或 'bitcoin_otc')
        """
        self.num_nodes = num_nodes
        self.attack_ratio = attack_ratio
        self.dataset_name = dataset_name
        self.nodes: List[Node] = []
        self.graph: nx.DiGraph = nx.DiGraph()  # 有向图表示可交易关系

        self._init_nodes()
        self._init_graph()

    def _init_nodes(self):
        """初始化节点（恶意/正常）"""
        num_malicious = int(self.num_nodes * self.attack_ratio)

        for i in range(self.num_nodes):
            is_mal = i < num_malicious
            self.nodes.append(Node(i, is_mal))

        # 打乱顺序，避免恶意节点都集中在前面
        random.shuffle(self.nodes)

    def _init_graph(self):
        """初始化社交网络拓扑"""
        if GRAPH_TYPE == "random":
            self._build_random_graph()
        elif GRAPH_TYPE == "small_world":
            self._build_small_world_graph()
        elif GRAPH_TYPE == "dataset":
            self._build_dataset_graph()
        else:
            self._build_random_graph()

    def _build_dataset_graph(self):
        """
        从数据集构建可交易关系图

        核心思路：
        - 选取数据集中度数最高的 num_nodes 个节点
        - 这些节点的信任边 = 可交易配对
        - 交易的反馈（好评/差评）由 perform_transaction 动态生成
        """
        adj = _load_dataset(self.dataset_name)
        if adj is None:
            self._build_random_graph()
            return

        # 按度数排序，取 top num_nodes 个节点
        degrees = {u: len(vs) for u, vs in adj.items()}
        sorted_by_degree = sorted(degrees, key=degrees.get, reverse=True)
        selected = sorted_by_degree[:self.num_nodes]
        selected_set = set(selected)

        if len(selected) < self.num_nodes:
            print(f"  警告: 数据集节点数({len(sorted_by_degree)})少于请求节点数({self.num_nodes})，使用全部可用节点")
            selected = sorted_by_degree
            selected_set = set(selected)
            self.num_nodes = len(selected)

        # 数据集ID -> 内部ID
        ds_to_internal = {ds_id: i for i, ds_id in enumerate(selected)}

        # 在选取的节点之间建立边
        edge_count = 0
        for ds_u in selected:
            if ds_u in adj:
                for ds_v in adj[ds_u]:
                    if ds_v in selected_set:
                        self.graph.add_edge(
                            ds_to_internal[ds_u],
                            ds_to_internal[ds_v]
                        )
                        edge_count += 1

        # 仅首次打印子图信息，避免重复刷屏
        if not hasattr(TrustNetwork, '_subgraph_info_printed'):
            TrustNetwork._subgraph_info_printed = True
            print(f"  数据集子图: {self.num_nodes} 节点, {edge_count} 条可交易边")

    def _build_random_graph(self):
        """构建随机图"""
        for node in self.nodes:
            others = [n for n in self.nodes if n.id != node.id]
            neighbors = random.sample(others, min(DEGREE, len(others)))
            for neighbor in neighbors:
                self.graph.add_edge(node.id, neighbor.id)

    def _build_small_world_graph(self):
        """构建小世界网络（Watts-Strogatz模型）"""
        k = DEGREE
        for i, node in enumerate(self.nodes):
            for j in range(1, k // 2 + 1):
                neighbor = self.nodes[(i + j) % self.num_nodes]
                self.graph.add_edge(node.id, neighbor.id)
                self.graph.add_edge(neighbor.id, node.id)

        p = 0.1
        edges = list(self.graph.edges())
        for u, v in edges:
            if random.random() < p:
                self.graph.remove_edge(u, v)
                new_v = random.choice([n.id for n in self.nodes if n.id != u])
                self.graph.add_edge(u, new_v)

    def perform_transaction(self, timestamp: int) -> Transaction:
        """
        执行一笔随机交易

        数据集模式：从 Epinions 信任边中选取买方→卖方配对
                      反馈（好评/差评）由攻击模型动态生成

        随机图模式：完全随机选取买卖双方
        """
        if GRAPH_TYPE == "dataset" and self.graph.number_of_edges() > 0:
            # 从数据集的信任边中随机选一条作为交易配对
            all_edges = list(self.graph.edges())
            buyer_id, seller_id = random.choice(all_edges)
            buyer = self.get_node_by_id(buyer_id)
            seller = self.get_node_by_id(seller_id)
        else:
            buyer = random.choice(self.nodes)
            seller = random.choice([n for n in self.nodes if n != buyer])

        if buyer is None or seller is None:
            buyer = random.choice(self.nodes)
            seller = random.choice([n for n in self.nodes if n != buyer])

        amount = random.uniform(1, 100)

        # =====================================================
        # 攻击模型：交易反馈的动态生成
        #
        # 真实反馈：卖家恶意 → 差评(-1)，卖家正常 → 好评(+1)
        # 谎报反馈：恶意买家有30%概率颠倒反馈
        # =====================================================
        true_feedback = -1 if seller.is_malicious else 1

        reported_feedback = true_feedback
        if buyer.is_malicious and random.random() < 0.3:
            reported_feedback = -true_feedback  # 颠倒反馈

        tx = Transaction(
            buyer_id=buyer.id,
            seller_id=seller.id,
            feedback=reported_feedback,
            true_feedback=true_feedback,
            amount=amount,
            timestamp=timestamp
        )

        buyer.add_transaction({
            'other': seller.id,
            'feedback': reported_feedback,
            'true_feedback': true_feedback,
            'amount': amount,
            'timestamp': timestamp
        })

        return tx

    def get_node_by_id(self, node_id: int) -> Optional[Node]:
        """根据ID获取节点"""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    @property
    def num_malicious(self) -> int:
        return sum(1 for n in self.nodes if n.is_malicious)

    @property
    def num_honest(self) -> int:
        return self.num_nodes - self.num_malicious