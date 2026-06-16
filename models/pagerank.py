# models/pagerank.py
r"""
标准 PageRank 信任传播算法（严格遵循原论文定义）
1. 无权图转移矩阵：M_ij = 1 / L(j)
2. 保持概率分布性质：满足 \sum PR(i) = 1，不进行任何破坏概率分布的归一化
3. 收敛判断：采用 L1 范数

支持两种输入模式：
1. TrustNetwork 对象（完整仿真链路）
2. networkx.DiGraph（高效模式，适用于大数据集）
"""

import numpy as np
import time
import networkx as nx
from core.network import TrustNetwork
from config import PAGERANK_DAMPING


class PageRankTrust:
    """标准 PageRank 信任传播"""

    @staticmethod
    def compute(network, damping: float = PAGERANK_DAMPING,
                max_iter: int = 50, track_convergence=False):
        start_time = time.time()
        
        # 判断输入类型
        is_graph_input = isinstance(network, nx.DiGraph)
        
        if is_graph_input:
            # 高效模式：直接使用 networkx.DiGraph
            G = network
            nodes = list(G.nodes())
            n = len(nodes)
            node_id_to_idx = {node: i for i, node in enumerate(nodes)}
        else:
            # 标准模式：TrustNetwork 对象
            n = len(network.nodes)
            node_id_to_idx = {node.id: i for i, node in enumerate(network.nodes)}
            G = network.graph

        # 1. 构建标准邻接矩阵（无权，出链均匀平分概率流量）
        # 对于大数据集，使用稀疏矩阵避免内存问题
        try:
            from scipy.sparse import lil_matrix, csr_matrix
            
            adj = lil_matrix((n, n))
            dangling_nodes = []
            
            for from_id in G.nodes():
                if from_id not in node_id_to_idx:
                    continue
                from_idx = node_id_to_idx[from_id]
                out_degree = G.out_degree(from_id)
                
                if out_degree > 0:
                    # 原论文：对于有出链的节点，将其出链的转移概率均分
                    for _, to_id in G.out_edges(from_id):
                        if to_id in node_id_to_idx:
                            to_idx = node_id_to_idx[to_id]
                            adj[to_idx, from_idx] = 1.0 / out_degree
                else:
                    dangling_nodes.append(from_idx)
            
            # 处理悬空节点（无出链）
            if dangling_nodes:
                # 对每个悬空节点，其出链均匀分配到所有节点
                for idx in dangling_nodes:
                    adj[:, idx] = 1.0 / n

            # 转换为 CSR 格式便于计算
            adj = adj.tocsr()
                
        except ImportError:
            # 如果没有 scipy，使用稠密矩阵
            adj = np.zeros((n, n))
            for from_id in G.nodes():
                if from_id not in node_id_to_idx:
                    continue
                from_idx = node_id_to_idx[from_id]
                out_degree = G.out_degree(from_id)
                
                if out_degree > 0:
                    for _, to_id in G.out_edges(from_id):
                        if to_id in node_id_to_idx:
                            to_idx = node_id_to_idx[to_id]
                            adj[to_idx][from_idx] = 1.0 / out_degree
                else:
                    adj[:, from_idx] = 1.0 / n

        # 2. 初始化概率分布向量（各节点概率均等，且和为 1）
        t = np.ones(n) / n
        convergence_history = [] if track_convergence else None

        # 3. 幂迭代计算
        for _ in range(max_iter):
            # 严格按照原论文公式迭代：t^(k+1) = d * M * t^(k) + (1-d)/n * e
            t_new = damping * adj @ t + (1.0 - damping) / n

            # 采用 L1 范数（Manhattan distance）来计算前后两次迭代向量的差异
            change = np.linalg.norm(t_new - t, ord=1)
            
            if track_convergence:
                convergence_history.append(change)

            # 更新向量
            t = t_new

            # 判断收敛
            if change < 1e-6:
                break

        # 4. 保持概率分布特性
        # 理论上幂迭代过程满足列随机矩阵特征，t 的和始终为 1。
        # 为了防止浮点数精度累积误差，进行一次标准的概率归一化（使和严格为 1）
        t_sum = np.sum(t)
        if t_sum > 0:
            t = t / t_sum
        else:
            t = np.ones(n) / n

        # 将计算出的标准 PageRank 值分配给节点（仅标准模式）
        if not is_graph_input:
            for i, node in enumerate(network.nodes):
                node.trust_value = t[i]

        compute_time = time.time() - start_time

        if track_convergence:
            return t, convergence_history, compute_time
        return t, compute_time