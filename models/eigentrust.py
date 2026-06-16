"""
EigenTrust 算法

基于论文 "The EigenTrust Algorithm for Reputation Management in P2P Networks"
(Kamvar, Schlosser & Garcia-Molina, WWW 2003)
- 局部信任矩阵归一化
- 幂迭代求解全局信任向量
- 悬挂节点处理
- 阻尼系数控制

支持两种输入模式：
1. TrustNetwork 对象（完整仿真链路）
2. 直接接收信任矩阵 C（高效模式，适用于大数据集）
"""

import numpy as np
import time
from typing import Optional, Tuple, List, Dict, Union
from core.network import TrustNetwork


class EigenTrust:
    """
    EigenTrust 声誉管理算法
    
    核心公式: t^(k+1) = (1-α) * C^T * t^(k) + α * p
    其中:
        t: 全局信任向量
        C: 归一化的局部信任矩阵 (C[i][j] = 节点i对节点j的信任)
        p: 预信任向量
        α: 预信任权重 (论文中 a < 1)
    """
    
    @staticmethod
    def compute(
        network: Union[TrustNetwork, np.ndarray],
        max_iter: int = 100,
        tolerance: float = 1e-6,
        alpha: float = 0.15,
        pre_trust: Optional[np.ndarray] = None,  # 新增：支持自定义预信任向量
        track_convergence: bool = False,
        verbose: bool = False
    ) -> Tuple[np.ndarray, Optional[List[float]], float]:
        """
        计算网络中所有节点的 EigenTrust 值
        
        Args:
            network: 信任网络对象，包含节点和交易历史
            max_iter: 最大迭代次数
            tolerance: 收敛容差
            alpha: 预信任向量权重 (0 < alpha < 1)
            pre_trust: 自定义预信任向量，None 时使用均匀分布
            track_convergence: 是否追踪收敛历史
            verbose: 是否打印详细信息
            
        Returns:
            如果 track_convergence=True: (信任向量, 收敛历史, 计算时间)
            否则: (信任向量, None, 计算时间)
        """
        
        # 参数验证
        if not 0 < alpha < 1:
            raise ValueError(f"alpha 必须在 (0, 1) 范围内，当前值: {alpha}")
        if max_iter <= 0:
            raise ValueError(f"max_iter 必须大于 0，当前值: {max_iter}")
        if tolerance <= 0:
            raise ValueError(f"tolerance 必须大于 0，当前值: {tolerance}")
        
        # 判断输入类型
        is_matrix_input = isinstance(network, np.ndarray)
        
        if is_matrix_input:
            # 高效模式：直接使用输入的信任矩阵
            C = network
            n = C.shape[0]
            node_id_to_idx = None
        else:
            # 标准模式：TrustNetwork 对象
            n = len(network.nodes)
            if n == 0:
                raise ValueError("网络中没有任何节点")
            node_id_to_idx = {node.id: i for i, node in enumerate(network.nodes)}
        
        start_time = time.time()
        
        # 1. 初始化预信任向量 p
        if pre_trust is None:
            pre_trust = np.ones(n) / n
        else:
            pre_trust = np.asarray(pre_trust, dtype=float)
            if len(pre_trust) != n:
                raise ValueError(f"预信任向量长度 {len(pre_trust)} 与节点数 {n} 不匹配")
            # 归一化
            sum_p = np.sum(pre_trust)
            if sum_p > 0:
                pre_trust = pre_trust / sum_p
            else:
                pre_trust = np.ones(n) / n
        
        if verbose:
            print(f"开始计算 EigenTrust: {n} 个节点, alpha={alpha}, max_iter={max_iter}")
            if is_matrix_input:
                print("  输入模式: 直接矩阵输入（高效模式）")
            else:
                print("  输入模式: TrustNetwork 对象（标准模式）")
        
        # 2. 构建或使用归一化的局部信任矩阵 C
        if is_matrix_input:
            # 直接使用输入的矩阵
            pass
        else:
            C = EigenTrust._build_trust_matrix(network, node_id_to_idx, pre_trust, verbose)
        
        # 3. 幂迭代求解全局信任向量
        t, convergence_history = EigenTrust._power_iteration(
            C, pre_trust, alpha, max_iter, tolerance, track_convergence, verbose
        )
        
        compute_time = time.time() - start_time
        
        # 4. 将信任值写回节点（仅标准模式）
        if not is_matrix_input:
            for i, node in enumerate(network.nodes):
                node.trust_value = t[i]
        
        if verbose:
            iter_count = len(convergence_history) if convergence_history else max_iter
            print(f"计算完成，耗时: {compute_time:.4f} 秒，迭代次数: {iter_count}")
            print(f"信任值范围: [{t.min():.6f}, {t.max():.6f}]")
            print(f"信任值总和: {t.sum():.6f}")
        
        if track_convergence:
            return t, convergence_history, compute_time
        return t, None, compute_time
    
    @staticmethod
    def _build_trust_matrix(
        network: TrustNetwork,
        node_id_to_idx: Dict[str, int],
        pre_trust: np.ndarray,
        verbose: bool = False
    ) -> np.ndarray:
        """
        构建并归一化局部信任矩阵 C
        
        C[i][j] 表示节点 i 对节点 j 的归一化信任值
        满足: 对于每个 i, sum_j C[i][j] = 1 (除非 i 是悬挂节点)
        
        论文定义 (Section 4.1):
        s_ij = sat(i,j) - unsat(i,j)
        c_ij = max(s_ij, 0) / sum_j max(s_ij, 0)
        悬挂节点: c_ij = p_j
        """
        n = len(network.nodes)
        C = np.zeros((n, n))
        hanging_nodes = 0
        
        for i, node in enumerate(network.nodes):
            # 聚合交易反馈
            pos_feedback: Dict[int, float] = {}
            neg_feedback: Dict[int, float] = {}
            
            for tx in node.transaction_history:
                other_id = tx.get('other')
                if not other_id:
                    continue
                    
                if other_id not in node_id_to_idx:
                    if verbose:
                        print(f"警告: 节点 {node.id} 的交易目标 {other_id} 不存在于网络中")
                    continue
                
                j = node_id_to_idx[other_id]
                feedback = tx.get('feedback', 0.0)
                
                # 聚合正负反馈 (论文 Section 3)
                if feedback > 0:
                    pos_feedback[j] = pos_feedback.get(j, 0.0) + feedback
                elif feedback < 0:
                    neg_feedback[j] = neg_feedback.get(j, 0.0) + abs(feedback)
            
            # 计算原始信任分数 s_ij = max(sat - unsat, 0)
            for j in range(n):
                pos = pos_feedback.get(j, 0.0)
                neg = neg_feedback.get(j, 0.0)
                C[i][j] = max(pos - neg, 0.0)
            
            # 行归一化 (论文 Section 4.1)
            row_sum = np.sum(C[i, :])
            if row_sum > 1e-12:
                C[i, :] = C[i, :] / row_sum
            else:
                # 悬挂节点：无交易历史或所有信任值为0，退化至预信任向量 p
                C[i, :] = pre_trust.copy()
                hanging_nodes += 1
        
        if verbose and hanging_nodes > 0:
            print(f"检测到 {hanging_nodes} 个悬挂节点（无有效交易历史）")
        
        return C
    
    @staticmethod
    def _power_iteration(
        C: np.ndarray,
        pre_trust: np.ndarray,
        alpha: float,
        max_iter: int,
        tolerance: float,
        track_convergence: bool,
        verbose: bool = False
    ) -> Tuple[np.ndarray, Optional[List[float]]]:
        """
        幂迭代求解全局信任向量
        
        迭代公式 (论文 Algorithm 2): 
        t^(k+1) = (1-α) * C^T * t^(k) + α * p
        
        Args:
            C: 归一化的信任矩阵
            pre_trust: 预信任向量
            alpha: 预信任权重
            max_iter: 最大迭代次数
            tolerance: 收敛容差
            track_convergence: 是否追踪收敛历史
            verbose: 是否打印详细信息
            
        Returns:
            全局信任向量 t，以及可选的收敛历史
        """
        n = len(pre_trust)
        t = pre_trust.copy()
        C_T = C.T  # 提前转置以提高效率
        
        convergence_history = [] if track_convergence else None
        
        for iteration in range(max_iter):
            # 核心迭代公式
            t_new = (1 - alpha) * (C_T @ t) + alpha * pre_trust
            
            # 数值稳定性处理：确保非负
            t_new = np.maximum(t_new, 0)
            
            # 归一化（由于浮点误差，保证概率分布）
            sum_t_new = np.sum(t_new)
            if sum_t_new > 1e-12:
                t_new = t_new / sum_t_new
            else:
                if verbose:
                    print(f"警告: 迭代 {iteration} 时信任向量和为0，重置为预信任向量")
                t_new = pre_trust.copy()
            
            # 计算收敛性
            diff = np.linalg.norm(t_new - t, ord=1)
            
            if track_convergence:
                convergence_history.append(diff)
            
            if verbose and (iteration + 1) % 20 == 0:
                print(f"  迭代 {iteration + 1}/{max_iter}, 变化量: {diff:.8f}")
            
            # 先更新，再判断收敛
            t = t_new
            
            if diff < tolerance:
                if verbose:
                    print(f"收敛于迭代 {iteration + 1}, 变化量: {diff:.8f}")
                break
        
        return t, convergence_history
    
    @staticmethod
    def compute_with_custom_pre_trust(
        network: TrustNetwork,
        pre_trust_weights: Dict[str, float],
        **kwargs
    ) -> Tuple[np.ndarray, Optional[List[float]], float]:
        """
        使用自定义预信任向量计算 EigenTrust
        
        Args:
            network: 信任网络
            pre_trust_weights: 节点ID到预信任权重的映射，如 {'node1': 0.5, 'node2': 0.5}
            **kwargs: 传递给 compute() 的其他参数
        
        Returns:
            同 compute() 方法
        """
        n = len(network.nodes)
        node_id_to_idx = {node.id: i for i, node in enumerate(network.nodes)}
        
        # 构建自定义预信任向量
        pre_trust = np.zeros(n)
        total_weight = 0.0
        
        for node_id, weight in pre_trust_weights.items():
            if node_id in node_id_to_idx:
                idx = node_id_to_idx[node_id]
                pre_trust[idx] = weight
                total_weight += weight
        
        if total_weight <= 0:
            raise ValueError("预信任权重总和必须大于 0")
        
        # 归一化
        pre_trust = pre_trust / total_weight
        
        # 调用主计算函数
        return EigenTrust.compute(network, pre_trust=pre_trust, **kwargs)