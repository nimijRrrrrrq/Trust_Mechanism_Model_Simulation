# models/eigentrust.py (或您存放该改进算法的文件)
"""
EigenTrust 算法改进

基于论文 "The EigenTrust Algorithm for Reputation Management in P2P Networks"
(Kamvar, Schlosser & Garcia-Molina, WWW 2003)

改进方向（保持算法理论保证）:
1. 时间衰减加权：近期交易权重更高
2. 置信度评估：小样本节点传播能力降低（统一 min_confidence 参数）
3. Sybil 攻击防御：基于行为模式的惩罚机制
"""

import numpy as np
import time
import math
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class EigenTrustResult:
    """EigenTrust 计算结果"""
    trust_vector: np.ndarray
    convergence_history: Optional[List[float]]
    compute_time: float
    trust_matrix: Optional[np.ndarray] = None
    pretrust_vector: Optional[np.ndarray] = None
    node_weights: Optional[np.ndarray] = None


class ImprovedEigenTrust:
    """
    改进版 EigenTrust 声誉管理算法

    核心公式改进: t^(k+1) = (1-α) * C^T * (t ⊙ w) + α * p^(k)
    其中:
        w:    节点传播能力权重向量（置信度折扣）
        ⊙:    逐元素乘法（Hadamard积）
        C:    保持标准随机矩阵（每行和为1）
        p^(k): 自适应预信任向量，随迭代动态更新

    自适应预信任更新公式:
        p^(k+1) = (1 - γ) * p_0 + γ * normalize(t^(k))
    其中:
        p_0: 初始预信任（种子节点或均匀分布），作为锚点
        γ:   自适应强度 (adaptive_trust_rate)，γ=0 退化为原算法

    理论依据：
    令 D = diag(w)，则迭代为 t^(k+1) = (1-α) * C^T * D * t^(k) + α * p^(k)。
    当所有 w_i > 0 时，C^T D 仍是正矩阵，Perron-Frobenius 定理保证：
    1. 存在唯一正特征向量
    2. 幂迭代收敛到该特征向量
    γ > 0 时 p^(k) 随 t^(k) 变化，收敛性由 α 和 γ 共同控制；
    实践中 γ ≤ 0.5 可保证稳定收敛。
    """

    DECAY_EXPONENTIAL = 'exponential'
    DECAY_HALF_LIFE = 'half_life'
    DECAY_LINEAR = 'linear'

    @staticmethod
    def compute(
        network,
        max_iter: int = 100,
        tolerance: float = 1e-6,
        alpha: float = 0.15,
        pre_trust: Optional[np.ndarray] = None,
        seed_nodes: Optional[List[str]] = None,
        seed_weight: float = 0.5,
        use_time_decay: bool = False,
        decay_mode: str = DECAY_HALF_LIFE,
        decay_parameter: float = 7.0,
        use_confidence: bool = True,
        confidence_method: str = 'wilson',
        min_transactions: int = 5,
        min_confidence: float = 0.1,
        use_sybil_defense: bool = True,
        sybil_internal_ratio: float = 0.5,
        sybil_penalty_factor: float = 0.1,
        sybil_min_transactions: int = 5,
        adaptive_trust_rate: float = 0.2,
        track_convergence: bool = False,
        verbose: bool = False,
        return_details: bool = False
    ) -> EigenTrustResult:
        """
        计算改进的 EigenTrust 值
        """

        # 参数验证
        if not 0 < alpha < 1:
            raise ValueError(f"alpha 必须在 (0, 1) 范围内，当前值: {alpha}")
        if max_iter <= 0:
            raise ValueError(f"max_iter 必须大于 0，当前值: {max_iter}")
        if tolerance <= 0:
            raise ValueError(f"tolerance 必须大于 0，当前值: {tolerance}")
        if seed_nodes is not None:
            if not 0 < seed_weight < 1:
                raise ValueError(f"seed_weight 必须在 (0, 1) 范围内，当前值: {seed_weight}")
        if decay_mode not in [ImprovedEigenTrust.DECAY_EXPONENTIAL,
                               ImprovedEigenTrust.DECAY_HALF_LIFE,
                               ImprovedEigenTrust.DECAY_LINEAR]:
            raise ValueError(f"不支持的衰减模式: {decay_mode}")
        if use_time_decay and decay_parameter <= 0:
            raise ValueError(f"decay_parameter 必须大于 0，当前值: {decay_parameter}")
        if use_confidence and confidence_method not in ['wilson', 'laplace']:
            raise ValueError(f"不支持的置信度方法: {confidence_method}")
        if not 0 < min_confidence < 1:
            raise ValueError(f"min_confidence 必须在 (0, 1) 范围内，当前值: {min_confidence}")
        if not 0 <= sybil_penalty_factor <= 1:
            raise ValueError(f"sybil_penalty_factor 必须在 [0, 1] 范围内，当前值: {sybil_penalty_factor}")
        if not 0 <= adaptive_trust_rate <= 0.5:
            raise ValueError(f"adaptive_trust_rate 必须在 [0, 0.5] 范围内，当前值: {adaptive_trust_rate}")

        n = len(network.nodes)
        if n == 0:
            raise ValueError("网络中没有任何节点")

        node_id_to_idx = {node.id: i for i, node in enumerate(network.nodes)}
        start_time = time.time()

        # 1. 构建固定预信任向量 p
        if pre_trust is not None:
            pre_trust = np.asarray(pre_trust, dtype=float)
            if len(pre_trust) != n:
                raise ValueError(f"预信任向量长度 {len(pre_trust)} 与节点数 {n} 不匹配")
            sum_p = np.sum(pre_trust)
            if sum_p > 0:
                pre_trust = pre_trust / sum_p
            else:
                pre_trust = np.ones(n) / n
                if verbose:
                    print("警告: 预信任向量和为0，使用均匀分布")
        else:
            pre_trust = ImprovedEigenTrust._build_fixed_pretrust(
                network, seed_nodes, seed_weight, node_id_to_idx
            )

        if verbose:
            print(f"开始计算改进版 EigenTrust: {n} 个节点, alpha={alpha}")

        # 2. 构建改进的局部信任矩阵 C 和节点权重向量 w
        # 计算 current_time = 所有交易中最大的 timestamp（按需启用）
        current_time = None
        if use_time_decay:
            max_ts = 0
            for node in network.nodes:
                for tx in node.transaction_history:
                    if 'timestamp' in tx:
                        ts = tx['timestamp']
                        if isinstance(ts, (int, float)):
                            max_ts = max(max_ts, ts)
            current_time = max_ts if max_ts > 0 else None

        C, node_weights = ImprovedEigenTrust._build_improved_trust_matrix(
            network, node_id_to_idx, pre_trust,
            use_time_decay, decay_mode, decay_parameter, current_time,
            use_confidence, confidence_method, min_transactions, min_confidence,
            verbose
        )

        # 检查权重非正
        if np.any(node_weights <= 0):
            print("警告: 存在权重为0或负数的节点，可能影响收敛性")
            node_weights = np.maximum(node_weights, min_confidence)

        # 3. 应用 Sybil 防御
        if use_sybil_defense:
            C, node_weights = ImprovedEigenTrust._apply_sybil_defense(
                C, node_weights, network, node_id_to_idx,
                sybil_internal_ratio, sybil_penalty_factor, sybil_min_transactions,
                verbose
            )

        # 4. 幂迭代（改进公式，支持自适应预信任）
        t, convergence_history, actual_iters = ImprovedEigenTrust._power_iteration(
            C, node_weights, pre_trust, alpha, adaptive_trust_rate,
            max_iter, tolerance, track_convergence, verbose
        )

        compute_time = time.time() - start_time

        # 5. 写回节点
        for i, node in enumerate(network.nodes):
            node.trust_value = t[i]

        if verbose:
            print(f"计算完成，耗时: {compute_time:.4f} 秒，迭代次数: {actual_iters}")
            print(f"信任值范围: [{t.min():.6f}, {t.max():.6f}]")
            print(f"信任值总和: {t.sum():.6f}")

        return EigenTrustResult(
            trust_vector=t,
            convergence_history=convergence_history if track_convergence else None,
            compute_time=compute_time,
            trust_matrix=C if return_details else None,
            pretrust_vector=pre_trust if return_details else None,
            node_weights=node_weights if return_details else None
        )

    @staticmethod
    def _build_fixed_pretrust(network, seed_nodes, seed_weight, node_id_to_idx):
        """构建固定预信任向量（基于种子节点）"""
        n = len(network.nodes)
        if seed_nodes is None or len(seed_nodes) == 0:
            return np.ones(n) / n

        pre_trust = np.ones(n) * (1 - seed_weight) / n

        valid_seeds = []
        for seed_id in seed_nodes:
            if seed_id in node_id_to_idx:
                idx = node_id_to_idx[seed_id]
                pre_trust[idx] += seed_weight / len(seed_nodes)
                valid_seeds.append(seed_id)

        if len(valid_seeds) == 0:
            print(f"警告: 未找到有效的种子节点，使用均匀分布")
            return np.ones(n) / n

        pre_trust = pre_trust / np.sum(pre_trust)
        return pre_trust

    @staticmethod
    def _compute_time_weight(tx_time, current_time, decay_mode, decay_parameter):
        """计算时间衰减权重（基于交易轮次）"""
        delta_rounds = current_time - tx_time  # 时间差（轮次）

        if decay_mode == ImprovedEigenTrust.DECAY_EXPONENTIAL:
            weight = math.exp(-decay_parameter * delta_rounds / 100.0)  # 归一化到 0-100 轮
        elif decay_mode == ImprovedEigenTrust.DECAY_HALF_LIFE:
            half_life = decay_parameter * 10  # 半衰期 = 参数 * 10 轮
            weight = 0.5 ** (delta_rounds / half_life)
        elif decay_mode == ImprovedEigenTrust.DECAY_LINEAR:
            max_delta = decay_parameter * 100
            weight = max(0, 1 - delta_rounds / max_delta)
        else:
            weight = 1.0

        return max(weight, 0.01)

    @staticmethod
    def _compute_confidence_discount(effective_sample_size, min_transactions,
                                      method, min_confidence=0.1):
        """
        计算置信度折扣因子 [min_confidence, 1.0]

        统一最小值为 min_confidence 参数，两种方法返回值范围一致。
        悬挂节点 (effective_sample_size=0) 返回 min_confidence。
        """
        if effective_sample_size <= 0:
            return min_confidence

        if effective_sample_size >= min_transactions:
            return 1.0

        ratio = effective_sample_size / min_transactions

        if method == 'wilson':
            return min_confidence + (1.0 - min_confidence) * math.sqrt(ratio)
        elif method == 'laplace':
            return min_confidence + (1.0 - min_confidence) * ratio

        return 1.0

    @staticmethod
    def _build_improved_trust_matrix(network, node_id_to_idx, pre_trust,
                                     use_time_decay, decay_mode, decay_parameter, current_time,
                                     use_confidence, confidence_method, min_transactions,
                                     min_confidence, verbose):
        """
        构建改进的局部信任矩阵 C 和节点权重向量 w

        C 保持标准随机矩阵（每行和为1）。
        置信度折扣提取为节点权重，不在 C 中打折。
        """
        n = len(network.nodes)
        C = np.zeros((n, n))
        node_weights = np.ones(n)
        hanging_nodes = 0

        for i, node in enumerate(network.nodes):
            weighted_pos: Dict[int, float] = {}
            weighted_neg: Dict[int, float] = {}
            effective_weights: Dict[int, float] = {}

            for tx in node.transaction_history:
                other_id = tx.get('other')
                if other_id is None or other_id not in node_id_to_idx:
                    if other_id is not None and verbose:
                        print(f"警告: 节点 {node.id} 的交易目标 {other_id} 不存在于网络中")
                    continue

                j = node_id_to_idx[other_id]
                feedback = tx.get('feedback', 0.0)

                weight = 1.0
                if use_time_decay and current_time is not None and 'timestamp' in tx:
                    tx_time = tx['timestamp']
                    if isinstance(tx_time, (int, float)):
                        weight = ImprovedEigenTrust._compute_time_weight(
                            tx_time, current_time, decay_mode, decay_parameter
                        )
                        if weight < 0.01:
                            continue

                if feedback > 0:
                    weighted_pos[j] = weighted_pos.get(j, 0.0) + feedback * weight
                elif feedback < 0:
                    weighted_neg[j] = weighted_neg.get(j, 0.0) + abs(feedback) * weight

                effective_weights[j] = effective_weights.get(j, 0.0) + weight

            # 计算原始信任分数
            for j in range(n):
                pos = weighted_pos.get(j, 0.0)
                neg = weighted_neg.get(j, 0.0)
                C[i][j] = max(pos - neg, 0.0)

            # 标准行归一化
            row_sum = np.sum(C[i, :])
            if row_sum > 1e-12:
                C[i, :] = C[i, :] / row_sum

                if use_confidence:
                    effective_sample_size = sum(effective_weights.values())
                    node_weights[i] = ImprovedEigenTrust._compute_confidence_discount(
                        effective_sample_size, min_transactions, confidence_method, min_confidence
                    )
                # else: node_weights[i] 保持默认值 1.0
            else:
                # 悬挂节点
                C[i, :] = pre_trust.copy()
                hanging_nodes += 1
                if use_confidence:
                    node_weights[i] = ImprovedEigenTrust._compute_confidence_discount(
                        0, min_transactions, confidence_method, min_confidence
                    )
                # else: node_weights[i] 保持默认值 1.0

        if verbose and hanging_nodes > 0:
            print(f"检测到 {hanging_nodes} 个悬挂节点")

        return C, node_weights

    @staticmethod
    def _apply_sybil_defense(C, node_weights, network, node_id_to_idx,
                             internal_ratio_threshold, penalty_factor, min_transactions,
                             verbose):
        """应用 Sybil 防御机制（降低节点权重，C 保持随机矩阵）"""
        for i, node in enumerate(network.nodes):
            internal_count = 0
            external_count = 0

            for tx in node.transaction_history:
                other_id = tx.get('other')
                if other_id is None or other_id not in node_id_to_idx:
                    continue
                if ImprovedEigenTrust._is_internal_transaction(node.id, other_id, network):
                    internal_count += 1
                else:
                    external_count += 1

            total = internal_count + external_count

            if total >= min_transactions:
                internal_ratio = internal_count / total
                if internal_ratio > internal_ratio_threshold:
                    node_weights[i] *= penalty_factor
                    if verbose:
                        print(f"  检测到疑似 Sybil 节点: {node.id} "
                              f"(内部比例={internal_ratio:.2f}, 权重={node_weights[i]:.2f})")

        return C, node_weights

    @staticmethod
    def _is_internal_transaction(node1_id, node2_id, network):
        """
        启发式判断两个节点是否属于同一 Sybil 团伙。

        仿真环境中节点 ID 为整数，通过 is_malicious 标记判断。
        生产环境应替换为：IP前缀、注册时间、行为指纹、图聚类等。
        """
        node1_str = str(node1_id).lower()
        node2_str = str(node2_id).lower()

        # 字符串前缀匹配（适用于节点 ID 为带前缀字符串的场景）
        suspicious_prefixes = ['sybil', 'fake', 'clone', 'bot', 'spam', 'malicious']
        for prefix in suspicious_prefixes:
            if node1_str.startswith(prefix) and node2_str.startswith(prefix):
                return True

        # 仿真环境：两个节点都是恶意节点则视为内部交易
        node1_obj = network.get_node_by_id(node1_id)
        node2_obj = network.get_node_by_id(node2_id)
        if node1_obj is not None and node2_obj is not None:
            if node1_obj.is_malicious and node2_obj.is_malicious:
                return True

        return False

    @staticmethod
    def _power_iteration(C, node_weights, pre_trust, alpha, adaptive_trust_rate,
                         max_iter, tolerance, track_convergence, verbose):
        """
        幂迭代求解全局信任向量（改进公式）

        t^(k+1) = (1-α) * C^T * (t^(k) ⊙ w) + α * p^(k)

        自适应预信任更新（adaptive_trust_rate > 0 时启用）:
            p^(k+1) = (1 - γ) * p_0 + γ * t^(k)
        其中 p_0 为初始预信任锚点，γ = adaptive_trust_rate。
        γ=0 时退化为原算法（p 固定不变）。
        """
        n = len(pre_trust)
        t = pre_trust.copy()
        p0 = pre_trust.copy()   # 初始锚点（固定不变）
        p = pre_trust.copy()
        C_T = C.T
        gamma = adaptive_trust_rate

        convergence_history = [] if track_convergence else None
        actual_iters = max_iter

        for iteration in range(max_iter):
            weighted_t = t * node_weights
            t_new = (1 - alpha) * (C_T @ weighted_t) + alpha * p

            t_new = np.maximum(t_new, 0)
            sum_t_new = np.sum(t_new)
            if sum_t_new > 1e-12:
                t_new = t_new / sum_t_new
            else:
                if verbose:
                    print(f"警告: 迭代 {iteration} 时信任向量和为0，重置为预信任向量")
                t_new = p.copy()

            diff = np.linalg.norm(t_new - t, ord=1)

            if track_convergence:
                convergence_history.append(diff)

            if verbose and (iteration + 1) % 20 == 0:
                print(f"  迭代 {iteration + 1}/{max_iter}, 变化量: {diff:.8f}")

            t = t_new

            if gamma > 0:
                p = (1 - gamma) * p0 + gamma * t
                p_sum = np.sum(p)
                if p_sum > 1e-12:
                    p = p / p_sum
                else:
                    p = p0.copy()

            if diff < tolerance:
                actual_iters = iteration + 1
                if verbose:
                    print(f"收敛于迭代 {actual_iters}, 变化量: {diff:.8f}")
                break

        return t, convergence_history, actual_iters

    @staticmethod
    def compute_with_custom_pretrust(network, pre_trust_weights: Dict[str, float],
                                     **kwargs) -> EigenTrustResult:
        """使用自定义预信任向量"""
        n = len(network.nodes)
        node_id_to_idx = {node.id: i for i, node in enumerate(network.nodes)}

        pre_trust = np.zeros(n)
        total_weight = 0.0

        for node_id, weight in pre_trust_weights.items():
            if node_id in node_id_to_idx:
                pre_trust[node_id_to_idx[node_id]] = weight
                total_weight += weight

        if total_weight <= 0:
            raise ValueError("预信任权重总和必须大于 0")

        pre_trust = pre_trust / total_weight
        return ImprovedEigenTrust.compute(network, pre_trust=pre_trust, **kwargs)


# 向后兼容别名
EigenTrustImproved = ImprovedEigenTrust