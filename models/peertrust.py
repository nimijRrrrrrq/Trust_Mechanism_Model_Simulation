import numpy as np
import time
from core.network import TrustNetwork


class PeerTrust:

    @staticmethod
    def compute(
        network: TrustNetwork,
        beta: float = 0.1,
        max_iter: int = 50,
        epsilon: float = 1e-6,
        default_trust: float = 0.5,
        buyer_key: str = "buyer",
        seller_key: str = "seller",
        amount_key: str = "amount",
        feedback_key: str = "feedback",
        track_convergence: bool = False
    ):
        start_time = time.time()
        nodes = network.nodes
        n = len(nodes)

        node2idx = {node.id: i for i, node in enumerate(nodes)}

        # -------------------------
        # 初始化 T
        # -------------------------
        T = np.array([
            node.trust_value if node.trust_value is not None else default_trust
            for node in nodes
        ])

        # -------------------------
        # 去重交易，记录是否有真实反馈
        # feedback 归一化：原始值在 [-1, 1]，映射到 [0, 1]
        # -------------------------
        unique_txs = []
        seen = set()

        for node in nodes:
            for tx in node.transaction_history:
                b = node.id
                s = tx.get("other")
                if s is None:
                    continue
                t = tx.get("timestamp", 0)
                a = tx.get(amount_key, 0.0)

                sig = (b, s, t, a)
                if sig in seen:
                    continue
                seen.add(sig)

                raw = tx.get(feedback_key)
                has_feedback = raw is not None
                S = (raw + 1.0) / 2.0 if has_feedback else default_trust

                unique_txs.append({
                    "buyer": b,
                    "seller": s,
                    "amount": a,
                    "S": S,
                    "has_feedback": has_feedback
                })

        # -------------------------
        # 预计算：seller_total_trans / seller_buyer_trans / seller_feed_count
        # -------------------------
        seller_total_trans: dict = {}
        seller_buyer_trans: dict = {}
        seller_feed_count: dict = {}

        for tx in unique_txs:
            s = tx["seller"]
            b = tx["buyer"]
            seller_total_trans[s] = seller_total_trans.get(s, 0) + 1
            if s not in seller_buyer_trans:
                seller_buyer_trans[s] = {}
            seller_buyer_trans[s][b] = seller_buyer_trans[s].get(b, 0) + 1
            if tx["has_feedback"]:
                seller_feed_count[s] = seller_feed_count.get(s, 0) + 1

        # CF[i] = 卖家 nodes[i] 收到反馈的比例
        CF = np.zeros(n)
        for i, node in enumerate(nodes):
            total = seller_total_trans.get(node.id, 0)
            if total > 0:
                CF[i] = seller_feed_count.get(node.id, 0) / total
            else:
                CF[i] = default_trust

        # -------------------------
        # 收敛历史
        # -------------------------
        convergence_history = []

        # -------------------------
        # 迭代
        # -------------------------
        for _ in range(max_iter):

            T_new = np.zeros(n)

            for i, node in enumerate(nodes):

                seller_txs = [tx for tx in unique_txs if tx["seller"] == node.id]
                I_u = seller_total_trans.get(node.id, 0)

                if not seller_txs or I_u == 0:
                    T_basic = default_trust
                else:
                    num = 0.0
                    den = 0.0

                    for tx in seller_txs:
                        S = tx["S"]
                        rater = tx["buyer"]

                        # Cr(p_i)：评价者的当前信任值
                        Cr = T[node2idx[rater]] if rater in node2idx else default_trust
                        Cr = max(Cr, 1e-6)

                        # TF(i,u) = Trans(p_i, u) / I(u)
                        # Trans(p_i, u)：买家 rater 对卖家 u 的交易次数
                        trans_pi_u = seller_buyer_trans.get(node.id, {}).get(rater, 0)
                        TF = trans_pi_u / I_u

                        num += S * Cr * TF
                        den += Cr * TF

                    T_basic = num / den if den > 0 else default_trust

                # PeerTrust 主公式：T(u) = (1-β)·T_basic + β·CF(u)
                # 数值稳定性保护：确保信任值非负
                T_new[i] = max((1 - beta) * T_basic + beta * CF[i], 0.0)

            # 收敛检测：使用 L1 范数（与 EigenTrust/PageRank 一致）
            diff = np.linalg.norm(T_new - T, ord=1)

            if track_convergence:
                # 最小值保护：避免残差为 0（在对数坐标图上无法显示）
                convergence_history.append(max(diff, 1e-8))

            T = T_new

            if diff < epsilon:
                break

        # 写回
        for i, node in enumerate(nodes):
            node.trust_value = T[i]

        compute_time = time.time() - start_time

        if track_convergence:
            return T, convergence_history, compute_time

        return T, compute_time
