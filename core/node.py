"""
node.py - 网络节点（改进版）

支持：
1. 信任传播模型
2. Sybil / Whitewashing 攻击
3. 多算法兼容
"""

from typing import List, Dict, Any


class Node:
    """信任网络节点（增强版）"""

    def __init__(self, node_id: int, is_malicious: bool = False):

        self.id = node_id
        self.is_malicious = is_malicious

        # =====================================================
        # 信任值（统一到 [0,1]）
        # =====================================================
        self.trust_value = 0.5

        # =====================================================
        # 行为分数（可用于扩展模型）
        # =====================================================
        self.behavior_score = 1.0

        # =====================================================
        # 交易历史
        # =====================================================
        self.transaction_history: List[Dict[str, Any]] = []

        # =====================================================
        # 白洗计数（新增）
        # =====================================================
        self.reset_count = 0

    # =========================================================
    # 添加交易
    # =========================================================
    def add_transaction(self, transaction: Dict[str, Any]):
        self.transaction_history.append(transaction)

    # =========================================================
    # 最近交易
    # =========================================================
    def get_recent_transactions(self, window_size: int):

        return self.transaction_history[-window_size:]

    # =========================================================
    # 白洗攻击（关键修改点）
    # =========================================================
    def reset(self):
        """
        白洗攻击版本 reset：

        模拟：
        - 删除历史
        - 重新注册身份
        - 保留攻击属性
        """

        self.transaction_history = []

        # 信任重置
        self.trust_value = 0.5

        # 行为略微惩罚（关键！！）
        self.behavior_score *= 0.95

        # 记录白洗次数
        self.reset_count += 1

        # 如果多次白洗 → 降初始可信度
        if self.reset_count > 1:
            self.trust_value = 0.4

    # =========================================================
    # Sybil 初始化辅助（新增）
    # =========================================================
    def clone_as_sybil(self, new_id: int):
        """
        创建 Sybil 节点（用于攻击实验）
        """
        return Node(
            node_id=new_id,
            is_malicious=True
        )

    # =========================================================
    # 调试输出
    # =========================================================
    def __repr__(self):

        return (
            f"Node(id={self.id}, "
            f"malicious={self.is_malicious}, "
            f"trust={self.trust_value:.3f}, "
            f"reset={self.reset_count})"
        )