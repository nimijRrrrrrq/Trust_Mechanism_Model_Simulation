"""
transaction.py - 定义交易记录
每笔交易包含：
买卖双方、反馈、金额、时间戳
"""

from dataclasses import dataclass

@dataclass
class Transaction:
    """交易记录"""

    buyer_id: int
    seller_id: int

    # 反馈：
    # 1  = 好评
    # -1 = 差评
    feedback: int

    # 真实反馈：
    # 1  = 正常行为
    # -1 = 恶意行为
    true_feedback: int

    amount: float
    timestamp: int

    def __repr__(self) -> str:
        return (
            f"Transaction("
            f"buyer={self.buyer_id}, "
            f"seller={self.seller_id}, "
            f"feedback={self.feedback}, "
            f"amount={self.amount:.2f}"
            f")"
        )