"""
simulation.py - 仿真引擎
负责：运行多轮交易、注入攻击
"""

from typing import List
from core.network import TrustNetwork
from core.transaction import Transaction
from config import TRANSACTION_ROUNDS

class Simulation:
    """交易仿真器"""
    
    def __init__(self, network: TrustNetwork):
        self.network = network
        self.transactions: List[Transaction] = []
        
    def run(self, rounds: int = TRANSACTION_ROUNDS):
        """
        运行多轮交易仿真
        
        参数:
            rounds: 交易轮数
        """
        for t in range(rounds):
            tx = self.network.perform_transaction(t)
            self.transactions.append(tx)
    
    def get_transaction_count(self) -> int:
        return len(self.transactions)