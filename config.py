"""
config.py - 所有实验参数的配置文件
修改这里的参数可以控制整个实验的行为
"""

import os

# ==================== 数据集路径配置 ====================
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BTC_OTC_PATH = os.path.join(_BASE_DIR, 'data/soc-sign-bitcoinotc.csv/soc-sign-bitcoinotc.csv')

DATA_PATHS = {
    'epinions': os.path.join(_BASE_DIR, 'data/soc-Epinions1.txt/soc-Epinions1.txt'),
    'filmtrust_ratings': os.path.join(_BASE_DIR, 'data/filmtrust/rating/'),
    'filmtrust_trust': os.path.join(_BASE_DIR, 'data/filmtrust/trust/trust.txt'),
    'bitcoin_otc': BTC_OTC_PATH,
}

# ==================== 网络参数 ====================
NUM_NODES = 200              # 节点总数（可调范围：100-1000）
GRAPH_TYPE = "dataset"       # 网络类型: "random", "small_world", "dataset"
DEGREE = 5                   # 平均度数（每个节点连接几个其他节点）
DATASET_PATH = DATA_PATHS['epinions']  # 默认数据集路径

# ==================== 交易参数 ====================
TRANSACTION_ROUNDS = 2000    # 仿真轮数（每节点平均10笔交易）
TRANSACTIONS_PER_ROUND = 1   # 每轮交易次数（通常为1）

# ==================== 攻击参数 ====================
ATTACK_RATIOS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  # 测试的恶意节点比例（最高60%）
SYBIL_GROUP_SIZE = 5         # 每个Sybil团伙控制的节点数
WHITEWASH_PROB = 0.1         # 白洗攻击概率

# ==================== 算法参数 ====================
# EigenTrust
EIGENTRUST_MAX_ITER = 50     # 最大迭代次数
EIGENTRUST_TOLERANCE = 1e-6  # 收敛阈值

# PageRank
PAGERANK_DAMPING = 0.85      # 阻尼系数

# ==================== 实验参数 ====================
REPEAT_TIMES = 5             # 每组参数重复次数（降低以加速）
RANDOM_SEED = 42             # 随机种子（保证可复现）

# ==================== 评估阈值 ====================
MALICIOUS_THRESHOLD = 0.5    # 信任值低于此值判定为恶意