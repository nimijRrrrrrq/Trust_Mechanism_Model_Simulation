# experiments/run_attack_tests.py
"""
攻击测试实验 - 测试不同攻击类型下的模型性能（系统一致性修正版）

本实验模拟两类经典的图安全攻击：
1. 信誉操纵女巫攻击 (Reputation Manipulation Sybil Attack)：通过动态注册虚假账号，并结合合规交易操纵评价矩阵。
2. 身份轮换白洗攻击 (Identity Swapping Whitewashing Attack)：通过抛弃高污染旧账号并注册干净新账号来规避惩罚。
"""

import sys
import os
import copy
import random
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.network import TrustNetwork
from core.simulation import Simulation
from models.eigentrust import EigenTrust
from models.eigentrust_improved import ImprovedEigenTrust
from models.peertrust import PeerTrust
from models.pagerank import PageRankTrust
from config import NUM_NODES, TRANSACTION_ROUNDS, REPEAT_TIMES, RANDOM_SEED, EIGENTRUST_MAX_ITER


ATTACK_ALGORITHMS = ('EigenTrust', 'ImprovedEigenTrust', 'PeerTrust', 'PageRank')


def _run_algorithm(algo_name, network):
    """统一运行攻击测试中的信任算法。"""
    if algo_name == 'EigenTrust':
        EigenTrust.compute(network, max_iter=EIGENTRUST_MAX_ITER)
    elif algo_name == 'ImprovedEigenTrust':
        # [TIME_DECAY_DISABLED] 攻击实验统一使用 ImprovedEigenTrust 调强参数
        ImprovedEigenTrust.compute(
            network,
            max_iter=EIGENTRUST_MAX_ITER,
            verbose=False,
            adaptive_trust_rate=0.2,
            min_transactions=5,
            sybil_internal_ratio=0.5,
            sybil_penalty_factor=0.1,
        )
    elif algo_name == 'PeerTrust':
        PeerTrust.compute(network)
    elif algo_name == 'PageRank':
        PageRankTrust.compute(network)
    else:
        raise ValueError(f"未知算法: {algo_name}")


def evaluate(network, true_labels=None) -> tuple:
    """
    评估声誉系统对恶意节点的识别准确率。

    排序法预测：信任值最低的前 num_mal 个节点判为恶意，
    num_mal 由真实标签决定（不再使用固定先验比例）。

    参数:
        network:     TrustNetwork，节点信任值已由算法写回
        true_labels: 可选，长度等于 len(network.nodes) 的 0/1 列表/数组
                     （1=恶意，0=正常）。为 None 时从节点 is_malicious 属性读取。
    """
    nodes = network.nodes
    n_total = len(nodes)

    # 真实标签：优先使用传入的 true_labels，否则从节点属性读取
    if true_labels is not None:
        y_true = np.array(true_labels, dtype=int)
    else:
        y_true = np.array([1 if n.is_malicious else 0 for n in nodes])

    num_mal = int(y_true.sum())
    if num_mal == 0:
        return 1.0, None, None, 1.0, 0.5, 0.0

    # 中性初始值填充，约束到 [0, 1]
    default_trust = 0.5
    trust_values = np.array([
        n.trust_value if n.trust_value is not None else default_trust
        for n in nodes
    ])
    trust_values = np.clip(trust_values, 0.0, 1.0)

    # 排序预测：信任值最低的 num_mal 个预测为恶意
    sorted_indices = np.argsort(trust_values)
    y_pred = np.zeros(n_total)
    y_pred[sorted_indices[:num_mal]] = 1

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, zero_division=0)

    # 无阈值稳健指标（ROC-AUC / PR-AUC）
    y_score = 1.0 - trust_values
    try:
        roc_auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.5
    except Exception:
        roc_auc = 0.5
    try:
        pr_auc = average_precision_score(y_true, y_score)
    except Exception:
        pr_auc = 0.0

    return acc, None, None, f1, roc_auc, pr_auc


def _add_attack_transaction(network, buyer_id: str, seller_id: str, feedback: float, 
                            true_feedback: float, amount: float, timestamp: int):
    """
    向网络中注入一笔攻击交易，同时更新交易历史列表和 NetworkX 拓扑图，确保数据状态同步。
    """
    # 1. 更新买家的交易历史 (供 EigenTrust, PeerTrust 等使用)
    buyer_node = next((n for n in network.nodes if n.id == buyer_id), None)
    if buyer_node is not None:
        buyer_node.transaction_history.append({
            'other': seller_id,
            'feedback': feedback,
            'true_feedback': true_feedback,
            'amount': amount,
            'timestamp': timestamp
        })
    
    # 2. 同步更新 NetworkX 图结构 (供 PageRank 等图算法使用)
    if network.graph is not None:
        if not network.graph.has_node(buyer_id):
            network.graph.add_node(buyer_id)
        if not network.graph.has_node(seller_id):
            network.graph.add_node(seller_id)
        # 建立有向出边（买家指向卖家，代表信任流向）
        network.graph.add_edge(buyer_id, seller_id)


def _whitewash_node_identity(network, old_id: str) -> str:
    """
    模拟洗白节点的身份重置（注册全新的新号凭证）：
    1. 生成一个全新的节点实例，ID 为 f"{old_id}_new"。
    2. 新号的初始 trust_value 设为 None，保持中立初始化（由 evaluate 时动态填充默认值）。
    3. 将新号节点追加至系统节点列表中，系统总节点数与恶意节点总数同步递增。
    4. 在拓扑图中独立注册该新号节点。
    5. 保持老账号及其历史所有负面交易记录和评价在系统中原地保留（老账号继续背锅）。
    """
    old_node = next((n for n in network.nodes if n.id == old_id), None)
    if old_node is not None:
        new_id = f"{old_id}_new"
        
        # 克隆原恶意节点，并重置其属性以生成一个干净且无历史评估关联的新账号
        new_node = copy.copy(old_node)
        new_node.id = new_id
        new_node.transaction_history = []
        new_node.trust_value = None  # 设为 None，保持冷启动的中立评估状态
        
        # 追加进系统全局节点列表（老账号不删除，系统规模动态扩增）
        network.nodes.append(new_node)
        
        # 在拓扑图中同步添加该新号
        if network.graph is not None:
            network.graph.add_node(new_id)
            
        return new_id
    return old_id


def test_sybil_attack():
    """
    测试女巫攻击下的模型鲁棒性（信誉操纵攻击类型）

    威胁模型设计：
    1. 真实女巫注入 (Sybil Generation)：攻击者注册 sybil_size 个全新的独立女巫节点加入系统。
    2. 女巫内部协同 (Sybil-to-Sybil Loop)：女巫节点两两互刷好评（feedback=1），抬高整体信用评分。
    3. 信誉定向输送 (Sybil-to-Primary Boost)：女巫节点向系统既有的主恶意节点（malicious_nodes）刷正反馈，将膨胀后的信用流向核心收割端。
    4. 诱导外部背书 (Honest Leakage)：随机欺骗 10% 的诚实节点，使其随机背书 1~2 个女巫节点，使虚假信用群接入合法图拓扑。
    """
    print("\n" + "=" * 60)
    print("Sybil 攻击测试（排序法评估）")
    print("=" * 60)

    results = {
        'accuracy': {name: [] for name in ATTACK_ALGORITHMS},
        'f1':       {name: [] for name in ATTACK_ALGORITHMS},
        'roc_auc':  {name: [] for name in ATTACK_ALGORITHMS},
        'pr_auc':   {name: [] for name in ATTACK_ALGORITHMS},
    }

    # 保证每项实验具有至少 20 轮实验的重复次数，控制随机性
    local_repeats = max(20, REPEAT_TIMES)

    for sybil_size in [2, 4, 6, 8, 10]:
        print(f"\nSybil 团伙规模: {sybil_size}")

        for algo_name in results['accuracy'].keys():
            acc_sum = f1_sum = roc_sum = pr_sum = 0

            for repeat in range(local_repeats):
                # 严格重置种子以确保每轮实验环境和结果高度可复现
                np.random.seed(RANDOM_SEED + repeat)
                random.seed(RANDOM_SEED + repeat)
                network = TrustNetwork(
                    NUM_NODES,
                    attack_ratio=0.2
                )

                sim = Simulation(network)
                sim.run(TRANSACTION_ROUNDS)

                malicious_nodes = [n for n in network.nodes if n.is_malicious]
                honest_nodes = [n for n in network.nodes if not n.is_malicious]
                
                # Step 1: 真实注册 Sybil 女巫节点，并追加至全局，实现网络身份的真实膨胀
                sybils = []
                for i in range(sybil_size):
                    ref_node = malicious_nodes[0] if malicious_nodes else honest_nodes[0]
                    sybil_node = copy.copy(ref_node)
                    sybil_node.id = f"sybil_{i}"
                    sybil_node.is_malicious = True
                    sybil_node.trust_value = None  # 设为 None，保持冷启动的中立评估状态
                    sybil_node.transaction_history = []
                    
                    network.nodes.append(sybil_node)
                    if network.graph is not None:
                        network.graph.add_node(sybil_node.id)
                    sybils.append(sybil_node)

                # Step 2: 注入内部协同抬升（Sybils 节点以 30% 概率两两概率性互刷正反馈）
                for sybil_a in sybils:
                    for sybil_b in sybils:
                        if sybil_a.id != sybil_b.id:
                            if random.random() < 0.3:
                                _add_attack_transaction(
                                    network,
                                    buyer_id=sybil_a.id,
                                    seller_id=sybil_b.id,
                                    feedback=1.0,
                                    true_feedback=-1.0,  # 物理交易行为为欺诈
                                    amount=100.0,
                                    timestamp=TRANSACTION_ROUNDS - 1
                                )

                # Step 3: 信誉定向输送（Sybils 将积累的虚假信用导向原始的主恶意节点）
                for sybil_node in sybils:
                    # 假定攻击者有 2 个主要收割端
                    target_primaries = malicious_nodes[:2]
                    for primary_mal in target_primaries:
                        _add_attack_transaction(
                            network,
                            buyer_id=sybil_node.id,
                            seller_id=primary_mal.id,
                            feedback=1.0,
                            true_feedback=-1.0,
                            amount=100.0,
                            timestamp=TRANSACTION_ROUNDS - 1
                        )

                # Step 4: 诱导诚实背书（随机蒙蔽 10% 的诚实节点，使其随机评价 1~2 个 Sybil 节点）
                num_tricked = max(1, int(len(honest_nodes) * 0.10))
                tricked_honest = random.sample(honest_nodes, num_tricked)
                for honest_node in tricked_honest:
                    target_sybils = random.sample(
                        sybils,
                        min(random.randint(1, 2), len(sybils))
                    )
                    for sybil_node in target_sybils:
                        _add_attack_transaction(
                            network,
                            buyer_id=honest_node.id,
                            seller_id=sybil_node.id,
                            feedback=1.0,
                            true_feedback=-1.0,
                            amount=100.0,
                            timestamp=TRANSACTION_ROUNDS - 1
                        )

                _run_algorithm(algo_name, network)

                # 用实际 is_malicious 标记构建真实标签（含动态新增的 Sybil 节点）
                true_labels = [1 if n.is_malicious else 0 for n in network.nodes]
                acc, _, _, f1, roc_auc, pr_auc = evaluate(network, true_labels=true_labels)
                acc_sum += acc
                f1_sum += f1
                roc_sum += roc_auc
                pr_sum += pr_auc

            results['accuracy'][algo_name].append(acc_sum / local_repeats)
            results['f1'][algo_name].append(f1_sum / local_repeats)
            results['roc_auc'][algo_name].append(roc_sum / local_repeats)
            results['pr_auc'][algo_name].append(pr_sum / local_repeats)
            print(f"  {algo_name}: ROC-AUC={roc_sum/local_repeats:.4f}, PR-AUC={pr_sum/local_repeats:.4f} | (Ref: Acc={acc_sum/local_repeats:.4f}, F1={f1_sum/local_repeats:.4f})")

    return results


def test_whitewashing_attack():
    """
    测试白洗攻击（洗白-重建-重新作恶 完整闭环测试）

    攻击原理：
    1. 注册新凭证 (Identity Swapping)：洗白节点注册并使用全新的账号凭证（new_id），其历史恶意评价原样留存在废弃账号上（老账号继续背锅），切断坏账关联。
    2. 声誉重建 (Rehab)：新号通过 15 次小额合规交易骗取诚实节点的好评（feedback=1.0, true_feedback=1.0），快速堆叠基础声誉。
    3. 重新作恶 (Re-offend)：利用恢复的虚假良好声誉重新骗取信任，在仿真结束前再次发动 5 次大额欺诈行为（feedback=-1.0, true_feedback=-1.0），获取暴利。
    """
    print("\n" + "=" * 60)
    print("白洗攻击测试（排序法评估）")
    print("=" * 60)

    results = {
        'accuracy': {name: [] for name in ATTACK_ALGORITHMS},
        'f1':       {name: [] for name in ATTACK_ALGORITHMS},
        'roc_auc':  {name: [] for name in ATTACK_ALGORITHMS},
        'pr_auc':   {name: [] for name in ATTACK_ALGORITHMS},
    }

    # 保证每项实验具有至少 20 轮实验的重复次数，控制随机性
    local_repeats = max(20, REPEAT_TIMES)

    for whitewash_rate in [0.05, 0.10, 0.15, 0.20, 0.25]:
        print(f"\n白洗率: {whitewash_rate * 100:.0f}%")

        for algo_name in results['accuracy'].keys():
            acc_sum = f1_sum = roc_sum = pr_sum = 0
            for repeat in range(local_repeats):
                np.random.seed(RANDOM_SEED + repeat)
                random.seed(RANDOM_SEED + repeat)
                network = TrustNetwork(
                    NUM_NODES,
                    attack_ratio=0.2
                )

                sim = Simulation(network)
                sim.run(TRANSACTION_ROUNDS)

                malicious_nodes = [n for n in network.nodes if n.is_malicious]
                honest_nodes = [n for n in network.nodes if not n.is_malicious]
                num_whitewash = max(1, int(len(malicious_nodes) * whitewash_rate))

                # 拷贝待攻击的恶意节点引用，防止就地添加新节点时导致迭代逻辑出现不一致
                target_malicious = list(malicious_nodes[:num_whitewash])

                for node in target_malicious:
                    old_id = node.id
                    # 1. 模拟身份重置：新建新号凭证（trust_value 为 None），旧号在全局的历史交易和评价原样保留，拒绝全网失忆
                    new_id = _whitewash_node_identity(network, old_id)
                    
                    # 2. 模拟声誉积累（中期）：洗白后的新账号通过 15 次小额合规交易重新获取好评
                    for _ in range(15):
                        honest_node = random.choice(honest_nodes)
                        
                        # 引入时间戳安全区间约束，保证时序关系严密
                        recent_ts = min(max(TRANSACTION_ROUNDS - np.random.randint(11, 51), 0), TRANSACTION_ROUNDS)
                        _add_attack_transaction(
                            network,
                            buyer_id=honest_node.id,
                            seller_id=new_id,          # 针对洗白后的新凭证进行交易
                            feedback=1.0,
                            true_feedback=1.0,         # 通过真实的诚实履约堆叠好评
                            amount=10.0,
                            timestamp=recent_ts
                        )
                    
                    # 3. 模拟二次作恶（末期）：利用刚刚重建的良好声誉，再次进行 5 次大额欺诈交易获取暴利
                    for _ in range(5):
                        honest_node = random.choice(honest_nodes)
                        
                        # 安全时间戳修正：临近仿真结束的最晚时段，并保证不越界
                        latest_ts = min(max(TRANSACTION_ROUNDS - np.random.randint(1, 10), 0), TRANSACTION_ROUNDS)
                        _add_attack_transaction(
                            network,
                            buyer_id=honest_node.id,
                            seller_id=new_id,          # 针对洗白后的新凭证进行欺诈
                            feedback=-1.0,             # 恶意欺诈行为
                            true_feedback=-1.0,
                            amount=100.0,
                            timestamp=latest_ts
                        )

                _run_algorithm(algo_name, network)

                # 用实际 is_malicious 标记构建真实标签（含白洗新增节点）
                true_labels = [1 if n.is_malicious else 0 for n in network.nodes]
                acc, _, _, f1, roc_auc, pr_auc = evaluate(network, true_labels=true_labels)
                acc_sum += acc
                f1_sum += f1
                roc_sum += roc_auc
                pr_sum += pr_auc

            results['accuracy'][algo_name].append(acc_sum / local_repeats)
            results['f1'][algo_name].append(f1_sum / local_repeats)
            results['roc_auc'][algo_name].append(roc_sum / local_repeats)
            results['pr_auc'][algo_name].append(pr_sum / local_repeats)
            print(f"  {algo_name}: ROC-AUC={roc_sum/local_repeats:.4f}, PR-AUC={pr_sum/local_repeats:.4f} | (Ref: Acc={acc_sum/local_repeats:.4f}, F1={f1_sum/local_repeats:.4f})")

    return results


def generate_attack_plots(sybil_results, whitewash_results):
    """生成攻击测试图表"""
    try:
        from utils.visualization import plot_attack_comparison

        output_dir = 'results/figures'

        if sybil_results:
            plot_attack_comparison(sybil_results['accuracy'], output_dir, attack_type='Sybil', attack_params=[2, 4, 6, 8, 10], metric_name='Accuracy')
            plot_attack_comparison(sybil_results['f1'], output_dir, attack_type='Sybil', attack_params=[2, 4, 6, 8, 10], metric_name='F1')
            plot_attack_comparison(sybil_results['roc_auc'], output_dir, attack_type='Sybil', attack_params=[2, 4, 6, 8, 10], metric_name='ROC AUC')
            plot_attack_comparison(sybil_results['pr_auc'], output_dir, attack_type='Sybil', attack_params=[2, 4, 6, 8, 10], metric_name='PR AUC')

        if whitewash_results:
            plot_attack_comparison(whitewash_results['accuracy'], output_dir, attack_type='Whitewashing', attack_params=[0.05, 0.10, 0.15, 0.20, 0.25], metric_name='Accuracy')
            plot_attack_comparison(whitewash_results['f1'], output_dir, attack_type='Whitewashing', attack_params=[0.05, 0.10, 0.15, 0.20, 0.25], metric_name='F1')
            plot_attack_comparison(whitewash_results['roc_auc'], output_dir, attack_type='Whitewashing', attack_params=[0.05, 0.10, 0.15, 0.20, 0.25], metric_name='ROC AUC')
            plot_attack_comparison(whitewash_results['pr_auc'], output_dir, attack_type='Whitewashing', attack_params=[0.05, 0.10, 0.15, 0.20, 0.25], metric_name='PR AUC')

        print(f"\n攻击测试图表已保存到 {output_dir}")

    except ImportError as e:
        print(f"警告: 无法导入可视化模块: {e}")
    except Exception as e:
        print(f"警告: 生成图表时出错: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='运行攻击测试实验')
    parser.add_argument('--plot', action='store_true', help='生成实验图表')
    
    args = parser.parse_args()
    
    print("攻击测试实验")
    sybil_results = test_sybil_attack()
    whitewash_results = test_whitewashing_attack()
    
    if args.plot:
        generate_attack_plots(sybil_results, whitewash_results)