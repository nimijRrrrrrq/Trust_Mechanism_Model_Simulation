# experiments/run_ablation.py
"""
消融实验 - 验证 ImprovedEigenTrust 各模块的独立贡献（系统一致性修正版 v2）

对齐说明（修复 vs models/eigentrust_improved.py 的不一致）：
  - α 参数：使用 0.15（与 ImprovedEigenTrust 默认值一致；旧版用 0.85）
  - 模块开关：严格对应 ImprovedEigenTrust.compute() 的 4 个改进模块
      * use_time_decay
      * use_confidence
      * use_sybil_defense
      * adaptive_trust_rate
  - 时间衰减：复用 ImprovedEigenTrust._compute_time_weight()（基于交易轮次；
      旧版自定义的 DECAY 系数已废弃，避免与算法主体出现两套语义）
  - AblationEigenTrust 不再维护独立的算法实现，统一调用
    ImprovedEigenTrust.compute()，仅通过 enabled_modules 切换模块开关

支持的 6 种变体：
  1. EigenTrust (标准)              - 全部模块关闭
  2. + 时间衰减                     - 仅 use_time_decay
  3. + 置信度折扣                   - 仅 use_confidence
  4. + Sybil 防御                   - 仅 use_sybil_defense
  5. + 自适应预信任                 - 仅 adaptive_trust_rate
  6. EigenTrust (完整改进)           - 全部模块启用

支持三个实验场景：
  1. 'sybil'        - 仅注入 Sybil 攻击
  2. 'whitewashing' - 仅注入白洗攻击
  3. 'hybrid'       - 混合攻击
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
from core.network import TrustNetwork
from core.simulation import Simulation
from config import NUM_NODES, TRANSACTION_ROUNDS, ATTACK_RATIOS, REPEAT_TIMES, RANDOM_SEED
from models.eigentrust_improved import ImprovedEigenTrust
from experiments.test_improved_eigentrust import _evaluate_rank_based as evaluate_rank_based


class AblationEigenTrust:
    """
    消融实验用的 EigenTrust 变体（系统一致性修正版）

    本类为薄包装（thin wrapper），不再维护独立的算法实现，
    严格通过 enabled_modules 字典控制 ImprovedEigenTrust 的 4 个改进模块开关：

        'time_decay'        -> use_time_decay
        'confidence'        -> use_confidence
        'sybil_defense'     -> use_sybil_defense
        'adaptive_pretrust' -> adaptive_trust_rate (启用时 = ADAPTIVE_TRUST_RATE)

    其他参数（alpha、max_iter、tolerance、track_convergence 等）
    直接透传给 ImprovedEigenTrust.compute()，保持与主实验完全一致。

    返回值：与 ImprovedEigenTrust.compute() 一致（EigenTrustResult 数据类）。
    注意：ImprovedEigenTrust.compute() 内部已将最终信任向量写回
    network.nodes[i].trust_value，因此调用方无需再处理返回值即可
    通过 _evaluate_rank_based(network) 评估。
    """

    # 与 ImprovedEigenTrust 文档说明保持一致（γ ≤ 0.5 可保证稳定收敛）
    ADAPTIVE_TRUST_RATE = 0.3

    @staticmethod
    def compute(
        network: TrustNetwork,
        enabled_modules: dict = None,
        alpha: float = 0.15,
        max_iter: int = 100,
        tolerance: float = 1e-6,
        track_convergence: bool = False,
    ):
        if enabled_modules is None:
            enabled_modules = {}

        return ImprovedEigenTrust.compute(
            network,
            max_iter=max_iter,
            tolerance=tolerance,
            alpha=alpha,  # 0.15，与 ImprovedEigenTrust 默认一致
            use_time_decay=enabled_modules.get('time_decay', False),
            use_confidence=enabled_modules.get('confidence', False),
            use_sybil_defense=enabled_modules.get('sybil_defense', False),
            adaptive_trust_rate=(
                AblationEigenTrust.ADAPTIVE_TRUST_RATE
                if enabled_modules.get('adaptive_pretrust', False)
                else 0.0
            ),
            track_convergence=track_convergence,
        )


def run_ablation_experiment(scenario='hybrid', verbose=False):
    """
    运行指定场景下的消融实验

    6 种变体严格对齐 ImprovedEigenTrust 的 4 个改进模块（enabled_modules 字典）：
        1. EigenTrust (标准)              - {}                                  全部关闭
        2. + 时间衰减                     - {'time_decay': True}
        3. + 置信度折扣                   - {'confidence': True}
        4. + Sybil 防御                   - {'sybil_defense': True}
        5. + 自适应预信任                 - {'adaptive_pretrust': True}
        6. EigenTrust (完整改进·关闭时间衰减)  - 3 模块，= main.py 实际版本
        7. EigenTrust (完整改进·开启时间衰减)  - 4 模块，反例假设版本
    """
    variants = {
        'EigenTrust (标准)': {},
        'EigenTrust + 时间衰减': {'time_decay': True},
        'EigenTrust + 置信度折扣': {'confidence': True},
        'EigenTrust + Sybil防御': {'sybil_defense': True},
        'EigenTrust + 自适应预信任': {'adaptive_pretrust': True},
        # ⭐ 关键：两个"完整改进"版本直接对比
        # "关闭时间衰减"= main.py 实际运行的版本（3 模块）
        'EigenTrust (完整改进·关闭时间衰减)': {
            'confidence': True,
            'sybil_defense': True,
            'adaptive_pretrust': True,
        },
        # "开启时间衰减"= 假设版本（4 模块），用于反例消融
        'EigenTrust (完整改进·开启时间衰减)': {
            'time_decay': True,
            'confidence': True,
            'sybil_defense': True,
            'adaptive_pretrust': True,
        },
    }

    results = {name: [] for name in variants.keys()}

    if verbose:
        print("\n" + "=" * 70)
        print(f"消融实验 - 场景: [{scenario.upper()}]")
        print("=" * 70)

    for attack_ratio in ATTACK_RATIOS:
        if verbose:
            print(f"▶ 攻击比例: {attack_ratio*100:.0f}%")

        for variant_name, enabled_modules in variants.items():
            acc_sum = 0
            f1_sum = 0

            for repeat in range(REPEAT_TIMES):
                np.random.seed(RANDOM_SEED + repeat)
                random.seed(RANDOM_SEED + repeat)
                network = TrustNetwork(NUM_NODES, attack_ratio)
                sim = Simulation(network)
                sim.run(TRANSACTION_ROUNDS)

                # =========================================================
                # 注入单点或混合攻击（确保图结构与历史记录一致，避免系统冲突）
                # =========================================================
                malicious_nodes = [n for n in network.nodes if n.is_malicious]
                sybil_size = 0

                # 1. 注入 Sybil 攻击（timestamp=0 模拟集中在历史某时刻的大量伪造好评，
                #    这正是 Sybil 攻击的典型特征，时间衰减应能识别并压制此类行为）
                if scenario in ['sybil', 'hybrid']:
                    sybil_size = min(6, len(malicious_nodes))
                    sybil_group = malicious_nodes[:sybil_size]
                    for node in sybil_group:
                        for other in sybil_group:
                            if node.id == other.id:
                                continue
                            node.transaction_history.append({
                                'other': other.id,
                                'feedback': 1,
                                'true_feedback': 1,
                                'amount': 100,
                                'timestamp': 0
                            })

                # 2. 注入白洗攻击（只清空个人历史，重置初始值，保证数据流一致）
                if scenario in ['whitewashing', 'hybrid']:
                    start_idx = sybil_size if scenario == 'hybrid' else 0
                    num_whitewash = int(len(malicious_nodes) * 0.20)
                    whitewash_group = malicious_nodes[start_idx : start_idx + num_whitewash]
                    
                    for node in whitewash_group:
                        node.transaction_history.clear()
                        node.trust_value = 0.5

                # =========================================================
                # 运行对应的消融变体（薄包装 -> ImprovedEigenTrust.compute()）
                # =========================================================
                AblationEigenTrust.compute(network, enabled_modules=enabled_modules)

                acc, _, _, f1 = evaluate_rank_based(network)
                acc_sum += acc
                f1_sum += f1

            avg_acc = acc_sum / REPEAT_TIMES
            avg_f1 = f1_sum / REPEAT_TIMES
            results[variant_name].append(avg_acc)

            if verbose:
                print(f"  {variant_name:30s} | 准确率: {avg_acc:.4f} | F1: {avg_f1:.4f}")

    contribution = analyze_contributions(results)
    return results, contribution


def analyze_contributions(results):
    """
    分析各模块的独立贡献（4 个模块全独立量化）+ 完整改进版的两种时间衰减设置对比

    归因公式（与消融学界惯例一致）：
        module_contribution = (only_module - baseline) * 100
        total_improvement   = (full - baseline) * 100
        synergy_effect      = (full - baseline
                                - sum(module_contributions)) * 100

    完整版对比：
        time_decay_marginal_in_full = (full_with_td - full_no_td) * 100
        含义：在完整改进版中，时间衰减的边际贡献
    """
    baseline  = np.array(results['EigenTrust (标准)'])
    only_td   = np.array(results['EigenTrust + 时间衰减'])
    only_conf = np.array(results['EigenTrust + 置信度折扣'])
    only_syb  = np.array(results['EigenTrust + Sybil防御'])
    only_adp  = np.array(results['EigenTrust + 自适应预信任'])
    full_no_td   = np.array(results['EigenTrust (完整改进·关闭时间衰减)'])
    full_with_td = np.array(results['EigenTrust (完整改进·开启时间衰减)'])

    contribution = {
        'attack_ratios': ATTACK_RATIOS,
        'baseline_accuracy':               baseline.tolist(),
        'time_decay_only_accuracy':        only_td.tolist(),
        'confidence_only_accuracy':        only_conf.tolist(),
        'sybil_defense_only_accuracy':     only_syb.tolist(),
        'adaptive_pretrust_only_accuracy': only_adp.tolist(),
        'full_improved_accuracy':          full_no_td.tolist(),         # 主版本（关闭TD）
        'full_with_td_accuracy':           full_with_td.tolist(),        # 反例假设（开启TD）

        'time_decay_contribution':        ((only_td   - baseline) * 100).tolist(),
        'confidence_contribution':        ((only_conf - baseline) * 100).tolist(),
        'sybil_defense_contribution':     ((only_syb  - baseline) * 100).tolist(),
        'adaptive_pretrust_contribution': ((only_adp  - baseline) * 100).tolist(),
        'total_improvement':              ((full_no_td - baseline) * 100).tolist(),  # 默认报告 3 模块版本
        'total_improvement_with_td':      ((full_with_td - baseline) * 100).tolist(),  # 4 模块版本
        # ⭐ 关键：在完整改进版中加入时间衰减的边际贡献（反例消融的核心证据）
        'time_decay_marginal_in_full':    ((full_with_td - full_no_td) * 100).tolist(),
        'synergy_effect': (
            (full_no_td - baseline
             - (only_td   - baseline)
             - (only_conf - baseline)
             - (only_syb  - baseline)
             - (only_adp  - baseline)) * 100
        ).tolist(),
    }
    return contribution


def print_ablation_summary(contribution, scenario='hybrid'):
    """
    打印消融实验的归因表格（4 个模块独立贡献 + 协同效应 + 总提升 + 完整版 TD 边际对比）
    """
    print("\n" + "=" * 70)
    print(f"消融实验结果分析 - 场景: [{scenario.upper()}]")
    print("=" * 70)

    print("\n[RESULT] 各攻击强度下各模块的独立贡献（百分点）:")
    header = (
        f"  {'攻击比例':<8}"
        f"{'时间衰减':<10}"
        f"{'置信度折扣':<12}"
        f"{'Sybil防御':<11}"
        f"{'自适应预信任':<14}"
        f"{'协同效应':<10}"
        f"{'总提升(关TD)':<14}"
        f"{'总提升(开TD)':<14}"
        f"{'完整版TD边际':<14}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for i, ratio in enumerate(contribution['attack_ratios']):
        td   = contribution['time_decay_contribution'][i]
        cf   = contribution['confidence_contribution'][i]
        sy   = contribution['sybil_defense_contribution'][i]
        ap   = contribution['adaptive_pretrust_contribution'][i]
        syn  = contribution['synergy_effect'][i]
        tot  = contribution['total_improvement'][i]
        tot2 = contribution['total_improvement_with_td'][i]
        marg = contribution['time_decay_marginal_in_full'][i]

        print(
            f"  {int(ratio*100):>5}%   "
            f"{td:>+8.2f}  "
            f"{cf:>+10.2f}  "
            f"{sy:>+9.2f}  "
            f"{ap:>+12.2f}  "
            f"{syn:>+8.2f}  "
            f"{tot:>+12.2f}  "
            f"{tot2:>+12.2f}  "
            f"{marg:>+12.2f}"
        )

    avg_td   = np.mean(contribution['time_decay_contribution'])
    avg_cf   = np.mean(contribution['confidence_contribution'])
    avg_sy   = np.mean(contribution['sybil_defense_contribution'])
    avg_ap   = np.mean(contribution['adaptive_pretrust_contribution'])
    avg_syn  = np.mean(contribution['synergy_effect'])
    avg_tot  = np.mean(contribution['total_improvement'])
    avg_tot2 = np.mean(contribution['total_improvement_with_td'])
    avg_marg = np.mean(contribution['time_decay_marginal_in_full'])

    print("\n[理论验证结论]:")
    print(f"   平均贡献：时间衰减 {avg_td:+.2f}% | 置信度折扣 {avg_cf:+.2f}% | "
          f"Sybil防御 {avg_sy:+.2f}% | 自适应预信任 {avg_ap:+.2f}% | "
          f"协同 {avg_syn:+.2f}%")
    print(f"   完整版（关TD）总提升 {avg_tot:+.2f}% | 完整版（开TD）总提升 {avg_tot2:+.2f}% | "
          f"完整版中 TD 边际贡献 {avg_marg:+.2f}%")
    print("   ⭐ 关键反例证据：完整改进版中开启时间衰减的边际贡献 = "
          f"{avg_marg:+.2f}%（如为负 → 实证支持'关闭时间衰减'的设计决策）")

    if scenario == 'sybil':
        print(
            f"   >>> 在 Sybil 场景下，Sybil防御应起主导作用（实测 {avg_sy:+.2f}%），"
            f"时间衰减与置信度折扣亦应有正贡献。"
        )
    elif scenario == 'whitewashing':
        print(
            f"   >>> 在白洗场景下，置信度折扣与时间衰减应有正贡献；"
            f"如某模块贡献为负，则该模块对此攻击方向无防御能力（属于'不可行改进方向'的合法负结果）。"
        )
    elif scenario == 'hybrid':
        print(
            f"   >>> 在混合攻击下，四模块合计 {avg_td + avg_cf + avg_sy + avg_ap:+.2f}% 的"
            f"独立贡献 + {avg_syn:+.2f}% 协同效应 = {avg_tot:+.2f}% 总提升。"
        )


if __name__ == "__main__":
    from utils.visualization import plot_ablation_comparison, plot_ablation_contribution

    scenarios = ['sybil', 'whitewashing', 'hybrid']
    
    for sc in scenarios:
        results, contribution = run_ablation_experiment(scenario=sc, verbose=True)
        print_ablation_summary(contribution, scenario=sc)

        plot_ablation_comparison(results, f"results/figures/06_ablation_{sc}_comparison.png")
        plot_ablation_contribution(contribution, f"results/figures/07_ablation_{sc}_contribution.png")

    print("\n" + "=" * 70)
    print("[DONE] 所有场景的学术消融实验已跑完！图表已分别生成在 results/figures/ 目录下。")
    print("=" * 70)