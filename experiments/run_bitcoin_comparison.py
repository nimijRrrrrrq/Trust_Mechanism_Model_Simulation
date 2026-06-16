# experiments/run_bitcoin_comparison.py
"""
Bitcoin OTC 数据集实验 - 对比时间衰减效果

对比内容：
1. 原版 EigenTrust（基线 C 矩阵）vs 改进版（时间衰减 C 矩阵）
2. 不同衰减系数 λ 下的收敛曲线
3. 最终信任分布差异

数据集：Bitcoin OTC
- rating ∈ [-10, 10] → 归一化到 [0, 1]: (rating + 10) / 20
- timestamp: Unix 时间戳（浮点）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

from models.eigentrust import EigenTrust
from data_loader import build_bitcoin_otc_baseline, build_bitcoin_otc_with_decay
from config import DATA_PATHS


def compute_eigentrust(C, max_iter=100, tolerance=1e-6, alpha=0.15, track_convergence=True):
    """
    调用 EigenTrust 算法计算信任值
    
    Args:
        C: 行随机信任矩阵
        max_iter: 最大迭代次数
        tolerance: 收敛阈值
        alpha: 预信任权重
        track_convergence: 是否跟踪收敛过程
    
    Returns:
        dict: {'trust_vector', 'convergence_history', 'compute_time', 'iterations'}
    """
    start_time = time.time()
    trust_vector, convergence_history, _ = EigenTrust.compute(
        C, 
        max_iter=max_iter, 
        tolerance=tolerance, 
        alpha=alpha,
        track_convergence=track_convergence,
        verbose=False
    )
    compute_time = time.time() - start_time
    
    iterations = len(convergence_history) if convergence_history else max_iter
    
    return {
        'trust_vector': trust_vector,
        'convergence_history': convergence_history,
        'compute_time': compute_time,
        'iterations': iterations
    }


def compare_distributions(t1, t2):
    """
    计算两个信任分布之间的差异
    
    Args:
        t1: 信任向量 1
        t2: 信任向量 2
    
    Returns:
        dict: 差异指标
    """
    # KL 散度（需要添加小常数避免 log(0)）
    eps = 1e-12
    t1_safe = np.maximum(t1, eps)
    t2_safe = np.maximum(t2, eps)
    kl_div = np.sum(t1_safe * np.log(t1_safe / t2_safe))
    
    # JS 散度
    m = 0.5 * (t1_safe + t2_safe)
    js_div = 0.5 * np.sum(t1_safe * np.log(t1_safe / m)) + 0.5 * np.sum(t2_safe * np.log(t2_safe / m))
    
    # L1 距离
    l1_dist = np.sum(np.abs(t1 - t2))
    
    # L2 距离
    l2_dist = np.linalg.norm(t1 - t2)
    
    # 相关性
    corr = np.corrcoef(t1, t2)[0, 1]
    
    return {
        'kl_divergence': kl_div,
        'js_divergence': js_div,
        'l1_distance': l1_dist,
        'l2_distance': l2_dist,
        'correlation': corr
    }


def run_comparison(lambda_decay_values=[0.05, 0.1, 0.5, 1.0], verbose=True):
    """
    运行对比实验
    
    Args:
        lambda_decay_values: 时间衰减系数列表
        verbose: 是否打印详细信息
    
    Returns:
        dict: 实验结果
    """
    if verbose:
        print("=" * 70)
        print("Bitcoin OTC 时间衰减对比实验")
        print("=" * 70)
        print(f"数据集路径: {DATA_PATHS['bitcoin_otc']}")
        print(f"测试衰减系数: {lambda_decay_values}")
        print()
    
    # 1. 加载基线数据（无时间衰减）
    if verbose:
        print("1. 构建基线信任矩阵（无时间衰减）...")
    C_baseline, node_idx, idx_node, df = build_bitcoin_otc_baseline()
    n = C_baseline.shape[0]
    
    if verbose:
        print(f"   节点数: {n}")
        print(f"   边数: {len(df)}")
        print(f"   数据时间范围: [{df['time'].min():.0f}, {df['time'].max():.0f}]")
        print()
    
    # 2. 计算基线 EigenTrust
    if verbose:
        print("2. 计算基线 EigenTrust...")
    baseline_result = compute_eigentrust(C_baseline)
    
    if verbose:
        print(f"   计算时间: {baseline_result['compute_time']:.4f}s")
        print(f"   迭代次数: {baseline_result['iterations']}")
        print(f"   信任值范围: [{baseline_result['trust_vector'].min():.6f}, {baseline_result['trust_vector'].max():.6f}]")
        print()
    
    # 3. 对比不同衰减系数
    decay_results = {}
    for lambda_decay in lambda_decay_values:
        if verbose:
            print(f"3. 处理衰减系数 λ = {lambda_decay}")
        
        # 构建带时间衰减的矩阵
        C_decay, _, _, _ = build_bitcoin_otc_with_decay(lambda_decay=lambda_decay)
        
        # 计算 EigenTrust
        result = compute_eigentrust(C_decay)
        
        # 计算与基线的差异
        dist_diff = compare_distributions(baseline_result['trust_vector'], result['trust_vector'])
        
        decay_results[lambda_decay] = {
            'result': result,
            'distance': dist_diff,
            'matrix_diff': np.linalg.norm(C_decay - C_baseline, 'fro')
        }
        
        if verbose:
            print(f"   计算时间: {result['compute_time']:.4f}s")
            print(f"   迭代次数: {result['iterations']}")
            print(f"   矩阵差异(Frobenius): {decay_results[lambda_decay]['matrix_diff']:.4f}")
            print(f"   KL散度: {dist_diff['kl_divergence']:.4f}")
            print(f"   JS散度: {dist_diff['js_divergence']:.4f}")
            print(f"   L1距离: {dist_diff['l1_distance']:.4f}")
            print(f"   相关性: {dist_diff['correlation']:.4f}")
            print()
    
    # 4. 输出汇总表格
    if verbose:
        print("=" * 70)
        print("实验结果汇总")
        print("=" * 70)
        
        # 表头
        print(f"{'λ':<10} {'时间(s)':<12} {'迭代':<6} {'KL散度':<10} {'JS散度':<10} {'L1距离':<10} {'相关性':<10}")
        print("-" * 70)
        
        # 基线（λ=0）
        print(f"{'0(基线)':<10} {baseline_result['compute_time']:<12.4f} {baseline_result['iterations']:<6} {'-':<10} {'-':<10} {'-':<10} {'-':<10}")
        
        # 不同衰减系数
        for lambda_decay in lambda_decay_values:
            res = decay_results[lambda_decay]
            dist = res['distance']
            print(f"{lambda_decay:<10} {res['result']['compute_time']:<12.4f} {res['result']['iterations']:<6} "
                  f"{dist['kl_divergence']:<10.4f} {dist['js_divergence']:<10.4f} "
                  f"{dist['l1_distance']:<10.4f} {dist['correlation']:<10.4f}")
        
        print("=" * 70)
    
    return {
        'baseline': baseline_result,
        'baseline_matrix': C_baseline,
        'decay_results': decay_results,
        'node_idx': node_idx,
        'idx_node': idx_node,
        'n': n
    }


def plot_convergence_curves(results, save_path=None):
    """
    绘制收敛曲线对比图
    
    Args:
        results: run_comparison 返回的结果
        save_path: 保存路径（可选）
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from utils.visualization import PALETTE

        plt.figure(figsize=(10, 6))

        # 基线（λ=0）使用与 EigenTrust 一致的蓝色，便于跨图表统一
        baseline_conv = results['baseline']['convergence_history']
        plt.plot(range(1, len(baseline_conv) + 1), baseline_conv,
                 label='λ=0 (基线)', color=PALETTE[5], linewidth=2, linestyle='--')

        # 不同衰减系数，按调色板顺序分配
        for i, (lambda_decay, res) in enumerate(results['decay_results'].items()):
            conv_history = res['result']['convergence_history']
            plt.plot(range(1, len(conv_history) + 1), conv_history,
                     label=f'λ={lambda_decay}', color=PALETTE[i % len(PALETTE)], linewidth=2)
        
        plt.xlabel('迭代次数')
        plt.ylabel('L1 变化量')
        plt.title('Bitcoin OTC 不同衰减系数的收敛曲线')
        plt.legend()
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"收敛曲线图已保存到: {save_path}")
        else:
            plt.show()
            
    except ImportError:
        print("警告: 无法导入 matplotlib，跳过绘图")


def plot_trust_distribution(results, save_path=None):
    """
    绘制信任值分布对比图
    
    Args:
        results: run_comparison 返回的结果
        save_path: 保存路径（可选）
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from utils.visualization import PALETTE

        plt.figure(figsize=(10, 6))

        # 基线（λ=0）使用与 EigenTrust 一致的蓝色
        baseline_t = results['baseline']['trust_vector']
        plt.hist(baseline_t, bins=50, alpha=0.5, label='λ=0 (基线)', density=True,
                 color=PALETTE[5])

        # 不同衰减系数，按调色板顺序分配
        for i, (lambda_decay, res) in enumerate(results['decay_results'].items()):
            t = res['result']['trust_vector']
            plt.hist(t, bins=50, alpha=0.5, label=f'λ={lambda_decay}',
                     color=PALETTE[i % len(PALETTE)], density=True)
        
        plt.xlabel('信任值')
        plt.ylabel('概率密度')
        plt.title('Bitcoin OTC 不同衰减系数的信任值分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"分布图已保存到: {save_path}")
        else:
            plt.show()
            
    except ImportError:
        print("警告: 无法导入 matplotlib，跳过绘图")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Bitcoin OTC 时间衰减对比实验')
    parser.add_argument('--lambdas', type=float, nargs='+', 
                        default=[0.05, 0.1, 0.5, 1.0],
                        help='时间衰减系数列表')
    parser.add_argument('--plot', action='store_true', 
                        help='是否生成收敛曲线图')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 运行对比实验
    results = run_comparison(lambda_decay_values=args.lambdas, verbose=True)
    
    # 生成图表
    if args.plot:
        convergence_path = os.path.join(args.output_dir, 'bitcoin_convergence.png')
        distribution_path = os.path.join(args.output_dir, 'bitcoin_distribution.png')
        plot_convergence_curves(results, convergence_path)
        plot_trust_distribution(results, distribution_path)
    
    print("\n实验完成！")
