# main.py
"""
主程序 - 一键运行所有实验并生成完整报告
运行方式：python main.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 清除数据集缓存，确保重新加载
try:
    import core.network
    core.network._dataset_cache = None
    core.network._dataset_loaded_print_done = False
except:
    pass


def main():
    """一键运行所有实验"""
    start_time = time.time()
    
    print("=" * 70)
    print("基于社交网络的信任机制模型改进与模拟")
    print("综合实验 - 自动运行所有测试")
    print("=" * 70)
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 确保输出目录存在
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/logs', exist_ok=True)

    # =========================================================
    # 第1步：烟雾测试（验证所有模块正常）
    # =========================================================
    print("\n" + "=" * 70)
    print("[第1步] 烟雾测试 - 验证所有模块")
    print("=" * 70)
    
    try:
        from experiments.run_smoke_test import run_all_tests
        smoke_passed = run_all_tests()
        if not smoke_passed:
            print("\n警告: 烟雾测试部分失败，但继续运行实验...")
    except Exception as e:
        print(f"警告: 烟雾测试运行失败: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================
    # 第2步：P2P 自模拟实验
    # =========================================================
    print("\n" + "=" * 70)
    print("[第2步] P2P 自模拟实验")
    print("=" * 70)
    
    p2p_results = None
    try:
        from experiments.run_baselines import (
            run_p2p_experiment, 
            print_results_table, 
            print_efficiency_table,
            _convergence_results,
            _metric_results
        )
        
        print("\n运行 P2P 仿真实验 (200节点，多种攻击比例)...")
        p2p_results = run_p2p_experiment(verbose=True)
        
        print("\n" + "-" * 70)
        print("P2P 实验结果")
        print("-" * 70)
        print_results_table(p2p_results)
        print_efficiency_table()
        
    except Exception as e:
        print(f"错误: P2P 实验运行失败: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================
    # 第3步：攻击测试实验
    # =========================================================
    print("\n" + "=" * 70)
    print("[第3步] 攻击测试实验")
    print("=" * 70)
    
    attack_results = {}
    try:
        from experiments.run_attack_tests import (
            test_sybil_attack, 
            test_whitewashing_attack
        )
        
        print("\n运行 Sybil 攻击测试...")
        attack_results['sybil'] = test_sybil_attack()
        
        print("\n运行 白洗攻击测试...")
        attack_results['whitewashing'] = test_whitewashing_attack()
        
        print("\n" + "-" * 70)
        print("攻击测试结果摘要")
        print("-" * 70)
        for attack_type, results in attack_results.items():
            if results:
                print(f"\n{attack_type.upper()} 攻击:")
                for metric_name in ['roc_auc', 'pr_auc', 'accuracy', 'f1']:
                    metric_results = results.get(metric_name, {})
                    if metric_results:
                        print(f"  {metric_name.upper()}:")
                        for algo, values in metric_results.items():
                            if values:
                                avg = sum(values) / len(values)
                                print(f"    {algo}: 平均 {avg:.4f}")
        
    except Exception as e:
        print(f"警告: 攻击测试运行失败: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================
    # 第4步：Bitcoin OTC 时间衰减对比实验
    # =========================================================
    print("\n" + "=" * 70)
    print("[第4步] Bitcoin OTC 时间衰减对比实验")
    print("=" * 70)
    
    bitcoin_results = None
    try:
        from experiments.run_bitcoin_comparison import run_comparison
        
        print("\n运行 Bitcoin OTC 时间衰减对比...")
        bitcoin_results = run_comparison(
            lambda_decay_values=[0.05, 0.1, 0.5],
            verbose=True
        )
        
        if bitcoin_results:
            print("\n" + "-" * 70)
            print("Bitcoin OTC 对比结果")
            print("-" * 70)
            baseline = bitcoin_results.get('baseline', {})
            print(f"  基线版本: 迭代 {baseline.get('iterations', 0)} 次")
            
            for lambda_decay, decay in bitcoin_results.get('decay_results', {}).items():
                result = decay.get('result', {})
                distance = decay.get('distance', {})
                print(f"  λ={lambda_decay}: 迭代 {result.get('iterations', 0)} 次, "
                      f"L1差异 {distance.get('l1_distance', 0):.4f}")
        
    except Exception as e:
        print(f"警告: Bitcoin OTC 实验运行失败: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================
    # 第5步：消融实验
    # =========================================================
    print("\n" + "=" * 70)
    print("[第5步] 消融实验 - 各改进模块贡献度分析")
    print("=" * 70)
    
    ablation_results = None
    ablation_contribution = None
    try:
        from experiments.run_ablation import (
            run_ablation_experiment, 
            print_ablation_summary
        )
        
        print("\n运行消融实验...")
        ablation_results, ablation_contribution = run_ablation_experiment(
            scenario='hybrid', 
            verbose=True
        )
        
        if ablation_contribution:
            print("\n" + "-" * 70)
            print("消融实验分析报告")
            print("-" * 70)
            print_ablation_summary(ablation_contribution)
        
    except Exception as e:
        print(f"警告: 消融实验运行失败: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================
    # 第6步：Epinions 四算法评估
    # =========================================================
    print("\n" + "=" * 70)
    print("[第6步] Epinions 四算法评估")
    print("=" * 70)
    
    epinions_eval_results = None
    try:
        from experiments.run_epinions_evaluation import (
            run_epinions_evaluation,
            plot_epinions_results,
        )
        
        print("\n运行 Epinions 四算法统一评估 (EigenTrust / ImprovedEigenTrust / PeerTrust / PageRank)...")
        epinions_eval_results = run_epinions_evaluation(verbose=True)
        
        if epinions_eval_results:
            print("\n" + "-" * 70)
            print("Epinions 四算法评估结果")
            print("-" * 70)
            from experiments.run_epinions_evaluation import print_results_table as print_epinions_eval_table
            print_epinions_eval_table(epinions_eval_results)
            
            # 生成 Epinions 四算法对比图
            output_dir_epinions = 'results/figures'
            os.makedirs(output_dir_epinions, exist_ok=True)
            plot_epinions_results(epinions_eval_results, output_dir_epinions)
        
    except Exception as e:
        print(f"警告: Epinions 四算法评估运行失败: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================
    # 第7步：生成所有图表
    # =========================================================
    print("\n" + "=" * 70)
    print("[第7步] 生成实验图表")
    print("=" * 70)
    
    try:
        from utils.visualization import (
            plot_efficiency_comparison,
            plot_accuracy_comparison,
            plot_f1_comparison,
            plot_convergence,
            plot_attack_comparison
        )
        
        output_dir = 'results/figures'
        
        # 1. 效率对比图
        if p2p_results:
            from experiments.run_baselines import _efficiency_results
            if _efficiency_results:
                efficiency_data = []
                for algo_name, times in _efficiency_results.items():
                    if times:
                        efficiency_data.append({
                            'algorithm': algo_name,
                            'time': times[-1] if times else 0,
                            'iterations': 0
                        })
                if efficiency_data:
                    plot_efficiency_comparison(efficiency_data, output_dir)
        
        # 2. 准确率对比图
        if p2p_results:
            plot_accuracy_comparison(p2p_results, output_dir)
        
        # 3. F1 对比图
        from experiments.run_baselines import _metric_results
        if _metric_results:
            f1_results = {algo: metrics['f1'] for algo, metrics in _metric_results.items()}
            plot_f1_comparison(f1_results, output_dir)
        
        # 4. 收敛曲线图
        from experiments.run_baselines import _convergence_results
        if _convergence_results:
            residuals = []
            labels = []
            for algo_name, conv_histories in _convergence_results.items():
                if conv_histories and conv_histories[0]:
                    residuals.append(conv_histories[0])
                    labels.append(algo_name)
            if residuals:
                plot_convergence(residuals, labels, output_dir)
        
        # 5. 攻击测试图
        # attack_results[type] 格式: {'accuracy': {algo:[...]}, 'roc_auc': {algo:[...]}, ...}
        if attack_results.get('sybil'):
            sybil_results = attack_results['sybil']
            for metric_key, metric_label in [('accuracy', 'Accuracy'), ('f1', 'F1'), ('roc_auc', 'ROC AUC'), ('pr_auc', 'PR AUC')]:
                if sybil_results.get(metric_key):
                    plot_attack_comparison(
                        sybil_results[metric_key], output_dir, attack_type='Sybil',
                        attack_params=[2, 4, 6, 8, 10], metric_name=metric_label
                    )
        if attack_results.get('whitewashing'):
            ww_results = attack_results['whitewashing']
            for metric_key, metric_label in [('accuracy', 'Accuracy'), ('f1', 'F1'), ('roc_auc', 'ROC AUC'), ('pr_auc', 'PR AUC')]:
                if ww_results.get(metric_key):
                    plot_attack_comparison(
                        ww_results[metric_key], output_dir, attack_type='Whitewashing',
                        attack_params=[0.05, 0.10, 0.15, 0.20, 0.25], metric_name=metric_label
                    )
        
        # 6. 消融实验图
        if ablation_results:
            from utils.visualization import plot_ablation_comparison, plot_ablation_contribution
            plot_ablation_comparison(ablation_results, os.path.join(output_dir, 'ablation_comparison.png'))
            if ablation_contribution:
                plot_ablation_contribution(ablation_contribution, os.path.join(output_dir, 'ablation_contribution.png'))
        
        print(f"\n图表已保存到 {output_dir}")
        
    except Exception as e:
        print(f"警告: 图表生成失败: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================
    # 最终汇总
    # =========================================================
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "=" * 70)
    print("✓ 全部实验完成！")
    print("=" * 70)
    print(f"总运行时间: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
    print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    print("\n📊 生成的文件:")
    print("  results/figures/")
    print("    - efficiency_comparison.png  (效率对比)")
    print("    - accuracy_comparison.png    (准确率对比)")
    print("    - f1_comparison.png          (F1分数对比)")
    print("    - convergence_plot.png       (收敛曲线)")
    print("    - attack_sybil_roc_auc_comparison.png (Sybil攻击ROC-AUC)")
    print("    - attack_whitewashing_roc_auc_comparison.png (白洗攻击ROC-AUC)")
    print("    - ablation_comparison.png    (消融实验对比)")
    print("    - ablation_contribution.png  (消融贡献度)")
    print("    - improvement_comparison.png (改进提升对比)")
    print("    - epinions_metrics_comparison.png (Epinions 四算法评估指标对比)")
    
    print("\n🎉 感谢使用信任机制模拟平台！")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)