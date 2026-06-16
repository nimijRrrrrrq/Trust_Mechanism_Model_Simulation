"""临时脚本：跑消融实验看完整版开启/关闭时间衰减的对比数据"""
import sys
sys.path.insert(0, '.')

from experiments.run_ablation import run_ablation_experiment, print_ablation_summary

print("=" * 70)
print("消融实验 - HYBRID 场景 - 7 个变体对比（重点看 TD 边际）")
print("=" * 70)

r, c = run_ablation_experiment(scenario='hybrid', verbose=False)
print_ablation_summary(c, scenario='hybrid')

print("\n\n[核心反例消融结论]")
print(f"  完整版·关闭时间衰减  = {sum(c['full_improved_accuracy'])/len(c['full_improved_accuracy']):.4f}")
print(f"  完整版·开启时间衰减  = {sum(c['full_with_td_accuracy'])/len(c['full_with_td_accuracy']):.4f}")
print(f"  边际差 (开TD - 关TD) = {sum(c['time_decay_marginal_in_full'])/len(c['time_decay_marginal_in_full']):+.4f} (百分点 * 100)")
print(f"  仅开时间衰减 vs 基线 = {sum(c['time_decay_contribution'])/len(c['time_decay_contribution']):+.2f} (百分点)")
