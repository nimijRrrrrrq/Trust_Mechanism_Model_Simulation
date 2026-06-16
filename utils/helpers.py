# utils/helpers.py
"""
辅助函数 - 随机数、数据保存、格式化输出等
"""

import random
import json
import csv
import os
from datetime import datetime
from config import RANDOM_SEED

def set_random_seed(seed=RANDOM_SEED):
    """设置随机种子，保证实验可复现"""
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)

def format_percentage(value):
    """格式化百分比输出"""
    return f"{value * 100:.1f}%"

def format_trust_value(value):
    """格式化信任值输出"""
    return f"{value:.4f}"

def save_results_to_csv(results, filename):
    """保存实验结果到 CSV 文件"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['攻击比例', 'EigenTrust', 'PeerTrust', 'PageRank'])
        for i, ratio in enumerate(results.get('attack_ratios', [])):
            writer.writerow([
                f"{int(ratio*100)}%",
                results['EigenTrust'][i],
                results['PeerTrust'][i],
                results['PageRank'][i]
            ])
    print(f"结果已保存: {filename}")

def save_results_to_json(results, filename):
    """保存实验结果到 JSON 文件"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"结果已保存: {filename}")

def get_timestamp():
    """获取当前时间戳字符串"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def generate_summary_table(results):
    """生成总结表格"""
    table = []
    table.append("=" * 60)
    table.append("实验结果总结")
    table.append("=" * 60)
    table.append(f"{'攻击比例':<10}{'EigenTrust':<15}{'PeerTrust':<15}{'PageRank':<15}")
    table.append("-" * 60)
    
    for i, ratio in enumerate(results.get('attack_ratios', [])):
        table.append(
            f"{int(ratio*100)}%{'':<6}"
            f"{results['EigenTrust'][i]:<15.4f}"
            f"{results['PeerTrust'][i]:<15.4f}"
            f"{results['PageRank'][i]:<15.4f}"
        )
    
    table.append("=" * 60)
    return "\n".join(table)

def calculate_statistics(data):
    """计算统计量"""
    if not data:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
    
    import numpy as np
    arr = np.array(data)
    return {
        'mean': np.mean(arr),
        'std': np.std(arr),
        'min': np.min(arr),
        'max': np.max(arr)
    }