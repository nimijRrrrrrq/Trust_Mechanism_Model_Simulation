# experiments/run_smoke_test.py
"""
烟雾测试 - 快速验证所有核心模块可正常导入和基本运行
被 main.py 第1步通过 run_all_tests() 调用
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_smoke_test() -> bool:
    """
    依次执行最小化验证，任何步骤失败则捕获异常并返回 False。
    全部通过则打印 "Smoke test passed" 并返回 True。
    """
    import traceback

    # ── Step 1: 导入 core 模块 ───────────────────────────────────────
    try:
        from core.node import Node
        from core.network import TrustNetwork
        from core.simulation import Simulation
        print("  [PASS] core 模块导入 (Node, TrustNetwork, Simulation)")
    except Exception as e:
        print(f"  [FAIL] core 模块导入失败: {e}")
        traceback.print_exc()
        return False

    # ── Step 2: 导入 models 模块 ────────────────────────────────────
    try:
        from models.eigentrust import EigenTrust
        from models.eigentrust_improved import ImprovedEigenTrust
        from models.pagerank import PageRankTrust
        from models.peertrust import PeerTrust
        print("  [PASS] models 模块导入 (EigenTrust, ImprovedEigenTrust, PageRank, PeerTrust)")
    except Exception as e:
        print(f"  [FAIL] models 模块导入失败: {e}")
        traceback.print_exc()
        return False

    # ── Step 3: 导入 experiments.run_baseline_validation ────────────
    try:
        import experiments.run_baseline_validation  # 仅导入，不运行
        print("  [PASS] experiments.run_baseline_validation 导入")
    except Exception as e:
        print(f"  [FAIL] experiments.run_baseline_validation 导入失败: {e}")
        traceback.print_exc()
        return False

    # ── Step 4: 创建最小网络与仿真对象 ───────────────────────────────
    try:
        import random
        import numpy as np
        random.seed(42)
        np.random.seed(42)
        net = TrustNetwork(num_nodes=5, attack_ratio=0.2)
        sim = Simulation(net)
        sim.run(rounds=2)
        print("  [PASS] TrustNetwork(5节点) + Simulation(2轮) 创建并运行")
    except Exception as e:
        print(f"  [FAIL] 网络/仿真创建失败: {e}")
        traceback.print_exc()
        return False

    # ── Step 5: 调用 EigenTrust.compute() ───────────────────────────
    try:
        t_vec, _, _ = EigenTrust.compute(net, max_iter=10)
        assert len(t_vec) == len(net.nodes), "信任向量长度与节点数不匹配"
        assert abs(t_vec.sum() - 1.0) < 1e-3, f"信任向量和不为1: {t_vec.sum()}"
        print(f"  [PASS] EigenTrust.compute() 返回长度={len(t_vec)} 的信任向量，和={t_vec.sum():.6f}")
    except Exception as e:
        print(f"  [FAIL] EigenTrust.compute() 失败: {e}")
        traceback.print_exc()
        return False

    # ── Step 6: 调用 ImprovedEigenTrust.compute() ───────────────────
    try:
        result = ImprovedEigenTrust.compute(net, max_iter=10, verbose=False)
        assert len(result.trust_vector) == len(net.nodes)
        assert abs(result.trust_vector.sum() - 1.0) < 1e-3
        print(f"  [PASS] ImprovedEigenTrust.compute() 返回长度={len(result.trust_vector)} 的信任向量")
    except Exception as e:
        print(f"  [FAIL] ImprovedEigenTrust.compute() 失败: {e}")
        traceback.print_exc()
        return False

    print("\n  Smoke test passed")
    return True


def run_all_tests() -> bool:
    """main.py 调用的入口，内部调用 run_smoke_test()"""
    return run_smoke_test()


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
