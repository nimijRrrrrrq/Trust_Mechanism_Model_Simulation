# 信任机制仿真与评估平台 (Trust Simulation Platform)

> 基于社交网络的信任机制模型改进与模拟平台 —— 围绕经典 EigenTrust 声誉管理算法，提供算法改进、攻击测试、消融分析与可视化评估的完整实验框架。

## 🔗 项目链接

- **GitHub 仓库**：[`https://github.com/nimijRrrrrrq/Trust_Mechanism_Model_Simulation.git`](https://github.com/nimijRrrrrrq/Trust_Mechanism_Model_Simulation)
- **Issues**：[`https://github.com/nimijRrrrrrq/Trust_Mechanism_Model_Simulation/issues`](https://github.com/nimijRrrrrrq/Trust_Mechanism_Model_Simulation/issues)
- **作者 / 维护者**：`nimijRrrrrrq`
- **License**：MIT （*请按实际 license 调整*）

---

## 目录

- [一、项目简介](#一项目简介)
- [二、核心特性](#二核心特性)
- [三、项目结构](#三项目结构)
- [四、环境与安装](#四环境与安装)
- [五、快速开始](#五快速开始)
- [六、算法说明](#六算法说明)
  - [6.1 经典 EigenTrust](#61-经典-eigentrust)
  - [6.2 改进版 EigenTrust](#62-改进版-eigentrust)
  - [6.3 PeerTrust](#63-peertrust)
  - [6.4 PageRank](#64-pagerank)
  - [6.5 四种算法对比总结](#65-四种算法对比总结)
- [七、核心模块设计](#七核心模块设计)
- [八、数据集说明](#八数据集说明)
- [九、实验设计](#九实验设计)
  - [9.1 实验总览](#91-实验总览)
  - [9.2 P2P 仿真基线实验](#92-p2p-仿真基线实验)
  - [9.3 攻击测试实验](#93-攻击测试实验)
  - [9.4 Bitcoin OTC 时间衰减对比（参考脚本）](#94-bitcoin-otc-时间衰减对比参考脚本)
  - [9.5 消融实验](#95-消融实验)
  - [9.6 Epinions 四算法统一评估](#96-epinions-四算法统一评估)
  - [9.7 烟雾测试](#97-烟雾测试)
- [十、可视化输出](#十可视化输出)
- [十一、实验参数与可复现性](#十一实验参数与可复现性)
- [十二、依赖项](#十二依赖项)
- [十三、常见问题](#十三常见问题)
- [引用](#引用)
- [许可](#许可)

---

## 一、项目简介

本项目是一个面向 P2P/社交网络的**信任机制模型研究与仿真平台**，围绕经典 EigenTrust 声誉管理算法展开改进，并提供完整的对比、消融与攻击测试框架。

**研究目标**：

- 实现并验证**改进版 EigenTrust** 算法
- 对比多种信任算法在不同攻击场景下的表现
- 评估 **时间衰减、置信度评估、Sybil 防御、自适应预信任** 等改进机制的有效性
- 提供公平的实验环境与可复现的结果

**应用场景**：P2P 文件共享网络、电商信任评价、社交网络声誉管理等。

---

## 二、核心特性

| 特性 | 说明 |
|------|------|
| **多算法实现** | 经典 EigenTrust、改进版 EigenTrust、PeerTrust、PageRank 四种信任算法 |
| **网络拓扑多样** | 支持随机图、小世界网络、Epinions 真实数据集子图（FilmTrust/Bitcoin OTC 数据加载已实现但主实验未使用） |
| **攻击模型完备** | 内置 Sybil 攻击与 Whitewashing（白洗）攻击模拟 |
| **改进机制可插拔** | 时间衰减 / 置信度折扣 / Sybil 防御 / 自适应预信任四大模块可独立开关 |
| **统一评估标准** | 基于排序的 Accuracy / Precision / Recall / F1 / ROC-AUC / PR-AUC |
| **完整实验体系** | 烟雾测试、基线对比、攻击测试、消融实验、Epinions 四算法评估（Bitcoin OTC 仅为参考脚本） |
| **可视化报告** | 自动生成 20+ 张对比图（效率、准确率、F1、收敛曲线、攻击结果等） |
| **可复现性强** | 固定随机种子、统一参数、代码隔离、日志可追溯 |

---

## 三、项目结构

```
trust_simulation/
├── main.py                          # 主程序入口：一键运行所有实验
├── config.py                        # 全局配置（路径、网络、交易、攻击、算法参数）
├── data_loader.py                   # 数据集加载（Epinions/FilmTrust/Bitcoin OTC）
│
├── core/                            # 核心模块
│   ├── __init__.py
│   ├── node.py                      # 节点模型（含白洗攻击 reset、行为分数）
│   ├── network.py                   # 信任网络（拓扑构建、交易执行、数据集加载）
│   ├── transaction.py               # 交易定义
│   └── simulation.py                # 交易仿真引擎
│
├── models/                          # 信任算法模型
│   ├── eigentrust.py                # 经典 EigenTrust（Kamvar et al. WWW 2003）
│   ├── eigentrust_improved.py       # 改进版 EigenTrust（4 大改进模块）
│   ├── peertrust.py                 # PeerTrust 算法
│   └── pagerank.py                  # 标准 PageRank 信任传播
│
├── experiments/                     # 实验脚本
│   ├── run_smoke_test.py            # 烟雾测试（验证模块完整性）
│   ├── run_baselines.py             # P2P + Epinions 基线对比实验
│   ├── run_attack_tests.py          # Sybil / Whitewashing 攻击测试
│   ├── run_ablation.py              # 消融实验（验证各模块贡献）
│   ├── run_epinions_evaluation.py   # Epinions 四算法统一评估
│   ├── run_bitcoin_comparison.py    # Bitcoin OTC 时间衰减对比（**参考脚本，主实验未使用**）
│   └── run_baseline_validation.py   # 基线验证脚本
│
├── utils/                           # 工具模块
│   ├── helpers.py                   # 随机种子、结果保存、格式化
│   ├── logger.py                    # 日志记录
│   └── visualization.py             # 可视化（统一配色、中文字体）
│
├── data/                            # 数据集
│   ├── soc-Epinions1.txt/           # Epinions 社交信任网络（主实验使用）
│   ├── filmtrust/                   # FilmTrust（评分 + 信任，未实际使用）
│   │   ├── rating/                  # 4 个评分文件（ratings_0.txt ~ ratings_3.txt）
│   │   └── trust/trust.txt          # 信任关系
│   └── soc-sign-bitcoinotc.csv/     # Bitcoin OTC 交易信任（仅作参考脚本使用）
│
├── results/                         # 实验输出
│   ├── figures/                     # 可视化图表（PNG，300 DPI）
│   └── logs/                        # 实验日志
│
└── README.md                        # 本文件
```

---

## 四、环境与安装

### 4.1 Python 版本

推荐 **Python 3.8+**（项目内 `__pycache__` 缓存显示为 cpython-38）。

### 4.2 依赖项

```bash
pip install numpy pandas networkx scikit-learn scipy matplotlib
```

或使用 requirements.txt（若已有）：

```bash
pip install -r requirements.txt
```

### 4.3 数据集准备

数据集已放置于 `data/` 目录下：

| 数据集 | 位置 | 大小 | 是否被主实验使用 |
|--------|------|------|------------------|
| Epinions | `data/soc-Epinions1.txt/soc-Epinions1.txt` | ~84K 信任边 | ✓ **是**（P2P 仿真 + 四算法评估） |
| FilmTrust 评分 | `data/filmtrust/rating/ratings_{0,1,2,3}.txt` | ~15K 条 | ✗ 否（数据加载函数已实现） |
| FilmTrust 信任 | `data/filmtrust/trust/trust.txt` | ~1.8K 边 | ✗ 否（数据加载函数已实现） |
| Bitcoin OTC | `data/soc-sign-bitcoinotc.csv/soc-sign-bitcoinotc.csv` | ~20K 交易 | ✗ **否**（仅作参考脚本 `run_bitcoin_comparison`） |

如需更换数据集路径，修改 `config.py` 中的 `DATA_PATHS` 即可。

---

## 五、快速开始

### 5.1 一键运行所有实验

```bash
cd trust_simulation
python main.py
```

`main.py` 将依次执行（**实际参与主流程的实验**）：

1. **烟雾测试** —— 验证所有模块正常导入与运行
2. **P2P 自模拟实验** —— 200 节点，遍历 0%~60% 恶意节点比例
3. **攻击测试实验** —— Sybil 攻击 + Whitewashing 攻击
4. **消融实验** —— 验证各改进模块的独立贡献
5. **Epinions 四算法评估** —— 真实数据集上四算法统一比较
6. **可视化图表生成** —— 输出至 `results/figures/`

> **注意**：Bitcoin OTC 数据集与 `experiments/run_bitcoin_comparison.py` 时间衰减对比实验**仅作为参考脚本保留**，并未纳入 `main.py` 的主实验流程，也不参与主结论的生成。如需单独运行该参考实验，可执行 `python -m experiments.run_bitcoin_comparison`。

### 5.2 单独运行各实验

```bash
# 基线对比
python -m experiments.run_baselines

# 攻击测试
python -m experiments.run_attack_tests

# Bitcoin OTC 时间衰减对比（**参考脚本，未纳入主实验**）
python -m experiments.run_bitcoin_comparison

# 消融实验
python -m experiments.run_ablation

# Epinions 四算法评估
python -m experiments.run_epinions_evaluation

# 烟雾测试
python -m experiments.run_smoke_test
```

---

## 六、算法说明

本项目实现并对比 **四种信任算法**：

| 节 | 算法 | 角色 |
|----|------|------|
| 6.1 | 经典 EigenTrust | 本文改进的基线 |
| 6.2 | 改进版 EigenTrust | 本文核心方法 |
| 6.3 | PeerTrust | 引入反馈可信度的经典算法 |
| 6.4 | PageRank | 经典链接权威性算法 |
| 6.5 | 四种算法对比总结 | 横向对比 |

### 6.1 经典 EigenTrust

**论文**：Kamvar, Schlosser & Garcia-Molina. *The EigenTrust Algorithm for Reputation Management in P2P Networks*. WWW 2003.

**论文链接**：
- ACM 官方页：[The Eigentrust algorithm for reputation management in P2P networks | Proceedings of the 12th international conference on World Wide Web](https://dl.acm.org/doi/10.1145/775152.775242)
- DOI：[10.1145/775152.775242](https://doi.org/10.1145/775152.775242)
- 预印本 PDF：[Stanford SNAP](https://snap.stanford.edu/data/web-Epinions.html) / [Indiana University 备份](https://homes.luddy.indiana.edu/kapadia/courses/I590-Fall-09/internal/eigentrust.pdf)

**核心思想**：将网络中的局部信任关系建模为行随机矩阵 $C$（$C_{ij}$ 表示节点 $i$ 对节点 $j$ 的归一化局部信任），通过幂迭代求解矩阵 $C^T$ 的 Perron 特征向量作为全局信任值。

**迭代公式**：

$$t^{(k+1)} = (1-\alpha) \cdot C^{T} \cdot t^{(k)} + \alpha \cdot p$$

其中：
- $t$：全局信任向量（归一化为概率分布）
- $C$：归一化的局部信任矩阵（$C_{ij}$ = 节点 $i$ 对节点 $j$ 的信任）
- $p$：预信任向量（通常均匀分布或集中于种子节点）
- $\alpha$：预信任权重（默认 0.15，避免悬挂节点问题）

**收敛保证**：当 $C$ 不可约且非负时，幂迭代收敛到 Perron 特征向量。

**关键步骤**：
1. 收集每个节点对邻居的局部信任评分 $s_{ij}$（正反馈累计 / 负反馈累计）
2. 行归一化得到 $C_{ij} = s_{ij} / \sum_k s_{ik}$，悬挂节点行设为均匀分布
3. 幂迭代至 $\|t^{(k+1)} - t^{(k)}\|_1 < 10^{-6}$

**实现位置**：[models/eigentrust.py](models/eigentrust.py)

支持两种输入模式：
1. `TrustNetwork` 对象（完整仿真链路）
2. 直接接收信任矩阵 $C$（高效模式，适用于大数据集）

**优点**：理论简洁、收敛有保证、对恶意集体合谋具有结构性抗性。
**缺点**：所有交易等权处理，无法反映信任的时效性；新节点无置信度折扣；无法防御 Sybil 攻击。

### 6.2 改进版 EigenTrust

**核心改进公式**：

$$t^{(k+1)} = (1-\alpha) \cdot C^{T} \cdot (t \odot w) + \alpha \cdot p^{(k)}$$

其中 $w$ 为节点传播能力权重向量（基于置信度折扣），$\odot$ 为 Hadamard 积。

**四大改进模块**：

| 改进项 | 公式/机制 | 解决的问题 |
|--------|-----------|------------|
| **时间衰减** | $w_t = e^{-\lambda \Delta t}$（指数/半衰期/线性） | 近期交易更能反映当前信任状态 |
| **置信度评估** | $w_c = \min_c + (1-\min_c)\cdot\sqrt{\text{ratio}}$ | 小样本节点传播能力被折扣 |
| **Sybil 防御** | 检测内部交易比例 $> 0.5$ 时降权 | 抑制 Sybil 团伙互刷信任 |
| **自适应预信任** | $p^{(k+1)} = (1-\gamma)p_0 + \gamma \cdot t^{(k)}$ | 动态调整锚点，加速收敛 |

**理论保证**：

- **收敛性**：当所有 $w_i > 0$ 时，$C^T D$（$D = \text{diag}(w)$）仍是正矩阵，Perron-Frobenius 定理保证收敛
- **稳定性**：$\gamma \leq 0.5$ 时自适应更新不会导致振荡
- **概率分布保持**：信任向量在每次迭代后归一化

**实现位置**：[models/eigentrust_improved.py](models/eigentrust_improved.py)

**关键参数**（见 [config.py](config.py)）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `alpha` | 0.15 | 预信任权重（PageRank 经典取值） |
| `adaptive_trust_rate` | 0.2 | 自适应预信任强度 |
| `min_confidence` | 0.1 | 悬挂节点最低传播权重 |
| `min_transactions` | 5 | 样本量阈值，低于此值置信度折扣 |
| `sybil_internal_ratio` | 0.5 | 内部交易比例阈值 |
| `sybil_penalty_factor` | 0.1 | Sybil 节点权重惩罚系数 |
| `decay_mode` | "half_life" | 衰减模式：exponential/half_life/linear |
| `decay_parameter` | 7.0 | 半衰期参数 |

### 6.3 PeerTrust

**论文**：Xiong, Liu. *PeerTrust: Supporting Reputation-Based Trust for Peer-to-Peer Electronic Communities*. IEEE TKDE 2004.

**论文链接**：
- IEEE Xplore：[PeerTrust: supporting reputation-based trust for peer-to-peer electronic communities](https://ieeexplore.ieee.org/document/1318566)
- DOI：[10.1109/TKDE.2004.1318566](https://doi.org/10.1109/TKDE.2004.1318566)
- 预印本 PDF：[Georgia Tech 备份](https://faculty.cc.gatech.edu/~lingliu/papers/2004/xiong04peertrust.pdf)

**核心思想**：在信任聚合中引入**多个调节因子**，不仅考虑直接反馈数量，还考虑反馈可信度、交易上下文、交易金额、社区因子等。

**信任计算公式**：

$$T(u) = \alpha \cdot \sum_{v \in I(u)} S(u,v) \cdot Cr(v) \cdot TF(u,v) + (1-\alpha) \cdot CF(u)$$

- $S(u,v)$：节点 $v$ 提交给 $u$ 的反馈满意度（feedback satisfaction）
- $Cr(v)$：节点 $v$ 作为评价者的可信度（credibility），通过与其他评价者的一致性估计
- $TF(u,v)$：交易上下文因子（transaction context factor），区分交易类型
- $CF(u)$：社区因子（community factor），如账号年龄、活跃度等
- $\alpha$：平衡参数（论文中取 0.1）

**本项目实现**（[models/peertrust.py](models/peertrust.py)）：
- 反馈归一化：原始 `feedback ∈ [-1, 1]` 映射到 `S ∈ [0, 1]`
- 评价者可信度 $Cr(v)$：通过 $v$ 历史反馈与全局平均的一致性估计
- 反馈密度作为平滑项（避免零交易节点的不可靠性）
- 幂迭代至信任向量收敛

**优点**：模型贴近真实交易场景，融合多种上下文因素。
**缺点**：依赖多轮交易、交易金额、评价上下文等丰富信息；在静态信任网络（如 Epinions）上这些信息不可得，模型优势难以发挥（见 [experiments/run_epinions_evaluation.py](experiments/run_epinions_evaluation.py) 说明）。

### 6.4 PageRank

**论文**：Brin & Page. *The Anatomy of a Large-Scale Hypertextual Web Search Engine*. Computer Networks 30 (1998), pp. 107-117.

**论文链接**：
- Stanford 主页（PDF）：[The Anatomy of a Large-Scale Hypertextual Web Search Engine](https://infolab.stanford.edu/~backrub/google.html)
- 论文 PDF 直链：[snap.stanford.edu](https://snap.stanford.edu/class/cs224w-readings/Brin98Anatomy.pdf)
- Google Research 出版页：[research.google](https://research.google/pubs/the-anatomy-of-a-large-scale-hypertextual-web-search-engine/)
- DOI：[10.1016/S0169-7552(98)00110-X](https://doi.org/10.1016/S0169-7552(98)00110-X)

**核心思想**：将信任视为"链接权威性"，从某个节点出发按转移概率在有向图上做随机游走，平稳分布即为节点的 PageRank 值（也即信任值）。

**迭代公式**：

$$\mathbf{t}^{(k+1)} = d \cdot M \cdot \mathbf{t}^{(k)} + \frac{1-d}{n} \cdot \mathbf{1}$$

- $M$：行随机转移矩阵，$M_{ji} = 1 / L(i)$，$L(i)$ 为节点 $i$ 的出度
- $d$：阻尼系数（默认 0.85），对应随机游走中"不点击链接直接跳转到任意节点"的概率
- 悬挂节点（出度为 0）的转移概率均匀分布到所有节点
- 收敛后信任向量 $t^*$ 满足 $\sum_i t^*_i = 1$

**本项目实现**（[models/pagerank.py](models/pagerank.py)）：
- 严格遵循原论文定义，不对结果做破坏概率分布的归一化
- 节点数 $> 5000$ 时自动使用 scipy 稀疏矩阵加速
- 支持 `TrustNetwork` 与 `networkx.DiGraph` 两种输入模式

**优点**：结构简洁、可解释性强（"被越多高权威节点指向越可信"）、工程实现成熟。
**缺点**：不区分边的方向语义（不区分"信任"与"被信任"）；忽略边的权重差异；无时间维度。

### 6.5 四种算法对比总结

| 维度 | 经典 EigenTrust | 改进版 EigenTrust | PeerTrust | PageRank |
|------|-----------------|--------------------|-----------|----------|
| **理论依据** | 矩阵 Perron 特征向量 | EigenTrust + 四改进模块 | 加权反馈聚合 + 评价者可信度 | 随机游走平稳分布 |
| **是否考虑交易金额/上下文** | ✗ | ✗ | ✓ | ✗ |
| **是否使用时间衰减** | ✗ | ✓（指数/半衰期/线性） | ✗ | ✗ |
| **是否有置信度折扣** | ✗ | ✓（Wilson 得分） | 部分（反馈密度） | ✗ |
| **是否有 Sybil 防御** | ✗ | ✓ | ✗ | ✗ |
| **是否需要交易时间戳** | ✗ | 可选 | ✗ | ✗ |
| **对静态信任网络适应性** | ✓ | ✓ | ✗（缺上下文） | ✓ |
| **对动态 P2P 仿真适应性** | ✓ | ✓ | ✓ | ✓ |
| **计算复杂度** | $O(k \cdot m)$ | $O(k \cdot m)$ | $O(k \cdot m)$ | $O(k \cdot m)$ |
| **是否容易解释** | 中 | 中（参数多） | 中（参数多） | 易 |

> 说明：$k$ 为迭代次数，$m$ 为网络边数；四者复杂度同阶，但 PageRank 的稀疏矩阵优化最成熟，常用于大规模图。

---

## 七、核心模块设计

### 7.1 节点模型（core/node.py）

```python
class Node:
    def __init__(self, node_id: int, is_malicious: bool = False):
        self.id = node_id
        self.is_malicious = is_malicious
        self.trust_value = 0.5          # 信任值 [0, 1]
        self.behavior_score = 1.0       # 行为分数（白洗惩罚使用）
        self.transaction_history = []    # 交易历史
        self.reset_count = 0             # 白洗攻击次数
```

**白洗攻击** `reset()` 方法：
- 清空交易历史
- 信任值重置为 0.5（多次白洗后降为 0.4）
- 行为分数衰减 5%（累积惩罚）
- 增加白洗计数

### 7.2 信任网络（core/network.py）

支持三种网络拓扑：

| 类型 | 说明 | 实现方法 |
|------|------|----------|
| `random` | Erdős–Rényi 随机图 | 每个节点连接 DEGREE 个随机邻居 |
| `small_world` | Watts-Strogatz 小世界模型 | 高聚类 + 短平均路径 |
| `dataset` | 真实数据集子图 | 取数据集度数最高的 N 个节点 |

**交易执行机制** `perform_transaction()`：
- 真实反馈：卖家恶意 → 差评 (-1)，卖家正常 → 好评 (+1)
- 谎报反馈：恶意买家有 30% 概率颠倒反馈
- 攻击模型动态生成好评/差评，模拟 P2P 中的恶意行为

### 7.3 仿真引擎（core/simulation.py）

```python
simulation = Simulation(network)
simulation.run(rounds=TRANSACTION_ROUNDS)  # 默认 2000 轮
```

每轮调用 `network.perform_transaction(timestamp)`，收集所有交易记录用于后续信任计算。

### 7.4 统一算法接口

所有算法均实现 `compute(network, ...)` 静态方法，屏蔽返回值差异：

```python
# 统一调用接口
def _compute(algo_class, network, track_convergence=False):
    ret = algo_class.compute(network, **kwargs)
    # 适配不同返回值格式（dataclass / tuple）
    if hasattr(ret, 'trust_vector'):
        return ret.trust_vector, ret.convergence_history, ret.compute_time
    if len(ret) == 3:
        return ret[0], ret[1], ret[2]
    return ret[0], None, ret[1]
```

---

## 八、数据集说明

> **数据集使用情况一览**：主实验主要使用 **Epinions**（含子图采样与四算法评估两个实验），**FilmTrust** 数据加载函数已实现但本项目未实际使用，**Bitcoin OTC** 同样**未纳入主实验流程**，仅作为参考脚本（`experiments/run_bitcoin_comparison.py`）保留。

### 8.1 Epinions（实际使用）

- **来源**：消费者评论网站 Epinions.com 的"信任"关系
- **格式**：每行 `trustor trustee value`，`value ∈ {-1, +1}`
- **规模**：~131K 用户，~84K 信任边
- **使用方式**：
  - **P2P 仿真**：取度数最高的 200 节点构成子图作为 P2P 网络拓扑（`core/network.py: TrustNetwork._build_dataset_graph`）
  - **四算法统一评估**：构造训练/测试边集合，按"正样本=真实信任边，负样本=未观测节点对"做链接预测评估

### 8.2 FilmTrust（已实现但本项目未实际使用）

- **来源**：电影评分网站 FilmTrust
- **格式**：
  - 评分：`user_id movie_id rating timestamp`
  - 信任：`trustor trustee trust_value`
- **规模**：~15K 评分，~1.8K 信任边
- **使用方式**：评分文件按 `ratings_*.txt` 模式自动合并加载
- **状态**：数据加载函数 `data_loader.load_filmtrust()` 已实现，但本项目主实验流程未实际调用该数据集。如需启用，可在 `core/network.py: TrustNetwork` 中新增 `_build_filmtrust_graph()` 构造方法。

### 8.3 Bitcoin OTC（**参考脚本保留，本项目未实际使用**）

- **来源**：Bitcoin OTC 平台的信任评价
- **格式**：CSV，`source target rating time`（rating ∈ [-10, 10]）
- **规模**：~6K 用户，~20K 条带时间戳的交易
- **理论使用方式**（**仅在参考脚本中体现**）：
  - rating 归一化到 [0, 1]：(rating + 10) / 20
  - 保留浮点 Unix 时间戳，用于**时间衰减机制**验证
  - 支持构建基线矩阵与衰减矩阵（`data_loader.build_bitcoin_otc_baseline` / `build_bitcoin_otc_with_decay`）
- **实际状态**：
  - 数据集已下载并放在 `data/soc-sign-bitcoinotc.csv/` 下
  - 参考脚本 `experiments/run_bitcoin_comparison.py` 可独立运行
  - **但 `main.py` 主流程并未实际调用该实验**，其结果不参与主结论
  - 若需纳入主流程，可在 `main.py` 中显式调用并收集结果

### 8.4 高效加载策略

大数据集采样（`core/network.py`）：

```python
# 最大节点数限制
MAX_NODES = 2000
top_nodes = set(n for n, _ in degree.most_common(MAX_NODES))

# 大矩阵自动启用稀疏表示
if n > 5000:
    from scipy.sparse import lil_matrix
    C = lil_matrix((n, n))
```

数据集仅在进程内加载一次（模块级缓存 `_dataset_cache`），避免重复磁盘 IO。

---

## 九、实验设计

### 9.1 实验总览

| 实验 | 脚本 | 实际是否纳入主实验 | 目的 | 输出 |
|------|------|--------------------|------|------|
| **烟雾测试** | [run_smoke_test.py](experiments/run_smoke_test.py) | ✓ 是 | 验证模块完整性 | 控制台 PASS/FAIL |
| **P2P 仿真基线** | [run_baselines.py](experiments/run_baselines.py) | ✓ 是 | 四算法在不同恶意比例下的对比 | 准确率、F1、效率、收敛曲线 |
| **攻击测试** | [run_attack_tests.py](experiments/run_attack_tests.py) | ✓ 是 | Sybil / Whitewashing 攻击下鲁棒性 | ROC-AUC、PR-AUC、Accuracy、F1 |
| **消融实验** | [run_ablation.py](experiments/run_ablation.py) | ✓ 是 | 各改进模块独立贡献度 | 6 种变体对比图 |
| **Epinions 评估** | [run_epinions_evaluation.py](experiments/run_epinions_evaluation.py) | ✓ 是 | 真实静态信任网络下四算法统一比较 | 完整指标对比 |
| **Bitcoin OTC 时间衰减** | [run_bitcoin_comparison.py](experiments/run_bitcoin_comparison.py) | ✗ **否（仅作参考脚本）** | 时间衰减机制在真实带时间戳数据上的效果 | 收敛迭代次数、L1/KL/JS 分布差异 |

### 9.2 P2P 仿真基线实验

**目标**：在受控仿真环境下比较四种算法的鲁棒性。

**实验设置**：
- **网络规模**：200 节点（固定）
- **拓扑类型**：Epinions 数据集子图（度数最高的 200 节点）
- **恶意节点比例**：`[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]`（共 7 个测试点）
- **交易轮数**：2000 轮（每节点平均 10 笔交易）
- **重复次数**：5 次取平均
- **攻击模式**：
  - 恶意卖家：总是提供差评（-1）
  - 恶意买家：30% 概率颠倒反馈

**评估方式**（排序法）：
- 按信任值升序排列
- 真实恶意节点数 = `num_mal`，将前 `num_mal` 个判定为恶意
- 计算 Accuracy / Precision / Recall / F1 / ROC-AUC / PR-AUC

**输出图表**：
- `efficiency_comparison.png`：算法效率（计算时间）
- `accuracy_comparison.png`：各攻击比例下的准确率
- `f1_comparison.png`：F1 分数对比
- `convergence_plot.png`：幂迭代收敛曲线

### 9.3 攻击测试实验

**目标**：评估算法在对抗性环境下的鲁棒性。

**两类攻击**：

#### 9.3.1 Sybil 攻击（女巫攻击）

**机制**：攻击者控制多个虚假身份相互交易，操纵信任评价。

**测试参数**：
- 团伙规模：`[2, 4, 6, 8, 10]` 节点/团伙
- 注入位置：基于攻击者（attacker）注册 Sybil 节点

**评估指标**：
- ROC-AUC、PR-AUC、Accuracy、F1
- 输出：`attack_sybil_{roc_auc|pr_auc|accuracy|f1}_comparison.png`

#### 9.3.2 Whitewashing 攻击（白洗攻击）

**机制**：节点在信任值低时丢弃旧身份注册新身份，重置历史。

**测试参数**：
- 白洗概率：`[0.05, 0.10, 0.15, 0.20, 0.25]`
- 行为惩罚：每次白洗后 `behavior_score *= 0.95`（累积惩罚）
- 多次白洗后初始信任度降低（0.5 → 0.4）

**评估指标**：与 Sybil 攻击相同
- 输出：`attack_whitewashing_*_comparison.png`

### 9.4 Bitcoin OTC 时间衰减对比（**参考脚本，未纳入主实验**）

> **状态声明**：本实验脚本 `experiments/run_bitcoin_comparison.py` **已实现但 `main.py` 主流程并未实际调用**，其结果**不参与**主结论。如下内容仅作参考。

**目标**：在带真实时间戳的数据上验证时间衰减机制的有效性。

**实验设计**：
1. **基线**：标准 EigenTrust（无时间衰减）
2. **改进版**：在 $C$ 矩阵构建中按 $e^{-\lambda \Delta t}$ 加权
3. **衰减系数扫描**：$\lambda \in [0.05, 0.1, 0.5]$

**对比指标**：
- 收敛迭代次数
- 最终信任向量的 L1 距离、L2 距离、KL 散度、JS 散度、皮尔逊相关性
- 收敛曲线对比

**运行方式**：

```bash
python -m experiments.run_bitcoin_comparison
```

**输出**（若单独运行该脚本）：
- `convergence_plot.png`：不同 $\lambda$ 下的收敛曲线
- `baseline_convergence.png`：基线收敛过程
- 终端输出详细的分布差异指标

### 9.5 消融实验

**目标**：验证改进版 EigenTrust 中各模块的独立贡献度。

**6 种变体**：

| 序号 | 变体 | 启用的模块 |
|------|------|------------|
| 1 | 经典 EigenTrust | 无（基线） |
| 2 | + 时间衰减 | `use_time_decay` |
| 3 | + 置信度折扣 | `use_confidence` |
| 4 | + Sybil 防御 | `use_sybil_defense` |
| 5 | + 自适应预信任 | `adaptive_trust_rate` |
| 6 | 完整改进版 | 全部启用 |

**三个实验场景**：
- `'sybil'`：仅注入 Sybil 攻击
- `'whitewashing'`：仅注入白洗攻击
- `'hybrid'`：混合攻击（默认）

**实现机制**：[AblationEigenTrust](experiments/run_ablation.py) 是 `ImprovedEigenTrust.compute()` 的薄包装（thin wrapper），通过 `enabled_modules` 字典控制四个模块的开关，确保与主实验完全一致。

**输出**：
- `ablation_comparison.png`：六种变体的指标对比
- `ablation_contribution.png`：各模块独立贡献度

### 9.6 Epinions 四算法统一评估

**目标**：在真实静态社交信任网络上对四算法做统一比较。

**数据集特点**：
- Epinions 是**静态社交信任网络**，一条信任边基本只对应一次静态评价
- 缺少交易上下文、多轮交易反馈、交易金额等信息
- 负样本从"未观测节点对"采样

**关键发现**（详见脚本注释）：
- PeerTrust 因缺少交易上下文，其依赖"反馈可信度+交易金额"的机制无法发挥
- 在静态信任网络中，EigenTrust/PageRank 基于全局结构传播更适配
- ImprovedEigenTrust 在保持排序性能上表现稳健

**实验目的**：证明改进算法的**通用性**（不局限于仿真场景）。

### 9.7 烟雾测试

**目的**：在主实验前快速验证所有模块可正常导入和基本运行。

**测试步骤**：
1. 导入 `core.*` 模块（Node, TrustNetwork, Simulation）
2. 导入 `models.*` 模块（4 种算法）
3. 导入 `experiments.run_baseline_validation`
4. 创建 5 节点最小网络 + 2 轮仿真
5. 调用 `EigenTrust.compute()` 验证返回
6. 调用 `ImprovedEigenTrust.compute()` 验证返回
7. 验证信任向量和为 1，长度匹配

任一步骤失败则 `main.py` 会打印警告但继续运行（避免单点失败阻塞全部实验）。

---

## 十、可视化输出

所有图表统一保存至 `results/figures/`，300 DPI，支持中文字体。

| 图表文件 | 内容 |
|----------|------|
| `efficiency_comparison.png` | 各算法计算时间对比 |
| `accuracy_comparison.png` | 准确率随攻击比例变化 |
| `f1_comparison.png` | F1 分数对比 |
| `convergence_plot.png` | 幂迭代收敛曲线 |
| `baseline_accuracy.png` / `baseline_f1.png` / `baseline_pr_auc.png` / `baseline_roc_auc.png` | 基线算法各指标 |
| `baseline_accuracy_bar.png` | 基线准确率柱状图 |
| `baseline_convergence.png` | 基线收敛曲线 |
| `baseline_efficiency.png` | 基线效率对比 |
| `attack_sybil_*_comparison.png` | Sybil 攻击下四指标对比 |
| `attack_whitewashing_*_comparison.png` | 白洗攻击下四指标对比 |
| `ablation_comparison.png` | 消融实验六变体对比 |
| `ablation_contribution.png` | 各模块贡献度 |
| `epinions_metrics_comparison.png` | Epinions 四算法统一评估 |

**配色方案**（`utils/visualization.py`）：
- 改进版 EigenTrust 使用**珊瑚红** `#f57c6e`（突出本文方法）
- EigenTrust 蓝色、PeerTrust 青绿、PageRank 橙黄
- 8 色调色板保持跨图表一致性

---

## 十一、实验参数与可复现性

### 11.1 全局配置（config.py）

```python
# 网络参数
NUM_NODES = 200              # 节点总数
GRAPH_TYPE = "dataset"       # "random" / "small_world" / "dataset"
DEGREE = 5                   # 平均度数

# 交易参数
TRANSACTION_ROUNDS = 2000    # 仿真轮数

# 攻击参数
ATTACK_RATIOS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
SYBIL_GROUP_SIZE = 5
WHITEWASH_PROB = 0.1

# 算法参数
EIGENTRUST_MAX_ITER = 50
EIGENTRUST_TOLERANCE = 1e-6
PAGERANK_DAMPING = 0.85

# 实验参数
REPEAT_TIMES = 5             # 每组参数重复次数
RANDOM_SEED = 42             # 随机种子
```

### 11.2 可复现性保证

1. **固定随机种子**：`RANDOM_SEED = 42`，每次实验以 `RANDOM_SEED + repeat` 作为种子
2. **统一评估标准**：所有算法使用相同的 `_evaluate_rank_based()` 函数
3. **代码隔离**：各算法独立实现，无交叉依赖，统一调用接口
4. **数据集缓存**：模块级 `_dataset_cache` 避免重复 IO 影响一致性
5. **参数透传**：消融实验使用薄包装，严格透传参数到 `ImprovedEigenTrust.compute()`

### 11.3 实验输出

- 终端输出详细的中间过程与汇总表
- 图表自动保存至 `results/figures/`
- 日志保存至 `results/logs/`

---

## 十二、依赖项

| 包 | 用途 | 最低版本 |
|----|------|----------|
| `numpy` | 数值计算、矩阵运算 | 1.19+ |
| `pandas` | 数据集加载（CSV） | 1.0+ |
| `networkx` | 图结构、PageRank 加速模式 | 2.4+ |
| `scikit-learn` | 评估指标（ROC-AUC、PR-AUC、F1） | 0.22+ |
| `scipy` | 稀疏矩阵支持（大数据集） | 1.5+ |
| `matplotlib` | 可视化 | 3.0+ |

---

## 十三、常见问题

**Q1：如何修改实验节点数 / 恶意比例？**

编辑 [config.py](config.py) 中的 `NUM_NODES` 和 `ATTACK_RATIOS`。

**Q2：如何更换数据集？**

修改 `config.py` 中的 `DATA_PATHS` 字典，并在 [core/network.py](core/network.py) 的 `_load_dataset()` 中添加对应解析逻辑。

**Q3：改进版 EigenTrust 与原版差异在哪里？**

主要在 `_build_improved_trust_matrix()` 与 `_power_iteration()` 两个方法中，前者加入时间衰减、置信度折扣、Sybil 防御，后者支持自适应预信任。

**Q4：为什么 PeerTrust 在 Epinions 上表现不佳？**

Epinions 是静态社交信任网络，缺少交易上下文、多轮反馈、交易金额，PeerTrust 的核心优势机制无法发挥。详见 [experiments/run_epinions_evaluation.py](experiments/run_epinions_evaluation.py) 头注释。

**Q5：如何单独验证某一改进模块的效果？**

使用消融实验的 `enabled_modules` 字典，例如：

```python
from experiments.run_ablation import AblationEigenTrust
AblationEigenTrust.compute(network, enabled_modules={'time_decay': True})
```

**Q6：Bitcoin OTC 数据集与时间衰减对比实验是否被使用？**

**没有**。本项目的主实验流程（`main.py`）**未实际调用** Bitcoin OTC 数据集与 `experiments/run_bitcoin_comparison.py`。该脚本仅作为参考实现保留，用于展示如何在带时间戳的真实数据上验证时间衰减机制。如需在主实验中加入该数据集，需在 `main.py` 中显式调用并收集结果到主报告。

**Q7：FilmTrust 数据集是否被使用？**

**没有**。`data_loader.load_filmtrust()` 函数已实现，但本项目主实验流程未实际使用 FilmTrust。如需启用，可在 `core/network.py: TrustNetwork` 中新增 `_build_filmtrust_graph()` 构造方法。

**Q8：运行 `main.py` 报错怎么办？**

依次检查：
1. 依赖是否完整安装
2. 数据集路径是否正确（参考 `config.py` 中 `DATA_PATHS`）
3. 先运行 `python -m experiments.run_smoke_test` 定位失败模块
4. 查看 `results/logs/` 下的日志

---

## 引用

本项目实现的核心算法基于以下三篇经典论文，使用 BibTeX 引用格式如下：

### [1] EigenTrust（基线 & 改进对象）

```bibtex
@inproceedings{kamvar2003eigentrust,
  title     = {The EigenTrust Algorithm for Reputation Management in P2P Networks},
  author    = {Kamvar, Sepandar D. and Schlosser, Mario T. and Garcia-Molina, Hector},
  booktitle = {Proceedings of the 12th International Conference on World Wide Web (WWW '03)},
  year      = {2003},
  pages     = {640--651},
  publisher = {ACM},
  doi       = {10.1145/775152.775242}
}
```

- **论文链接**：[The EigenTrust Algorithm for Reputation Management in P2P Networks](https://dl.acm.org/doi/10.1145/775152.775242)
- **DOI**：[10.1145/775152.775242](https://doi.org/10.1145/775152.775242)

### [2] PeerTrust

```bibtex
@article{xiong2004peertrust,
  author  = {Xiong, Li and Liu, Ling},
  title   = {PeerTrust: Supporting Reputation-Based Trust for Peer-to-Peer Electronic Communities},
  journal = {IEEE Transactions on Knowledge and Data Engineering},
  volume  = {16},
  number  = {7},
  pages   = {843--857},
  year    = {2004},
  doi     = {10.1109/TKDE.2004.1318566}
}
```

- **论文链接**：[PeerTrust: Supporting Reputation-Based Trust for Peer-to-Peer Electronic Communities](https://ieeexplore.ieee.org/document/1318566)
- **DOI**：[10.1109/TKDE.2004.1318566](https://doi.org/10.1109/TKDE.2004.1318566)

### [3] PageRank

```bibtex
@article{brin1998anatomy,
  author  = {Brin, Sergey and Page, Lawrence},
  title   = {The Anatomy of a Large-Scale Hypertextual Web Search Engine},
  journal = {Computer Networks and ISDN Systems},
  volume  = {30},
  number  = {1-7},
  pages   = {107--117},
  year    = {1998},
  doi     = {10.1016/S0169-7552(98)00110-X}
}
```

- **论文链接**：[The Anatomy of a Large-Scale Hypertextual Web Search Engine](https://infolab.stanford.edu/~backrub/google.html)
- **DOI**：[10.1016/S0169-7552(98)00110-X](https://doi.org/10.1016/S0169-7552(98)00110-X)

### [4] 本项目自身

```bibtex
@misc{trustsimulation2026,
  title  = {Trust Simulation Platform: Improved EigenTrust with Time Decay, Confidence Discount, Sybil Defense and Adaptive Pre-trust},
  author = {nimijRrrrrrq},
  year   = {2026},
  url    = {https://github.com/nimijRrrrrrq/Trust_Mechanism_Model_Simulation}
}
```

---

## 许可

本项目为综合实验课程作业，仅供学习与研究使用。
