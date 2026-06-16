"""
data_loader.py - 数据集加载模块

提供统一的数据集加载接口，支持：
- Epinions 社交网络数据集
- FilmTrust 信任与评分数据集（多文件合并）
- Bitcoin OTC 交易数据集（浮点时间戳支持）

高效加载工具：
- build_eigentrust_matrix: 直接构建 EigenTrust 信任矩阵（适用于大数据集）
- build_pagerank_graph: 直接构建 PageRank 有向图（适用于大数据集）
"""

import os
import glob
import csv
import pandas as pd
import numpy as np
import networkx as nx


def load_epinions(file_path=None):
    """
    加载 Epinions 社交网络数据集
    
    文件格式：每行包含 trustor trustee trust_value
    trust_value: -1 (不信任) 或 1 (信任)
    
    Args:
        file_path: 数据集文件路径，默认使用 config.DATA_PATHS['epinions']
    
    Returns:
        list of tuples: [(trustor, trustee, trust_value), ...]
    """
    from config import DATA_PATHS
    
    if file_path is None:
        file_path = DATA_PATHS['epinions']
    
    edges = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                trustor = int(parts[0])
                trustee = int(parts[1])
                trust_value = int(parts[2]) if len(parts) >= 3 else 1
                edges.append((trustor, trustee, trust_value))
    
    return edges


def load_filmtrust(ratings_dir=None, trust_file=None):
    """
    加载 FilmTrust 数据集（评分 + 信任关系）
    
    评分文件：ratings_0.txt ~ ratings_3.txt，格式相同，纵向合并
    评分格式：user_id movie_id rating timestamp
    信任格式：trustor trustee trust_value
    
    Args:
        ratings_dir: 评分文件目录，默认使用 config.DATA_PATHS['filmtrust_ratings']
        trust_file: 信任关系文件，默认使用 config.DATA_PATHS['filmtrust_trust']
    
    Returns:
        tuple: (ratings, trust_edges)
            ratings: list of tuples [(user_id, movie_id, rating, timestamp), ...]
            trust_edges: list of tuples [(trustor, trustee, trust_value), ...]
    """
    from config import DATA_PATHS
    
    if ratings_dir is None:
        ratings_dir = DATA_PATHS['filmtrust_ratings']
    if trust_file is None:
        trust_file = DATA_PATHS['filmtrust_trust']
    
    # 遍历并合并所有 ratings_*.txt 文件
    ratings = []
    rating_files = glob.glob(os.path.join(ratings_dir, 'ratings_*.txt'))
    rating_files.sort()  # 按文件名排序，确保顺序一致
    
    for rf in rating_files:
        with open(rf, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    user_id = int(parts[0])
                    movie_id = int(parts[1])
                    rating = float(parts[2])
                    # FilmTrust 评分文件无时间戳
                    timestamp = int(parts[3]) if len(parts) >= 4 else 0
                    ratings.append((user_id, movie_id, rating, timestamp))
    
    # 加载信任关系
    trust_edges = []
    with open(trust_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                trustor = int(parts[0])
                trustee = int(parts[1])
                trust_value = float(parts[2])
                trust_edges.append((trustor, trustee, trust_value))
    
    return ratings, trust_edges


def load_bitcoin_otc(file_path=None):
    """
    加载 Bitcoin OTC 交易数据集（带时间戳）
    
    文件格式（CSV）：source,target,rating,timestamp
    timestamp: 浮点数 Unix 时间戳，如 1296629343.62073
    
    Args:
        file_path: 数据集文件路径，默认使用 config.DATA_PATHS['bitcoin_otc']
    
    Returns:
        list of tuples: [(source, target, rating, timestamp), ...]
    """
    from config import DATA_PATHS
    
    if file_path is None:
        file_path = DATA_PATHS['bitcoin_otc']
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    transactions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 4:
                try:
                    source = int(row[0])
                    target = int(row[1])
                    rating = float(row[2])
                    # 时间戳保留浮点精度，用于时间衰减计算
                    timestamp = float(row[3])
                    transactions.append((source, target, rating, timestamp))
                except (ValueError, IndexError):
                    # 跳过格式不正确的行
                    continue
    
    return transactions


def load_dataset(dataset_name, **kwargs):
    """
    统一的数据集加载接口
    
    Args:
        dataset_name: 数据集名称，可选 'epinions', 'filmtrust', 'bitcoin_otc'
        **kwargs: 传递给具体加载函数的参数
    
    Returns:
        对应数据集的数据结构
    """
    if dataset_name == 'epinions':
        return load_epinions(**kwargs)
    elif dataset_name == 'filmtrust':
        return load_filmtrust(**kwargs)
    elif dataset_name == 'bitcoin_otc':
        return load_bitcoin_otc(**kwargs)
    else:
        raise ValueError(f"未知数据集: {dataset_name}")


def build_eigentrust_matrix(edges, node_idx, n, use_sparse=False, max_nodes=10000):
    """
    直接构建 EigenTrust 行随机信任矩阵 C
    
    Args:
        edges: 边列表 [(trustor, trustee, trust_value), ...]
        node_idx: 节点到索引的映射 dict
        n: 节点数量
        use_sparse: 是否使用稀疏矩阵（适用于大数据集）
        max_nodes: 最大节点数限制（超过时采样）
        
    Returns:
        C: 行随机信任矩阵 (n x n numpy array 或 scipy.sparse matrix)
    """
    # 如果节点数太大，进行采样
    if n > max_nodes:
        # 只保留前 max_nodes 个节点
        selected_nodes = list(node_idx.keys())[:max_nodes]
        selected_set = set(selected_nodes)
        node_idx = {node: i for i, node in enumerate(selected_nodes)}
        n = max_nodes
        
        # 过滤边
        edges = [(t, te, v) for t, te, v in edges if t in selected_set and te in selected_set]
    
    if use_sparse and n > 5000:
        # 使用稀疏矩阵（适用于大数据集）
        try:
            from scipy.sparse import lil_matrix
            
            C = lil_matrix((n, n))
            row_sums = np.zeros(n)
            
            for trustor, trustee, trust_value in edges:
                if trustor in node_idx and trustee in node_idx:
                    i = node_idx[trustor]
                    j = node_idx[trustee]
                    if trust_value > 0:
                        C[i, j] += trust_value
                        row_sums[i] += trust_value
            
            # 行归一化
            for i in range(n):
                if row_sums[i] > 1e-12:
                    C[i, :] = C[i, :] / row_sums[i]
                else:
                    C[i, :] = 1.0 / n
            
            # 转换为 CSR 格式便于计算
            C = C.tocsr()
        except ImportError:
            # 如果没有 scipy，回退到稠密矩阵
            use_sparse = False
            print("  警告: scipy 未安装，回退到稠密矩阵")
    
    if not use_sparse:
        # 稠密矩阵
        C = np.zeros((n, n))
        row_sums = np.zeros(n)
        
        for trustor, trustee, trust_value in edges:
            if trustor in node_idx and trustee in node_idx:
                i = node_idx[trustor]
                j = node_idx[trustee]
                if trust_value > 0:
                    C[i, j] += trust_value
                    row_sums[i] += trust_value
        
        for i in range(n):
            if row_sums[i] > 1e-12:
                C[i, :] = C[i, :] / row_sums[i]
            else:
                C[i, :] = np.ones(n) / n
    
    return C


def build_pagerank_graph(edges, max_nodes=10000):
    """
    直接构建 PageRank 有向图
    
    Args:
        edges: 边列表 [(trustor, trustee, trust_value), ...]
        max_nodes: 最大节点数限制（超过时采样）
        
    Returns:
        G: networkx.DiGraph 有向图
    """
    # 先收集所有节点
    all_nodes = set()
    for trustor, trustee, _ in edges:
        all_nodes.add(trustor)
        all_nodes.add(trustee)
    
    # 如果节点数超过限制，进行采样
    if len(all_nodes) > max_nodes:
        selected_nodes = set(list(all_nodes)[:max_nodes])
        edges = [(t, te, v) for t, te, v in edges if t in selected_nodes and te in selected_nodes]
    
    G = nx.DiGraph()
    
    for trustor, trustee, trust_value in edges:
        # 只添加正信任关系作为边
        if trust_value > 0:
            G.add_edge(trustor, trustee)
    
    return G


def build_bitcoin_otc_baseline(file_path=None):
    """
    构建 Bitcoin OTC 基线信任矩阵（无时间衰减）
    
    将 rating ∈ [-10, 10] 归一化到 [0, 1]: (rating + 10) / 20
    
    Args:
        file_path: 数据集文件路径，默认使用 config.DATA_PATHS['bitcoin_otc']
    
    Returns:
        tuple: (C, node_idx, idx_node, df)
            C: 行随机信任矩阵
            node_idx: 节点到索引的映射
            idx_node: 索引到节点的映射
            df: 原始数据 DataFrame
    """
    from config import DATA_PATHS
    
    if file_path is None:
        file_path = DATA_PATHS['bitcoin_otc']
    
    # 使用 pandas 读取数据
    df = pd.read_csv(file_path, header=None, names=['source', 'target', 'rating', 'time'])
    
    # 获取所有节点
    nodes = sorted(set(df['source'].unique()) | set(df['target'].unique()))
    node_idx = {node: idx for idx, node in enumerate(nodes)}
    idx_node = {idx: node for idx, node in enumerate(nodes)}
    n = len(nodes)
    
    # 构建信任矩阵
    C = np.zeros((n, n))
    row_sums = np.zeros(n)
    
    for _, row in df.iterrows():
        i = node_idx[row['source']]
        j = node_idx[row['target']]
        # 将 rating ∈ [-10, 10] 归一化到 [0, 1]
        normalized_rating = (row['rating'] + 10) / 20
        if normalized_rating > 0:
            C[i, j] += normalized_rating
            row_sums[i] += normalized_rating
    
    # 行归一化
    for i in range(n):
        if row_sums[i] > 1e-12:
            C[i, :] = C[i, :] / row_sums[i]
        else:
            C[i, :] = np.ones(n) / n
    
    return C, node_idx, idx_node, df


def build_bitcoin_otc_with_decay(file_path=None, lambda_decay=0.1):
    """
    构建 Bitcoin OTC 信任矩阵（带时间衰减）
    
    Args:
        file_path: 数据集文件路径，默认使用 config.DATA_PATHS['bitcoin_otc']
        lambda_decay: 时间衰减系数（默认 0.1，单位：天^-1）
    
    Returns:
        tuple: (C, node_idx, idx_node, df)
            C: 行随机信任矩阵（带时间衰减）
            node_idx: 节点到索引的映射
            idx_node: 索引到节点的映射
            df: 原始数据 DataFrame
    """
    from config import DATA_PATHS
    
    if file_path is None:
        file_path = DATA_PATHS['bitcoin_otc']
    
    # 使用 pandas 读取数据
    df = pd.read_csv(file_path, header=None, names=['source', 'target', 'rating', 'time'])
    
    # 获取所有节点
    nodes = sorted(set(df['source'].unique()) | set(df['target'].unique()))
    node_idx = {node: idx for idx, node in enumerate(nodes)}
    idx_node = {idx: node for idx, node in enumerate(nodes)}
    n = len(nodes)
    
    # 获取最新时间戳
    max_time = df['time'].max()
    
    # 构建信任矩阵（带时间衰减）
    C = np.zeros((n, n))
    row_sums = np.zeros(n)
    
    for _, row in df.iterrows():
        i = node_idx[row['source']]
        j = node_idx[row['target']]
        
        # 计算时间衰减因子（按天衰减）
        time_diff_days = (max_time - row['time']) / (3600 * 24)
        decay_factor = np.exp(-lambda_decay * time_diff_days)
        
        # 将 rating ∈ [-10, 10] 归一化到 [0, 1]，再乘以衰减因子
        normalized_rating = (row['rating'] + 10) / 20 * decay_factor
        
        if normalized_rating > 0:
            C[i, j] += normalized_rating
            row_sums[i] += normalized_rating
    
    # 行归一化
    for i in range(n):
        if row_sums[i] > 1e-12:
            C[i, :] = C[i, :] / row_sums[i]
        else:
            C[i, :] = np.ones(n) / n
    
    return C, node_idx, idx_node, df


class TrustDataLoader:
    """
    信任数据集加载器类
    提供面向对象的数据集加载接口
    """
    
    def __init__(self):
        pass
    
    def load_epinions(self, file_path):
        """
        加载 Epinions 社交网络数据集
        
        Args:
            file_path: 数据集文件路径
            
        Returns:
            tuple: (edges, node_idx, idx_node, n)
                edges: 边列表 [(trustor, trustee, trust_value), ...]
                node_idx: 节点到索引的映射 dict
                idx_node: 索引到节点的映射 dict
                n: 节点数量
        """
        edges = []
        nodes = set()
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    trustor = int(parts[0])
                    trustee = int(parts[1])
                    trust_value = int(parts[2]) if len(parts) >= 3 else 1
                    edges.append((trustor, trustee, trust_value))
                    nodes.add(trustor)
                    nodes.add(trustee)
        
        node_list = sorted(list(nodes))
        node_idx = {node: idx for idx, node in enumerate(node_list)}
        idx_node = {idx: node for idx, node in enumerate(node_list)}
        n = len(node_list)
        
        return edges, node_idx, idx_node, n
    
    def load_filmtrust(self, ratings_dir, trust_file):
        """
        加载 FilmTrust 数据集
        
        Args:
            ratings_dir: 评分文件目录
            trust_file: 信任关系文件
            
        Returns:
            tuple: (ratings_df, trust_df)
                ratings_df: 评分数据 DataFrame
                trust_df: 信任关系 DataFrame
        """
        import pandas as pd
        
        rating_files = glob.glob(os.path.join(ratings_dir, 'ratings_*.txt'))
        if not rating_files:
            # 尝试不带目录的模式
            rating_files = glob.glob('ratings_*.txt')
        
        rating_files.sort()
        print(f"DEBUG: Found {len(rating_files)} rating files: {rating_files}")
        
        ratings_data = []
        for rf in rating_files:
            # 如果路径不存在，尝试从 ratings_dir 目录查找
            full_path = rf if os.path.exists(rf) else os.path.join(ratings_dir, rf)
            if not os.path.exists(full_path):
                print(f"WARNING: File not found: {full_path}")
                continue
                
            with open(full_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) >= 3:
                        user_id = int(parts[0])
                        movie_id = int(parts[1])
                        rating = float(parts[2])
                        timestamp = int(parts[3]) if len(parts) >= 4 and parts[3].isdigit() else 0
                        ratings_data.append({
                            'user_id': user_id,
                            'movie_id': movie_id,
                            'rating': rating,
                            'timestamp': timestamp
                        })
        
        ratings_df = pd.DataFrame(ratings_data)
        
        trust_data = []
        with open(trust_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    trustor = int(parts[0])
                    trustee = int(parts[1])
                    trust_value = float(parts[2])
                    trust_data.append({
                        'trustor': trustor,
                        'trustee': trustee,
                        'trust_value': trust_value
                    })
        
        trust_df = pd.DataFrame(trust_data)
        
        return ratings_df, trust_df
    
    def load_bitcoin_otc(self, file_path, lambda_decay=0.1):
        """
        加载 Bitcoin OTC 交易数据集，构建带时间衰减的信任矩阵
        
        Args:
            file_path: 数据集文件路径
            lambda_decay: 时间衰减系数（默认 0.1）
            
        Returns:
            tuple: (C, node_idx, idx_node, df)
                C: 信任矩阵 (numpy array)
                node_idx: 节点到索引的映射 dict
                idx_node: 索引到节点的映射 dict
                df: 原始数据 DataFrame
        """
        import pandas as pd
        import numpy as np
        
        df = pd.read_csv(file_path, header=None, names=['source', 'target', 'rating', 'time'])
        
        nodes = sorted(set(df['source'].unique()) | set(df['target'].unique()))
        node_idx = {node: idx for idx, node in enumerate(nodes)}
        idx_node = {idx: node for idx, node in enumerate(nodes)}
        n = len(nodes)
        
        C = np.zeros((n, n))
        row_sums = np.zeros(n)
        df_sorted = df.sort_values('time')
        max_time = df_sorted['time'].max()
        
        for _, row in df_sorted.iterrows():
            i = node_idx[row['source']]
            j = node_idx[row['target']]
            time_diff = max_time - row['time']
            decay_factor = np.exp(-lambda_decay * time_diff / (3600 * 24))
            # 将 rating ∈ [-10, 10] 归一化到 [0, 1]
            normalized_rating = (row['rating'] + 10) / 20
            C[i, j] += normalized_rating * decay_factor
            row_sums[i] += normalized_rating * decay_factor
        
        # 行归一化
        for i in range(n):
            if row_sums[i] > 1e-12:
                C[i, :] = C[i, :] / row_sums[i]
            else:
                C[i, :] = np.ones(n) / n
        
        return C, node_idx, idx_node, df


if __name__ == "__main__":
    # 测试加载各数据集
    print("测试加载 Epinions 数据集...")
    epinions_data = load_epinions()
    print(f"  加载边数: {len(epinions_data)}")
    
    print("\n测试加载 FilmTrust 数据集...")
    filmtrust_ratings, filmtrust_trust = load_filmtrust()
    print(f"  评分数量: {len(filmtrust_ratings)}")
    print(f"  信任边数: {len(filmtrust_trust)}")
    
    print("\n测试加载 Bitcoin OTC 数据集...")
    try:
        bitcoin_data = load_bitcoin_otc()
        print(f"  交易数量: {len(bitcoin_data)}")
        if bitcoin_data:
            print(f"  第一个时间戳(浮点): {bitcoin_data[0][3]}")
    except FileNotFoundError as e:
        print(f"  警告: {e}")
    
    print("\n所有数据集加载测试完成！")