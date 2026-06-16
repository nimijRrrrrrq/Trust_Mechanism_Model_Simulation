# utils/logger.py
"""
日志模块 - 记录实验过程和结果
"""

import sys
import os
from datetime import datetime
from config import RANDOM_SEED

class Logger:
    """简单的日志记录器"""
    
    def __init__(self, log_dir="logs", verbose=True):
        self.verbose = verbose
        self.log_dir = log_dir
        self.log_file = None
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"experiment_{timestamp}.log")
    
    def log(self, message, level="INFO"):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] [{level}] {message}"
        
        # 写入文件
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(log_line + "\n")
        
        # 控制台输出
        if self.verbose:
            print(log_line)
    
    def info(self, message):
        self.log(message, "INFO")
    
    def warning(self, message):
        self.log(message, "WARNING")
    
    def error(self, message):
        self.log(message, "ERROR")
    
    def debug(self, message):
        if self.verbose:
            self.log(message, "DEBUG")
    
    def log_experiment_start(self, experiment_name):
        """记录实验开始"""
        self.info("=" * 60)
        self.info(f"实验开始: {experiment_name}")
        self.info(f"随机种子: {RANDOM_SEED}")
        self.info("=" * 60)
    
    def log_experiment_end(self):
        """记录实验结束"""
        self.info("=" * 60)
        self.info("实验结束")
        self.info(f"日志已保存: {self.log_path}")
        self.info("=" * 60)
    
    def log_config(self, config_dict):
        """记录配置参数"""
        self.info("实验配置:")
        for key, value in config_dict.items():
            self.info(f"  {key}: {value}")

# 全局日志实例
default_logger = Logger()

def get_logger():
    """获取全局日志实例"""
    return default_logger