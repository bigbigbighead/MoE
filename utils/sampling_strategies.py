"""
采样策略模块：提供多种重采样策略和数据分析功能以改善数据不平衡问题
"""

import numpy as np
import torch
import pandas as pd
from collections import Counter
from torch.utils.data import WeightedRandomSampler
import matplotlib.pyplot as plt
import os


class ClassDistributionAnalyzer:
    """分析类别分布和模型性能"""

    def __init__(self, csv_path):
        """
        初始化分析器

        参数:
        - csv_path: 包含类别分布和模型性能的CSV文件路径
        """
        self.csv_path = csv_path
        self.df = None
        self.class_counts = None
        self.class_accuracies = None
        self.load_data()

    def load_data(self):
        """加载CSV数据"""
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"Successfully loaded data from {self.csv_path}")

            # 提取类别数量和准确率
            self.class_counts = {}
            self.class_accuracies = {}

            for _, row in self.df.iterrows():
                if 'Class ID' in row and 'Train Count' in row and 'ResNet Accuracy (%)' in row:
                    class_id = int(row['Class ID'])
                    self.class_counts[class_id] = int(row['Train Count'])
                    self.class_accuracies[class_id] = float(row['ResNet Accuracy (%)'])
        except Exception as e:
            print(f"Error loading data: {e}")

    def plot_distribution(self, save_path=None):
        """
        可视化类别分布和准确率

        参数:
        - save_path: 图表保存路径，None表示不保存仅显示
        """
        if self.df is None:
            print("No data loaded.")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # 绘制类别样本数量
        class_ids = list(self.class_counts.keys())
        counts = list(self.class_counts.values())
        ax1.bar(class_ids, counts)
        ax1.set_title('Class Sample Distribution')
        ax1.set_xlabel('Class ID')
        ax1.set_ylabel('Sample Count')
        ax1.set_yscale('log')  # 使用对数尺度更容易查看不平衡

        # 绘制类别准确率
        accuracies = list(self.class_accuracies.values())
        ax2.bar(class_ids, accuracies, color='green')
        ax2.set_title('Class Accuracy Distribution')
        ax2.set_xlabel('Class ID')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_ylim(0, 100)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def get_class_weights(self, strategy='inverse', performance_based=False):
        """
        计算每个类别的权重

        参数:
        - strategy: 计算策略，'inverse'、'sqrt'或'log'
        - performance_based: 是否基于准确率调整权重

        返回:
        - class_weights: 字典，键为类别ID，值为权重
        """
        if self.class_counts is None:
            print("No data loaded.")
            return {}

        total_samples = sum(self.class_counts.values())
        class_weights = {}

        for class_id, count in self.class_counts.items():
            # 基于数量计算基础权重
            if strategy == 'inverse':
                weight = total_samples / count  # 频率倒数
            elif strategy == 'sqrt':
                weight = np.sqrt(total_samples / count)  # 平方根倒数，更温和
            elif strategy == 'log':
                weight = np.log(1 + total_samples / count)  # 对数倒数，更温和
            else:
                weight = 1.0  # 默认均等权重

            # 如果基于性能调整，则考虑准确率
            if performance_based and class_id in self.class_accuracies:
                accuracy = self.class_accuracies[class_id]
                # 准确率越低，权重越高
                weight *= (100 - min(accuracy, 99)) / 50  # 避免准确率100%导致权重为0

            class_weights[class_id] = weight

        return class_weights


class BalancedSamplingFactory:
    """创建各种平衡采样器"""

    @staticmethod
    def create_sampler(labels, class_weights=None, strategy='inverse', alpha=1.0):
        """
        创建加权随机采样器

        参数:
        - labels: 训练数据的标签
        - class_weights: 字典，键为类别ID，值为权重。如果为None则自动计算
        - strategy: 如果class_weights为None，则使用此策略计算权重
        - alpha: 平衡因子，控制重采样的强度

        返回:
        - WeightedRandomSampler实例
        """
        num_samples = len(labels)

        # 如果没有提供类别权重，则根据标签计算
        if class_weights is None:
            class_counts = Counter(labels.numpy())
            class_weights = {}

            if strategy == 'inverse':
                for cls, count in class_counts.items():
                    class_weights[cls] = 1.0 / count
            elif strategy == 'sqrt':
                for cls, count in class_counts.items():
                    class_weights[cls] = 1.0 / np.sqrt(count)
            elif strategy == 'log':
                for cls, count in class_counts.items():
                    class_weights[cls] = 1.0 / np.log(1 + count)
            else:
                for cls in class_counts:
                    class_weights[cls] = 1.0

        # 计算每个样本的权重
        weights = torch.zeros(num_samples)
        for idx, label in enumerate(labels):
            label_item = label.item()
            weights[idx] = class_weights.get(label_item, 1.0)

        # 应用平衡因子
        if alpha < 1.0:
            uniform_weights = torch.ones(num_samples) / num_samples
            weights = alpha * weights + (1 - alpha) * uniform_weights

        # 创建采样器
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=num_samples,
            replacement=True
        )

        return sampler


# 辅助函数：创建组合采样策略
def create_hybrid_sampling_strategy(labels, rare_threshold=10, common_threshold=100):
    """
    创建混合采样策略：
    - 对于稀有类别（样本数<rare_threshold）：过采样
    - 对于常见类别（样本数>common_threshold）：欠采样
    - 对于中间类别：保持原样

    参数:
    - labels: 训练数据的标签
    - rare_threshold: 定义稀有类别的阈值
    - common_threshold: 定义常见类别的阈值

    返回:
    - WeightedRandomSampler实例
    """
    class_counts = Counter(labels.numpy())
    num_samples = len(labels)

    # 计算目标采样数量（中间类别的平均样本数）
    middle_classes = [count for cls, count in class_counts.items()
                      if rare_threshold <= count <= common_threshold]
    target_count = int(np.mean(middle_classes)) if middle_classes else 50

    # 计算每个类别的权重
    class_weights = {}
    for cls, count in class_counts.items():
        if count < rare_threshold:
            # 稀有类别过采样
            class_weights[cls] = target_count / count
        elif count > common_threshold:
            # 常见类别欠采样
            class_weights[cls] = target_count / count
        else:
            # 中间类别保持不变
            class_weights[cls] = 1.0

    # 为每个样本分配权重
    weights = torch.zeros(num_samples)
    for idx, label in enumerate(labels):
        weights[idx] = class_weights[label.item()]

    # 创建采样器
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples,
        replacement=True
    )

    return sampler
