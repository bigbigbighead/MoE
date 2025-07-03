import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from collections import Counter
from torch.utils.data import WeightedRandomSampler
import os

# 导入自定义的重采样函数
from data_loading_downsample import create_balanced_sampler, load_data, analyze_class_distribution


def plot_class_distribution(class_counts, title="类别分布", save_path=None):
    """绘制类别分布图"""
    plt.figure(figsize=(15, 8))

    # 类别ID和对应的计数
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    # 绘制条形图
    plt.bar(classes, counts)
    plt.xlabel("类别ID")
    plt.ylabel("样本数量")
    plt.title(title)
    plt.grid(axis='y', alpha=0.75)

    # 添加颜色区分不同层级的类别
    tier_colors = {
        (0, 49): 'blue',
        (50, 99): 'green',
        (100, 149): 'orange',
        (150, 199): 'red'
    }

    for (start, end), color in tier_colors.items():
        tier_classes = [c for c in classes if start <= c <= end]
        if tier_classes:
            tier_counts = [counts[classes.index(c)] for c in tier_classes]
            plt.bar(tier_classes, tier_counts, color=color, alpha=0.7,
                    label=f"类别 {start}-{end}")

    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"分布图已保存到 {save_path}")

    plt.show()


def plot_accuracy_distribution(csv_path="./results/AppClassNet/top200/ResNet/1/analysis/class_accuracy_ResNet.csv",
                               save_path=None):
    """绘制类别准确率分布图"""
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(15, 8))

    # 类别ID和对应的准确率
    classes = df['Class ID'].values
    accuracies = df['ResNet Accuracy (%)'].values

    # 绘制条形图
    plt.bar(classes, accuracies)
    plt.xlabel("类别ID")
    plt.ylabel("准确率(%)")
    plt.title("各类别准确率分布")
    plt.grid(axis='y', alpha=0.75)

    # 添加平均线
    plt.axhline(y=np.mean(accuracies), color='r', linestyle='-', label=f"平均准确率: {np.mean(accuracies):.2f}%")

    # 添加颜色区分不同层级的类别
    tier_colors = {
        (0, 49): 'blue',
        (50, 99): 'green',
        (100, 149): 'orange',
        (150, 199): 'red'
    }

    for (start, end), color in tier_colors.items():
        tier_indices = [(i, a) for i, (c, a) in enumerate(zip(classes, accuracies)) if start <= c <= end]
        if tier_indices:
            tier_idxs = [i for i, _ in tier_indices]
            tier_accs = [a for _, a in tier_indices]
            tier_cls = [classes[i] for i in tier_idxs]
            plt.bar(tier_cls, tier_accs, color=color, alpha=0.7,
                    label=f"类别 {start}-{end}, 平均: {np.mean(tier_accs):.2f}%")

    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"准确率分布图已保存到 {save_path}")

    plt.show()


def simulate_sampling(labels, strategy='tiered_importance', alpha=0.8, num_samples=10000):
    """模拟采样过程，分析重采样后的类别分布"""
    sampler = create_balanced_sampler(labels, strategy=strategy, alpha=alpha)

    # 模拟采样
    sampled_indices = list(sampler)[:num_samples]  # 采样指定数量的样本
    sampled_labels = labels[sampled_indices]

    # 计算采样后的类别分布
    sampled_counts = Counter(sampled_labels.numpy())

    return sampled_counts


def compare_sampling_strategies(labels, strategies=None, alpha=0.8, save_path=None):
    """比较不同采样策略的效果"""
    if strategies is None:
        strategies = ['inverse', 'sqrt', 'log', 'balanced', 'adaptive_performance', 'tiered_importance']

    original_counts = Counter(labels.numpy())

    plt.figure(figsize=(16, 10))

    # 为每种策略创建子图
    rows = len(strategies) // 2 + len(strategies) % 2
    cols = min(len(strategies), 2)

    for i, strategy in enumerate(strategies):
        plt.subplot(rows, cols, i + 1)

        # 模拟采样
        sampled_counts = simulate_sampling(labels, strategy=strategy, alpha=alpha)

        # 提取所有可能的类别ID
        all_classes = sorted(set(original_counts.keys()) | set(sampled_counts.keys()))

        # 获取原始和采样后的计数
        orig_counts = [original_counts.get(c, 0) for c in all_classes]
        samp_counts = [sampled_counts.get(c, 0) for c in all_classes]

        # 计算采样倍率
        sampling_ratio = []
        for orig, samp in zip(orig_counts, samp_counts):
            ratio = samp / max(1, orig) if orig > 0 else 0
            sampling_ratio.append(ratio)

        # 绘制采样倍率
        plt.bar(all_classes, sampling_ratio)
        plt.axhline(y=1.0, color='r', linestyle='--', label="原始分布比例")
        plt.xlabel("类别ID")
        plt.ylabel("采样倍率")
        plt.title(f"采样策略: {strategy}")
        plt.ylim(0, min(5, max(sampling_ratio) * 1.1))  # 限制y轴范围，但包含所有数据
        plt.grid(axis='y', alpha=0.5)

        # 添加颜色区分不同层级的类别
        tier_colors = {
            (0, 49): 'blue',
            (50, 99): 'green',
            (100, 149): 'orange',
            (150, 199): 'red'
        }

        for (start, end), color in tier_colors.items():
            tier_indices = [i for i, c in enumerate(all_classes) if start <= c <= end]
            if tier_indices:
                tier_classes = [all_classes[i] for i in tier_indices]
                tier_ratios = [sampling_ratio[i] for i in tier_indices]
                plt.bar(tier_classes, tier_ratios, color=color, alpha=0.7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"采样策略比较图已保存到 {save_path}")

    plt.show()


def main():
    """主函数：执行重采样分析"""
    # 创建保存目录
    os.makedirs("./results/sampling_analysis", exist_ok=True)

    # 加载训练数据
    train_x, train_y = load_data("train")

    # 分析原始类别分布
    class_counts = Counter(train_y.numpy())
    plot_class_distribution(
        class_counts,
        title="原始训练集类别分布",
        save_path="./results/sampling_analysis/original_distribution.png"
    )

    # 分析类别准确率分布
    plot_accuracy_distribution(
        save_path="./results/sampling_analysis/accuracy_distribution.png"
    )

    # 比较不同采样策略
    strategies = ['inverse', 'sqrt', 'adaptive_performance', 'tiered_importance']
    compare_sampling_strategies(
        train_y,
        strategies=strategies,
        save_path="./results/sampling_analysis/sampling_comparison.png"
    )

    # 详细分析最佳策略
    sampled_counts = simulate_sampling(train_y, strategy='tiered_importance')
    plot_class_distribution(
        sampled_counts,
        title="使用分层重要性采样后的类别分布",
        save_path="./results/sampling_analysis/tiered_importance_distribution.png"
    )

    print("重采样分析完成！")


if __name__ == "__main__":
    main()
