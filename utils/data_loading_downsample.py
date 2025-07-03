import numpy as np
import torch
import os
import time
import datetime
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset, WeightedRandomSampler
from torchvision import datasets, transforms
import pandas as pd
from collections import Counter

# 数据集路径
DATASET_PATH = "./data/AppClassNet/top200"
RESULTS_PATH = "./results/AppClassNet/top200/MoE/77"

# 确保结果目录存在
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(f"{RESULTS_PATH}/param", exist_ok=True)
os.makedirs(f"{RESULTS_PATH}/logs", exist_ok=True)

# 创建日志文件
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"{RESULTS_PATH}/logs/training_log_{current_time}.txt"
# 优化超参数
BATCH_SIZE = 2048  # 批次大小
CLASS_RANGES = [(0, 199), (100, 199), (150, 199)]
NUM_CLASSES = 200
NUM_WORKERS = 4  # 数据加载的worker数量
PIN_MEMORY = True  # 确保启用pin_memory
PREFETCH_FACTOR = 2  # 增加预取因子

# 自动混合精度训练配置
USE_AMP = True  # 启用自动混合精度训练


# 日志记录函数
def log_message(message, log_file=LOG_FILE):
    """记录消息到日志文件"""
    with open(log_file, 'a') as f:
        f.write(f"{message}\n")
    print(message)


# 数据加载函数
def load_data(split, dataset_path=DATASET_PATH):
    """加载数据集"""
    x = np.load(f"{dataset_path}/{split}_x.npy", mmap_mode='r')
    y = np.load(f"{dataset_path}/{split}_y.npy")

    message = f"{split} data shape before processing: {x.shape}"
    log_message(message)

    # 根据输入数据的实际形状调整
    if len(x.shape) == 4:  # 如果已经是4D张量 [batch, channels, height, width]
        x = torch.tensor(x, dtype=torch.float32)
    elif len(x.shape) == 3:  # 如果是3D张量 [batch, height, width]
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)  # 添加通道维度
    else:
        # 如果是2D张量 [batch, features]，需要重塑为适合CNN的形状
        x = torch.tensor(x, dtype=torch.float32).reshape(-1, 1, 1, x.shape[1])

    log_message(f"{split} data shape after processing: {x.shape}")
    y = torch.tensor(y, dtype=torch.long)
    return x, y


# 新增函数：读取和分析类别分布数据
def analyze_class_distribution(csv_path="./results/AppClassNet/top200/ResNet/1/analysis/class_accuracy_ResNet.csv"):
    """读取CSV文件，分析类别分布"""
    try:
        # 尝试使用不同编码读取CSV文件
        encodings = ['utf-8', 'latin1', 'gbk', 'gb2312', 'gb18030', 'cp936', 'iso-8859-1']

        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(csv_path, encoding=encoding)
                log_message(f"Successfully loaded class distribution data from {csv_path} with encoding: {encoding}")
                break
            except UnicodeDecodeError:
                continue

        if df is None:
            raise ValueError(f"无法以任何已知编码读取文件: {csv_path}")

        return df
    except Exception as e:
        log_message(f"Error loading class distribution data: {e}")
        return None


# 修改函数：创建基于类别权重的采样器
def create_balanced_sampler(labels, strategy='adaptive_performance', alpha=1.0,
                            csv_path="./results/AppClassNet/top200/ResNet/1/analysis/class_accuracy_ResNet.csv",
                            top_n_limit=50, top_n_factor=0.5, rare_boost_factor=2.0, boundary_classes=None):
    """
    创建基于类别权重的采样器，支持多种采样策略

    参数:
    - labels: 训练数据的标签
    - strategy: 采样策略，可选:
        'inverse'(按类别频率的倒数加权)
        'sqrt'(按类别频率的平方根的倒数加权)
        'log'(按类别频率对数的倒数加权)
        'balanced'(完全平衡采样)
        'adaptive_performance'(基于模型性能和类别频率的自适应采样)
        'tiered_importance'(基于分层重要性的采样)
        'top_n_suppression'(降低前n个高频类别的采样率)
        'long_tail_boost'(提升长尾类别的采样率)
        'boundary_focus'(关注类别边界上的样本)
        'inverse_accuracy'(按分类准确率的倒数加权)
        'dynamic_difficulty'(按难易程度动态调整权重)
    - alpha: 平衡因子，控制重采样的强度（0-1之间，越接近1采样越均衡）
    - csv_path: 类别准确率数据的CSV文件路径
    - top_n_limit: 前n个高频类别的限制数量（用于'top_n_suppression'）
    - top_n_factor: 前n类采样比例（用于'top_n_suppression'）
    - rare_boost_factor: 稀有类提升因子（用于'long_tail_boost'）
    - boundary_classes: 需要重点关注的边界类别列表（用于'boundary_focus'）

    返回:
    - WeightedRandomSampler 实例
    """
    # 计算每个类别的样本数量
    class_counts = Counter(labels.numpy())
    num_samples = len(labels)
    class_ids = sorted(list(class_counts.keys()))
    max_class_count = max(class_counts.values())
    min_class_count = min(class_counts.values())

    # 读取类别准确率数据
    df = None
    if strategy in ['adaptive_performance', 'tiered_importance', 'inverse_accuracy', 'dynamic_difficulty']:
        df = analyze_class_distribution(csv_path)
        if df is None:
            log_message(f"Warning: 无法读取类别准确率数据，将回退到'inverse'策略")
            strategy = 'inverse'

    # 获取类别准确率映射
    class_accuracies = {}
    class_counts_from_csv = {}
    if df is not None:
        for _, row in df.iterrows():
            if 'Class ID' in row and 'ResNet Accuracy (%)' in row and 'Test Count' in row:
                class_id = int(row['Class ID'])
                accuracy = float(row['ResNet Accuracy (%)'])
                test_count = int(row['Test Count']) if not pd.isna(row['Test Count']) else 0
                class_accuracies[class_id] = accuracy / 100.0
                class_counts_from_csv[class_id] = test_count

    # 为每个样本分配权重
    weights = torch.zeros(num_samples)

    # 根据不同策略计算权重
    if strategy == 'inverse':
        # 按频率的倒数加权
        for idx, label in enumerate(labels):
            label_item = label.item()
            weights[idx] = 1.0 / max(1, class_counts[label_item])

    elif strategy == 'sqrt':
        # 按频率平方根的倒数加权（比inverse更温和）
        for idx, label in enumerate(labels):
            label_item = label.item()
            weights[idx] = 1.0 / np.sqrt(max(1, class_counts[label_item]))

    elif strategy == 'log':
        # 按频率对数的倒数加权（更温和的重采样）
        for idx, label in enumerate(labels):
            label_item = label.item()
            weights[idx] = 1.0 / np.log(1 + max(1, class_counts[label_item]))

    elif strategy == 'balanced':
        # 完全平衡采样
        for idx, label in enumerate(labels):
            label_item = label.item()
            weights[idx] = 1.0 / max(1, class_counts[label_item])

    elif strategy == 'adaptive_performance':
        # 基于模型性能的自适应采样 - 准确率低的类别获得更高权重
        for idx, label in enumerate(labels):
            label_item = label.item()
            # 获取准确率，默认为0.5（如果没有数据）
            accuracy = class_accuracies.get(label_item, 0.5)
            # 计算性能因子：准确率越低，权重越高
            performance_factor = 1.0 - accuracy
            # 计算频率因子：样本数量越少，权重越高
            frequency_factor = 1.0 / max(1, class_counts[label_item])
            # 结合性能因子和频率因子计算最终权重
            # 对性能因子进行平方以增强对准确率低的类别的重视
            weights[idx] = frequency_factor * (1 + performance_factor ** 2)

    elif strategy == 'tiered_importance':
        # 分层重要性采样 - 根据类别ID范围分配不同重要性权重
        tier_importance = {
            (0, 49): 1.2,  # 前50个类别重要性较高
            (50, 99): 1.0,  # 50-99类别重要性中等
            (100, 149): 0.9,  # 100-149类别重要性稍低
            (150, 199): 0.8  # 后50个类别重要性最低
        }

        for idx, label in enumerate(labels):
            label_item = label.item()
            # 基础权重仍然基于样本频率的倒数
            base_weight = 1.0 / max(1, class_counts[label_item])

            # 根据准确率调整 - 准确率低的类别权重更高
            accuracy = class_accuracies.get(label_item, 0.5)
            accuracy_factor = 1.0 + (1.0 - accuracy) ** 2

            # 根据类别ID范围应用重要性调整
            importance_factor = 1.0
            for (start, end), factor in tier_importance.items():
                if start <= label_item <= end:
                    importance_factor = factor
                    break

            # 最终权重是这三个因素的组合
            weights[idx] = base_weight * accuracy_factor * importance_factor

    # 新增策略1: 降低前n类高频样本的采样率
    elif strategy == 'top_n_suppression':
        # 按样本数排序类别，获取前N个高频类别
        sorted_classes = sorted(class_ids, key=lambda c: class_counts[c], reverse=True)
        top_n_classes = set(sorted_classes[:top_n_limit])

        log_message(f"使用top_n_suppression策略，将前{top_n_limit}个高频类别的权重乘以因子 {top_n_factor}")

        for idx, label in enumerate(labels):
            label_item = label.item()
            # 如果是高频类别，降低其权重
            if label_item in top_n_classes:
                weights[idx] = top_n_factor
            else:
                # 其他类别保持默认权重
                weights[idx] = 1.0

    # 新增策略2: 长尾分布提升
    elif strategy == 'long_tail_boost':
        # 计算类别频率的中位数作为阈值
        median_count = np.median(list(class_counts.values()))

        for idx, label in enumerate(labels):
            label_item = label.item()
            count = class_counts[label_item]
            # 基础权重
            base_weight = 1.0 / max(1, count)

            # 对于低于中位数的稀有类别，提高其权重
            if count < median_count:
                # 计算提升系数 - 越稀有提升越多
                boost = rare_boost_factor * (1 - count / median_count)
                weights[idx] = base_weight * (1 + boost)
            else:
                weights[idx] = base_weight

        log_message(f"使用long_tail_boost策略，稀有类提升因子: {rare_boost_factor}, 中位数阈值: {median_count}")

    # 新增策略3: 边界类别重点关注
    elif strategy == 'boundary_focus':
        # 如果没有指定边界类别，使用准确率在40%-60%之间的类别作为边界类别
        if boundary_classes is None and df is not None:
            boundary_classes = [
                class_id for class_id, accuracy in class_accuracies.items()
                if 0.4 <= accuracy <= 0.6
            ]

        # 如果仍然没有边界类别，使用排名在中间的类别
        if not boundary_classes:
            sorted_classes = sorted(class_ids, key=lambda c: class_counts[c])
            mid_point = len(sorted_classes) // 2
            boundary_classes = sorted_classes[mid_point - 5:mid_point + 5]  # 中间附近的10个类别

        log_message(f"使用boundary_focus策略，重点关注的边界类别: {boundary_classes}")

        for idx, label in enumerate(labels):
            label_item = label.item()
            # 基础权重是类别频率的倒数
            base_weight = 1.0 / max(1, class_counts[label_item])

            # 如果是边界类别，提高权重
            if label_item in boundary_classes:
                weights[idx] = base_weight * 2.0
            else:
                weights[idx] = base_weight

    # 新增策略4: 准确率倒数加权
    elif strategy == 'inverse_accuracy':
        for idx, label in enumerate(labels):
            label_item = label.item()
            # 获取准确率，默认为0.5（如果没有数据）
            accuracy = class_accuracies.get(label_item, 0.5)
            # 避免除以零，同时确保准确率低的类别获得更高权重
            accuracy_weight = 1.0 / max(0.1, accuracy)
            # 结合类别频率和准确率的倒数
            frequency_factor = 1.0 / max(1, class_counts[label_item])
            weights[idx] = frequency_factor * accuracy_weight

        log_message(f"使用inverse_accuracy策略，根据类别准确率的倒数进行加权")

    # 新增策略5: 动态难度加权
    elif strategy == 'dynamic_difficulty':
        # 计算每个类别的难度系数 = (1-准确率) * log(样本数)
        # 这样既考虑了准确率，也考虑了样本量 - 难以分类且样本较多的类别会获得更多关注
        difficulty_scores = {}
        for class_id in class_ids:
            accuracy = class_accuracies.get(class_id, 0.5)
            count = class_counts[class_id]
            # 难度分数 = 错误率 * log(样本数)
            difficulty_scores[class_id] = (1 - accuracy) * np.log(1 + count)

        # 归一化难度分数
        max_score = max(difficulty_scores.values())
        min_score = min(difficulty_scores.values())
        score_range = max(1e-5, max_score - min_score)  # 避免除以零

        for idx, label in enumerate(labels):
            label_item = label.item()
            # 归一化后的难度分数
            norm_score = (difficulty_scores[label_item] - min_score) / score_range
            # 基础权重是类别频率的倒数
            base_weight = 1.0 / max(1, class_counts[label_item])
            # 结合难度分数和基础权重
            weights[idx] = base_weight * (1 + norm_score)

        log_message(f"使用dynamic_difficulty策略，根据类别难度和样本量动态调整权重")

    # 应用平衡因子，实现部分平衡（alpha控制平衡程度）
    if alpha < 1.0:
        uniform_weights = torch.ones(num_samples) / num_samples
        weights = alpha * weights + (1 - alpha) * uniform_weights

    # 创建采样器，采样数量为原数据集大小
    sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)
    log_message(f"Created balanced sampler with strategy: {strategy}, alpha: {alpha}")

    return sampler


# 数据加载器
def get_dataloaders(class_ranges, dataset_path=DATASET_PATH, sampling_strategy='tiered_importance', alpha=0.8):
    # 加载数据集
    train_x, train_y = load_data("train")
    val_x, val_y = load_data("valid")
    test_x, test_y = load_data("test")

    # 创建按专家类别范围分割的数据集
    train_subsets = []
    val_subsets = []
    test_subsets = []

    # 将训练数据分割为每个专家负责的子集
    for start_class, end_class in class_ranges:
        # 训练集划分
        train_indices = torch.where((train_y >= start_class) & (train_y <= end_class))[0]
        train_subset_x = train_x[train_indices]
        train_subset_y = train_y[train_indices]
        train_subsets.append(TensorDataset(train_subset_x, train_subset_y))

        # 验证集划分
        val_indices = torch.where((val_y >= start_class) & (val_y <= end_class))[0]
        val_subset_x = val_x[val_indices]
        val_subset_y = val_y[val_indices]
        val_subsets.append(TensorDataset(val_subset_x, val_subset_y))

        # 测试集划分
        test_indices = torch.where((test_y >= start_class) & (test_y <= end_class))[0]
        test_subset_x = test_x[test_indices]
        test_subset_y = test_y[test_indices]
        test_subsets.append(TensorDataset(test_subset_x, test_subset_y))

    # 创建数据加载器（使用新的重采样策略）
    train_loaders = []
    for i, subset in enumerate(train_subsets):
        # 获取当前子集的标签
        _, subset_labels = subset[:]

        # 创建该子集的平衡采样器，使用更智能的重采样策略
        sampler = create_balanced_sampler(subset_labels, strategy=sampling_strategy, alpha=alpha)

        # 创建使用采样器的数据加载器
        train_loaders.append(DataLoader(
            subset,
            batch_size=BATCH_SIZE,
            sampler=sampler,  # 使用平衡采样器替代shuffle=True
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            prefetch_factor=PREFETCH_FACTOR,
            persistent_workers=False
        ))
        log_message(f"Created adaptive sampler for class range {class_ranges[i]} using strategy: {sampling_strategy}")

    # 创建验证集加载器
    val_loaders = []
    for subset in val_subsets:
        val_loaders.append(DataLoader(subset, batch_size=BATCH_SIZE * 2, shuffle=False,
                                      num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                                      prefetch_factor=PREFETCH_FACTOR, persistent_workers=False))

    # 创建测试集加载器
    test_loaders = []
    for subset in test_subsets:
        test_loaders.append(DataLoader(subset, batch_size=BATCH_SIZE * 2, shuffle=False,
                                       num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                                       prefetch_factor=PREFETCH_FACTOR, persistent_workers=False))

    # 完整数据集的加载器（同样使用自适应重采样）
    full_train_dataset = TensorDataset(train_x, train_y)
    # 创建完整数据集的平衡采样器
    full_sampler = create_balanced_sampler(train_y, strategy=sampling_strategy, alpha=alpha)
    full_train_loader = DataLoader(
        full_train_dataset,
        batch_size=BATCH_SIZE,
        sampler=full_sampler,  # 使用平衡采样器
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=False
    )
    log_message(f"Created adaptive DataLoader for full dataset with strategy: {sampling_strategy}")

    val_dataset = TensorDataset(val_x, val_y)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                            prefetch_factor=PREFETCH_FACTOR, persistent_workers=False)

    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                             prefetch_factor=PREFETCH_FACTOR, persistent_workers=False)

    return train_loaders, val_loaders, test_loaders, full_train_loader, val_loader, test_loader
