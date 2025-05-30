import numpy as np
import torch
import os
import time
import datetime
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets, transforms

# 数据集路径
DATASET_PATH = "./data/AppClassNet/top200"
RESULTS_PATH = "./results/AppClassNet/top200/MoE/52"

# 确保结果目录存在
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(f"{RESULTS_PATH}/param", exist_ok=True)
os.makedirs(f"{RESULTS_PATH}/logs", exist_ok=True)

# 创建日志文件
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"{RESULTS_PATH}/logs/training_log_{current_time}.txt"
# 优化超参数
BATCH_SIZE = 2048  # 批次大小
CLASS_RANGES = [(0, 199), (100, 199), (150, 199)]  # 专家类别范围
NUM_CLASSES = 200
NUM_WORKERS = 4  # 数据加载的worker数量
PIN_MEMORY = True  # 确保启用pin_memory
PREFETCH_FACTOR = 4  # 增加预取因子

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


# 数据加载器
def get_dataloaders(class_ranges, dataset_path=DATASET_PATH):
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
        # train_subset_y = train_y[train_indices] - start_class  # 调整标签使其从0开始
        train_subsets.append(TensorDataset(train_subset_x, train_subset_y))

        # 验证集划分
        val_indices = torch.where((val_y >= start_class) & (val_y <= end_class))[0]
        val_subset_x = val_x[val_indices]
        val_subset_y = val_y[val_indices]
        # val_subset_y = val_y[val_indices] - start_class  # 调整标签使其从0开始
        val_subsets.append(TensorDataset(val_subset_x, val_subset_y))

        # 测试集划分
        test_indices = torch.where((test_y >= start_class) & (test_y <= end_class))[0]
        test_subset_x = test_x[test_indices]
        test_subset_y = test_y[test_indices]
        # test_subset_y = test_y[test_indices] - start_class  # 调整标签使其从0开始
        test_subsets.append(TensorDataset(test_subset_x, test_subset_y))

    # 创建数据加载器
    train_loaders = []
    for subset in train_subsets:
        train_loaders.append(DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True,
                                        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                                        prefetch_factor=PREFETCH_FACTOR, persistent_workers=False))

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

    # 完整数据集的加载器
    full_train_dataset = TensorDataset(train_x, train_y)
    full_train_loader = DataLoader(full_train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                   num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                                   prefetch_factor=PREFETCH_FACTOR, persistent_workers=False)

    val_dataset = TensorDataset(val_x, val_y)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                            prefetch_factor=PREFETCH_FACTOR, persistent_workers=False)

    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                             prefetch_factor=PREFETCH_FACTOR, persistent_workers=False)

    return train_loaders, val_loaders, test_loaders, full_train_loader, val_loader, test_loader
