import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models.ResNet import resnet18
import numpy as np
import os
import time  # 添加time模块用于计时
import datetime  # 添加日期时间模块用于日志文件命名
import glob  # 添加glob模块用于查找文件
from utils.data_loading_downsample import create_balanced_sampler, load_data  # 导入重采样函数

# 数据集路径
DATASET_PATH = "./data/AppClassNet/top200"
RESULTS_PATH = "results/AppClassNet/top50/ResNet/6"  # 修改路径反映处理前100类
# 移除预训练模型路径配置
# PRETRAINED_MODEL_PATH = "./results/AppClassNet/top200/ResNet/1/param/model_epoch_800.pth"  # 预训练模型路径

# 确保结果目录存在
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(f"{RESULTS_PATH}/param", exist_ok=True)  # 确保参数存储目录存在
os.makedirs(f"{RESULTS_PATH}/logs", exist_ok=True)  # 确保日志存储目录存在

# 创建日志文件
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"{RESULTS_PATH}/logs/training_log_{current_time}.txt"


# 日志记录函数
def log_message(message, log_file=LOG_FILE):
    """记录消息到日志文件"""
    with open(log_file, 'a') as f:
        f.write(f"{message}\n")
    print(message)


# 新增重采样配置
# 可选策略：
# 'inverse' - 按类别频率的倒数加权
# 'sqrt' - 按类别频率的平方根的倒数加权
# 'log' - 按类别频率对数的倒数加权
# 'balanced' - 完全平衡采样
# 'adaptive_performance' - 基于模型性能和类别频率的自适应采样
# 'tiered_importance' - 基于分层重要性的采样
# 'top_n_suppression' - 降低前n个高频类别的采样率
# 'long_tail_boost' - 提升长尾类别的采样率
# 'boundary_focus' - 关注类别边界上的样本
# 'inverse_accuracy' - 按分类准确率的倒数加权
# 'dynamic_difficulty' - 按难易程度动态调整权重

SAMPLING_STRATEGY = 'top_n_suppression'  # 使用前N类抑制采样
SAMPLING_ALPHA = 0.8  # 重采样平衡因子
TOP_N_LIMIT = 10  # 前10个高频类别
TOP_N_FACTOR = 0.4  # 降低前N类的采样率至40%
RARE_BOOST_FACTOR = 2.0  # 稀有类提升因子

# 超参数
BATCH_SIZE = 1024
EPOCHS = 1000
LEARNING_RATE = 0.001
NUM_CLASSES = 200  # 修改为只处理前200类


# 初始化新模型函数 - 替换原来的load_pretrained_model函数
def initialize_new_model(num_classes=200):
    """
    初始化一个新的ResNet18模型，无需加载预训练参数

    Args:
        num_classes: 分类数量

    Returns:
        model: 新初始化的模型
    """
    # 初始化新模型
    model = resnet18(num_classes=num_classes)

    # 打印参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    log_message(f"模型参数信息 - 可训练参数: {trainable_params}, 总参数: {total_params}")

    return model


# 加载数据集并筛选前top_num类并应用重采样
def load_data_with_sampling(split, top_num=200, strategy=SAMPLING_STRATEGY, alpha=SAMPLING_ALPHA):
    x = np.load(f"{DATASET_PATH}/{split}_x.npy")
    y = np.load(f"{DATASET_PATH}/{split}_y.npy")

    # 检查并打印原始数据形状，以便调试
    message = f"{split} 原始数据形状: {x.shape}, 标签形状: {y.shape}"
    log_message(message)

    # 筛选前N类的数据
    mask = y < top_num
    x = x[mask]
    y = y[mask]

    message = f"{split} 筛选后数据形状 (仅前{top_num}类): {x.shape}, 标签形状: {y.shape}"
    log_message(message)

    # 根据输入数据的实际形状调整
    if len(x.shape) == 4:  # 如果已经是4D张量 [batch, channels, height, width]
        x = torch.tensor(x, dtype=torch.float32)
    elif len(x.shape) == 3:  # 如果是3D张量 [batch, height, width]
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)  # 添加通道维度
    else:
        # 如果是2D张量 [batch, features]，需要重塑为适合CNN的形状
        # 假设数据是1024维的特征向量
        x = torch.tensor(x, dtype=torch.float32).reshape(-1, 1, 1, x.shape[1])

    log_message(f"{split} 处理后数据形状: {x.shape}")
    y = torch.tensor(y, dtype=torch.long)

    # 只对训练集应用重采样
    if split == "train" and strategy:
        dataset = torch.utils.data.TensorDataset(x, y)
        # 确保csv路径存在
        csv_path = "./results/AppClassNet/top200/ResNet/1/analysis/class_accuracy_ResNet.csv"
        if not os.path.exists(csv_path):
            log_message(f"警告: 类别准确率数据文件不存在: {csv_path}, 将使用默认的'inverse'策略")
            strategy = 'inverse'

        # 根据选定的采样策略应用相应的参数
        if strategy == 'top_n_suppression':
            sampler = create_balanced_sampler(
                y, strategy=strategy, alpha=alpha,
                top_n_limit=TOP_N_LIMIT, top_n_factor=TOP_N_FACTOR
            )
            log_message(f"为{split}数据集创建了{strategy}重采样器，平衡因子: {alpha}, "
                        f"降低前{TOP_N_LIMIT}类采样率至{TOP_N_FACTOR}倍")
        elif strategy == 'long_tail_boost':
            sampler = create_balanced_sampler(
                y, strategy=strategy, alpha=alpha,
                rare_boost_factor=RARE_BOOST_FACTOR
            )
            log_message(f"为{split}数据集创建了{strategy}重采样器，平衡因子: {alpha}, "
                        f"稀有类提升因子: {RARE_BOOST_FACTOR}")
        elif strategy == 'boundary_focus':
            # 可以选择特定的边界类别，或让算法自动判断
            boundary_classes = None  # 自动判断
            sampler = create_balanced_sampler(
                y, strategy=strategy, alpha=alpha,
                boundary_classes=boundary_classes
            )
            log_message(f"为{split}数据集创建了{strategy}重采样器，平衡因子: {alpha}, "
                        f"自动选择边界类别")
        else:
            # 其他策略使用默认参数
            sampler = create_balanced_sampler(y, strategy=strategy, alpha=alpha)
            log_message(f"为{split}数据集创建了{strategy}重采样器，平衡因子: {alpha}")

        return dataset, sampler

    return x, y


# 训练和验证函数
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()  # 开始计时

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_time = time.time() - start_time  # 计算花费的时间
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc, epoch_time


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()  # 开始计时

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_time = time.time() - start_time  # 计算花费的时间
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc, epoch_time


# 查找并加载模型检查点
def load_checkpoint(checkpoint_path=None, model=None, optimizer=None):
    """
    加载模型检查点以继续训练

    Args:
        checkpoint_path: 模型检查点文件路径，如果为None则寻找最新的检查点
        model: 要加载参数的模型
        optimizer: 要加载状态的优化器

    Returns:
        start_epoch: 应该开始训练的轮次
        best_val_acc: 最佳验证准确率
        model: 加载了参数的模型
        optimizer: 加载了状态的优化器
    """
    start_epoch = 0
    best_val_acc = 0.0

    if checkpoint_path is None:
        # 寻找最新的epoch检查点
        checkpoint_files = glob.glob(f"{RESULTS_PATH}/param/model_epoch_*.pth")
        if checkpoint_files:
            # 提取轮次数并找到最大的
            epochs = [int(f.split('_')[-1].split('.')[0]) for f in checkpoint_files]
            max_epoch = max(epochs)
            checkpoint_path = f"{RESULTS_PATH}/param/model_epoch_{max_epoch}.pth"
            start_epoch = max_epoch  # 从下一个轮次开始
        else:
            # 如果没有epoch检查点，寻找最佳验证准确率的检查点
            best_model_files = glob.glob(f"{RESULTS_PATH}/param/best_model_*.pth")
            if best_model_files:
                # 提取准确率并找到最高的
                accuracies = [float(f.split('_')[-1].split('.pth')[0]) for f in best_model_files]
                best_idx = accuracies.index(max(accuracies))
                checkpoint_path = best_model_files[best_idx]
                best_val_acc = max(accuracies)
            else:
                log_message(f"未找到可用的检查点，将从头开始训练")
                return start_epoch, best_val_acc, model, optimizer

    # 检查文件是否存在
    if not os.path.isfile(checkpoint_path):
        log_message(f"检查点 {checkpoint_path} 不存在，将从头开始训练")
        return start_epoch, best_val_acc, model, optimizer

    # 加载检查点
    log_message(f"加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)

    # 检查检查点类型
    if isinstance(checkpoint, dict):
        # 完整检查点
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
            if 'best_val_acc' in checkpoint:
                best_val_acc = checkpoint['best_val_acc']
        else:
            # 仅模型参数
            model.load_state_dict(checkpoint)
    else:
        # 仅模型参数
        model.load_state_dict(checkpoint)

    log_message(f"成功加载检查点，从第 {start_epoch} 轮开始继续训练，当前最佳验证准确率: {best_val_acc:.2f}%")
    return start_epoch, best_val_acc, model, optimizer


if __name__ == "__main__":
    # 记录训练开始信息和配置信息
    log_message(f"=== 训练新 ResNet 模型开始于 {current_time} ===")
    log_message(f"BatchSize: {BATCH_SIZE}, Learning Rate: {LEARNING_RATE}, Epochs: {EPOCHS}")
    log_message(f"数据集路径: {DATASET_PATH} (仅使用前{NUM_CLASSES}类)")
    log_message(f"重采样策略: {SAMPLING_STRATEGY}, 平衡因子: {SAMPLING_ALPHA}")

    # 记录额外的采样策略参数
    if SAMPLING_STRATEGY == 'top_n_suppression':
        log_message(f"前{TOP_N_LIMIT}类采样抑制率: {TOP_N_FACTOR}")
    elif SAMPLING_STRATEGY == 'long_tail_boost':
        log_message(f"稀有类提升因子: {RARE_BOOST_FACTOR}")

    log_message(f"结果保存路径: {RESULTS_PATH}")
    log_message(f"模型初始化: 随机初始化（无预训练权重）")
    log_message(f"使用设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # 使用重采样加载训练数据
    train_dataset, train_sampler = load_data_with_sampling("train", NUM_CLASSES)
    valid_x, valid_y = load_data_with_sampling("valid", NUM_CLASSES)
    test_x, test_y = load_data_with_sampling("test", NUM_CLASSES)  # 加载测试集数据

    # 打印数据形状，用于调试
    log_message(
        f"最终数据形状 - 训练集: {len(train_dataset)} samples, 验证集: {valid_x.shape if hasattr(valid_x, 'shape') else len(valid_x)}, 测试集: {test_x.shape if hasattr(test_x, 'shape') else len(test_x)}")

    # 使用重采样器创建训练数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,  # 使用自适应重采样器
        num_workers=12,
        pin_memory=True,
        prefetch_factor=2
    )

    # 处理验证集数据
    if isinstance(valid_x, tuple) and isinstance(valid_x[0], torch.Tensor):
        # 如果是元组，说明是数据集和采样器
        val_dataset = valid_x[0]
    else:
        # 否则创建数据集
        val_dataset = torch.utils.data.TensorDataset(valid_x, valid_y)

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=12,
        pin_memory=True
    )

    # 处理测试集数据
    if isinstance(test_x, tuple) and isinstance(test_x[0], torch.Tensor):
        # 如果是元组，说明是数据集和采样器
        test_dataset = test_x[0]
    else:
        # 否则创建数据集
        test_dataset = torch.utils.data.TensorDataset(test_x, test_y)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=12,
        pin_memory=True
    )

    # 初始化全新的模型 - 不加载任何预训练权重
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize_new_model(NUM_CLASSES).to(device)

    # 损失函数和优化器 - 所有层都参与训练
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start_epoch = 0
    best_val_acc = 0.0

    # 记录模型信息
    log_message(f"模型: ResNet18（随机初始化），分类数: {NUM_CLASSES}")
    log_message(f"所有层参数都参与训练")
    log_message("=" * 50)

    # 训练循环
    for epoch in range(start_epoch, EPOCHS):
        train_loss, train_acc, train_time = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_time = validate(model, val_loader, criterion, device)
        test_loss, test_acc, test_time = validate(model, test_loader, criterion, device)

        epoch_message = f"Epoch {epoch + 1}/{EPOCHS}"
        train_message = f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Time: {train_time:.2f}s"
        val_message = f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Time: {val_time:.2f}s"
        test_message = f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, Time: {test_time:.2f}s"

        log_message(epoch_message)
        log_message(train_message)
        log_message(val_message)
        log_message(test_message)
        log_message("-" * 50)

        # 保存模型 (最好有个目录来存储)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 保存更多信息以便恢复训练
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'val_loss': val_loss,
                'train_loss': train_loss
            }
            torch.save(checkpoint, f"{RESULTS_PATH}/param/best_model_{val_acc:.2f}.pth")
            log_message(f"保存新的最佳模型，验证准确率: {val_acc:.2f}%")

        # 定期保存模型
        if (epoch + 1) % 10 == 0:
            # 同样保存更多信息以便恢复训练
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'val_loss': val_loss,
                'train_loss': train_loss
            }
            torch.save(checkpoint, f"{RESULTS_PATH}/param/model_epoch_{epoch + 1}.pth")

    # 训练结束记录
    log_message(f"=== 训练结束于 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    log_message(f"最佳验证准确率: {best_val_acc:.2f}%")
