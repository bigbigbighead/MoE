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
RESULTS_PATH = "./results/AppClassNet/top50/ResNet/6"  # 修改路径反映处理前100类
# 移除预训练模型路径配置
# PRETRAINED_MODEL_PATH = "./results/AppClassNet/top200/ResNet/1/param/model_epoch_800.pth"  # 预训练模型路径

# 确保结果目录存在
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(f"{RESULTS_PATH}/param", exist_ok=True)  # 确保参数存储目录存在
os.makedirs(f"{RESULTS_PATH}/logs", exist_ok=True)  # 确保日志存储目录存在

# 创建日志文件
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"{RESULTS_PATH}/logs/training_log_{current_time}.txt"

# 配置是否自动恢复训练
AUTO_RESUME = True  # 如果有检查点则自动恢复训练


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

SAMPLING_STRATEGY = 'tiered_importance'  # 使用前N类抑制采样
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


# 增强的检查点搜索和加载函数
def find_latest_checkpoint():
    """
    寻找最新的检查点文件

    Returns:
        checkpoint_path: 最新检查点的路径，如果没找到则返回None
        is_epoch_checkpoint: 是否是epoch检查点（而非best检查点）
        checkpoint_epoch: 检查点的轮次，如果是best检查点则为0
        best_val_acc: 检查点中记录的最佳验���准确率，如果是epoch检查点则可能为0
    """
    epoch_checkpoints = glob.glob(f"{RESULTS_PATH}/param/model_epoch_*.pth")
    best_checkpoints = glob.glob(f"{RESULTS_PATH}/param/best_model_*.pth")

    latest_checkpoint_path = None
    is_epoch_checkpoint = False
    checkpoint_epoch = 0
    best_val_acc = 0.0

    # 检查是否有epoch检查点
    if epoch_checkpoints:
        # 提取轮次数并找到最大的
        try:
            epochs = [int(f.split('_')[-1].split('.')[0]) for f in epoch_checkpoints]
            max_epoch_idx = epochs.index(max(epochs))
            latest_checkpoint_path = epoch_checkpoints[max_epoch_idx]
            checkpoint_epoch = max(epochs)
            is_epoch_checkpoint = True
            log_message(f"找到最新的轮次检查点: {latest_checkpoint_path}，轮次: {checkpoint_epoch}")
        except Exception as e:
            log_message(f"解析epoch检查点文件名时出错: {e}")

    # 如果没有epoch检查点或解析出错，查找best模型检查点
    if latest_checkpoint_path is None and best_checkpoints:
        try:
            # 提取准确率并找到最高的
            accuracies = [float(f.split('_')[-1].split('.pth')[0]) for f in best_checkpoints]
            best_idx = accuracies.index(max(accuracies))
            latest_checkpoint_path = best_checkpoints[best_idx]
            best_val_acc = max(accuracies)
            log_message(f"找到最佳验证准确率检查点: {latest_checkpoint_path}，准确率: {best_val_acc:.2f}%")
        except Exception as e:
            log_message(f"解析best检查点文件名时出错: {e}")

    if latest_checkpoint_path is None:
        log_message("未找到任何可用的检查点")

    return latest_checkpoint_path, is_epoch_checkpoint, checkpoint_epoch, best_val_acc


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
        resumed: 是否成功恢复了训练
    """
    resumed = False
    start_epoch = 0
    best_val_acc = 0.0

    # 如果未指定检查点路径，寻找最新的检查点
    if checkpoint_path is None:
        checkpoint_path, is_epoch_checkpoint, checkpoint_epoch, best_acc = find_latest_checkpoint()
        if is_epoch_checkpoint:
            start_epoch = checkpoint_epoch
        if best_acc > 0:
            best_val_acc = best_acc

    # 如果没找到检查点，返回未初始化的状态
    if checkpoint_path is None or not os.path.isfile(checkpoint_path):
        log_message(f"未找到可用的检查点，将从头开始训练")
        return start_epoch, best_val_acc, model, optimizer, resumed

    # 尝试加载检查点
    try:
        log_message(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path,
                                map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # 检查检查点类型
        if isinstance(checkpoint, dict):
            # 完整检查点
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                log_message("成功加载模型参数")

                if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    log_message("成功加载优化器状态")

                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch']
                    log_message(f"将从第 {start_epoch} 轮开始继续训练")

                if 'best_val_acc' in checkpoint:
                    best_val_acc = checkpoint['best_val_acc']
                    log_message(f"记录的最佳验证准确率: {best_val_acc:.2f}%")

                # 显示额外的训练信息
                if 'val_loss' in checkpoint:
                    log_message(f"上一次验证损失: {checkpoint['val_loss']:.4f}")
                if 'train_loss' in checkpoint:
                    log_message(f"上一次训练损失: {checkpoint['train_loss']:.4f}")

                resumed = True
            else:
                # 仅模型参数
                model.load_state_dict(checkpoint)
                log_message("加载了仅含模型参数的检查点")
                resumed = True
        else:
            # 仅模型参数
            model.load_state_dict(checkpoint)
            log_message("加载了旧格式的模型检查点")
            resumed = True

        log_message(f"成功恢复训练，从第 {start_epoch} 轮开始，当前最佳验证准确率: {best_val_acc:.2f}%")

    except Exception as e:
        log_message(f"加载检查点时发生错误: {e}")
        log_message("将从头开始训练")

    return start_epoch, best_val_acc, model, optimizer, resumed


# 保存检查点
def save_checkpoint(model, optimizer, epoch, best_val_acc, val_loss, train_loss, test_acc=None, test_loss=None,
                    is_best=False):
    """
    保存训练检查点

    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前轮次
        best_val_acc: 最佳验证准确率
        val_loss: 验证损失
        train_loss: 训练损失
        test_acc: 测试准确率（可选）
        test_loss: 测试损失（可选）
        is_best: 是否是最佳模型
    """
    checkpoint = {
        'epoch': epoch + 1,  # 保存下一轮次的起点
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc,
        'val_loss': val_loss,
        'train_loss': train_loss
    }

    if test_acc is not None:
        checkpoint['test_acc'] = test_acc
    if test_loss is not None:
        checkpoint['test_loss'] = test_loss

    # 保存周期性检查点
    checkpoint_path = f"{RESULTS_PATH}/param/model_epoch_{epoch + 1}.pth"
    torch.save(checkpoint, checkpoint_path)

    # 如果是最佳模型，额外保存一份
    if is_best:
        best_path = f"{RESULTS_PATH}/param/best_model_{best_val_acc:.2f}.pth"
        torch.save(checkpoint, best_path)
        log_message(f"保存新的最佳模型检查点，验证准确率: {best_val_acc:.2f}%")


if __name__ == "__main__":
    # 记录训练开始信息和配置信息
    if AUTO_RESUME:
        log_message(f"=== ResNet 模型训练/恢复开始于 {current_time} ===")
        log_message(f"自动恢复训练: 已启用（如果找到检查点）")
    else:
        log_message(f"=== ResNet 模型训练开始于 {current_time} ===")
        log_message(f"自动恢复训练: 已禁用")

    log_message(f"BatchSize: {BATCH_SIZE}, Learning Rate: {LEARNING_RATE}, Epochs: {EPOCHS}")
    log_message(f"数据集路径: {DATASET_PATH} (仅使用前{NUM_CLASSES}类)")
    log_message(f"重采样策略: {SAMPLING_STRATEGY}, 平衡因子: {SAMPLING_ALPHA}")

    # 记录额外的采样策略参数
    if SAMPLING_STRATEGY == 'top_n_suppression':
        log_message(f"前{TOP_N_LIMIT}类采样抑制率: {TOP_N_FACTOR}")
    elif SAMPLING_STRATEGY == 'long_tail_boost':
        log_message(f"稀有类提升因子: {RARE_BOOST_FACTOR}")

    log_message(f"结果保存路径: {RESULTS_PATH}")
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

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize_new_model(NUM_CLASSES).to(device)

    # 损失函数和优化器 - 所有层都参与训练
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 初始化训练状态
    start_epoch = 0
    best_val_acc = 0.0
    training_resumed = False

    # 如果启用了自动恢复，尝试加载最近的检查点
    if AUTO_RESUME:
        start_epoch, best_val_acc, model, optimizer, training_resumed = load_checkpoint(
            checkpoint_path=None,  # 自动寻找最新检查点
            model=model,
            optimizer=optimizer
        )

    # 记录模型信息
    if training_resumed:
        log_message(f"模型: ResNet18（从检查点恢复），分类数: {NUM_CLASSES}")
        log_message(f"恢复训练自第 {start_epoch} 轮，当前最佳验证准确率: {best_val_acc:.2f}%")
    else:
        log_message(f"模型: ResNet18（随机初始化），分类数: {NUM_CLASSES}")
        log_message(f"从第 0 轮开始全新训练")

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

        # 检查是否达到新的最佳验证准确率
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        # 使用新函数保存检查点
        if (epoch + 1) % 10 == 0 or is_best:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_acc=best_val_acc,
                val_loss=val_loss,
                train_loss=train_loss,
                test_acc=test_acc,
                test_loss=test_loss,
                is_best=is_best
            )

            if (epoch + 1) % 10 == 0:
                log_message(f"保存周期性检查点，轮次: {epoch + 1}")

    # 训练结束记录
    log_message(f"=== 训练结束于 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    log_message(f"最佳验证准确率: {best_val_acc:.2f}%")
