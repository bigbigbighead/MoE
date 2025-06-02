import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models.ResNet import resnet34, resnet18  # 修改为使用resnet34
import numpy as np
import os
import time
import datetime
import glob

# 数据集路径
DATASET_PATH = "./data/AppClassNet/top200"
RESULTS_PATH = "results/AppClassNet/top200/ResNet/6"  # 修改路径反映ResNet34

# 确保结果目录存在
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(f"{RESULTS_PATH}/param", exist_ok=True)
os.makedirs(f"{RESULTS_PATH}/logs", exist_ok=True)

# 创建日志文件
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"{RESULTS_PATH}/logs/training_log_{current_time}.txt"


# 日志记录函数
def log_message(message, log_file=LOG_FILE):
    """记录消息到日志文件"""
    with open(log_file, 'a') as f:
        f.write(f"{message}\n")
    print(message)


# 超参数
BATCH_SIZE = 2048  # 调小批量大小，因为ResNet34比ResNet18参数更多
EPOCHS = 1000
LEARNING_RATE = 0.001
NUM_CLASSES = 100  # 处理前n类


# 初始化新的ResNet18模型
def initialize_model(num_classes):
    """
    初始化新的ResNet34模型

    Args:
        num_classes: 分类数量

    Returns:
        model: 新初始化的ResNet34模型
    """
    model = resnet18(num_classes=num_classes)
    log_message(f"初始化新的ResNet18模型，分类数: {num_classes}")

    # 打印可训练参数数量
    total_params = sum(p.numel() for p in model.parameters())
    log_message(f"模型总参数: {total_params:,}")

    return model


# 加载数据集并筛选前NUM_CLASSES类
def load_data(split, top_num=50):
    x = np.load(f"{DATASET_PATH}/{split}_x.npy")
    y = np.load(f"{DATASET_PATH}/{split}_y.npy")

    # 检查并打印原始数据形状，以便调试
    message = f"{split} 原始数据形状: {x.shape}, 标签形状: {y.shape}"
    log_message(message)

    # 筛选前NUM_CLASSES类的数据
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
    log_message(f"=== 从头开始训练ResNet34，开始于 {current_time} ===")
    log_message(f"BatchSize: {BATCH_SIZE}, Learning Rate: {LEARNING_RATE}, Epochs: {EPOCHS}")
    log_message(f"数据集路径: {DATASET_PATH} (仅使用前{NUM_CLASSES}类)")
    log_message(f"结果保存路径: {RESULTS_PATH}")
    log_message(f"使用设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    train_x, train_y = load_data("train", NUM_CLASSES)
    valid_x, valid_y = load_data("valid", NUM_CLASSES)
    test_x, test_y = load_data("test", NUM_CLASSES)

    # 打印数据形状，用于调试
    log_message(f"最终数据形状 - 训练集: {train_x.shape}, 验证集: {valid_x.shape}, 测试集: {test_x.shape}")

    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    val_dataset = torch.utils.data.TensorDataset(valid_x, valid_y)
    test_dataset = torch.utils.data.TensorDataset(test_x, test_y)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        prefetch_factor=8
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
        prefetch_factor=8
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
        prefetch_factor=8
    )

    # 初始化模型 - 替换为从头初始化ResNet34
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize_model(NUM_CLASSES).to(device)

    # 损失函数和优化器 - 对所有参数进行优化，因为是从头训练
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 可以考虑添加学习率调度器，从头训练时很有用
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)

    start_epoch = 0
    best_val_acc = 0.0

    # 检查是否有检查点以继续训练
    start_epoch, best_val_acc, model, optimizer = load_checkpoint(None, model, optimizer)

    # 记录模型信息
    log_message(f"模型: ResNet34（从头训练）, 分类数: {NUM_CLASSES}")
    log_message("=" * 50)

    # 训练循环
    for epoch in range(start_epoch, EPOCHS):
        train_loss, train_acc, train_time = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_time = validate(model, val_loader, criterion, device)
        test_loss, test_acc, test_time = validate(model, test_loader, criterion, device)

        # 更新学习率调度器
        scheduler.step(val_loss)

        epoch_message = f"Epoch {epoch + 1}/{EPOCHS}"
        train_message = f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Time: {train_time:.2f}s"
        val_message = f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Time: {val_time:.2f}s"
        test_message = f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, Time: {test_time:.2f}s"
        lr_message = f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}"

        log_message(epoch_message)
        log_message(train_message)
        log_message(val_message)
        log_message(test_message)
        log_message(lr_message)
        log_message("-" * 50)

        # 保存模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 保存更多信息以便恢复训练
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),  # 保存学习率调度器状态
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
                'scheduler_state_dict': scheduler.state_dict(),  # 保存学习率调度器状态
                'best_val_acc': best_val_acc,
                'val_loss': val_loss,
                'train_loss': train_loss
            }
            torch.save(checkpoint, f"{RESULTS_PATH}/param/model_epoch_{epoch + 1}.pth")

    # 训练结束记录
    log_message(f"=== 训练结束于 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    log_message(f"最佳验证准确率: {best_val_acc:.2f}%")
