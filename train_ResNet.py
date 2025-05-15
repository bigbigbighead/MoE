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

# 数据集路径
DATASET_PATH = "./data/AppClassNet/top200"
RESULTS_PATH = "./results/AppClassNet/top200/ResNet/1"

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


# 超参数
BATCH_SIZE = 1024
EPOCHS = 1000
LEARNING_RATE = 0.001
NUM_CLASSES = 200  # AppClassNet 类别数


# 加载数据集
def load_data(split):
    x = np.load(f"{DATASET_PATH}/{split}_x.npy")
    y = np.load(f"{DATASET_PATH}/{split}_y.npy")

    # 检查并打印数据形状，以便调试
    message = f"{split} data shape before processing: {x.shape}"
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

    log_message(f"{split} data shape after processing: {x.shape}")
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
    log_message(f"=== 训练开始于 {current_time} ===")
    log_message(f"BatchSize: {BATCH_SIZE}, Learning Rate: {LEARNING_RATE}, Epochs: {EPOCHS}")
    log_message(f"数据集路径: {DATASET_PATH}")
    log_message(f"结果保存路径: {RESULTS_PATH}")
    log_message(f"使用设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    train_x, train_y = load_data("train")
    valid_x, valid_y = load_data("valid")
    test_x, test_y = load_data("test")  # 加载测试集数据

    # 打印数据形状，用于调试
    log_message(f"Final shapes - Train: {train_x.shape}, Valid: {valid_x.shape}, Test: {test_x.shape}")

    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    val_dataset = torch.utils.data.TensorDataset(valid_x, valid_y)
    test_dataset = torch.utils.data.TensorDataset(test_x, test_y)  # 创建测试集数据集

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=12,  # 增加工作进程数
        pin_memory=True,  # 使用页锁定内存加速CPU到GPU的数据传输
        prefetch_factor=2  # 预加载批次数
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=12,  # 增加工作进程数
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=12,  # 增加工作进程数
        pin_memory=True
    )

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(num_classes=NUM_CLASSES).to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 添加继续训练选项
    RESUME_TRAINING = False  # 可以通过命令行参数或配置文件设置此值
    CHECKPOINT_PATH = None  # 可以指定特定的检查点文件，None表示使用最新的

    start_epoch = 0
    best_val_acc = 0.0

    # 如果继续训练，加载检查点
    if RESUME_TRAINING:
        start_epoch, best_val_acc, model, optimizer = load_checkpoint(CHECKPOINT_PATH, model, optimizer)

    # 记录模型信息
    log_message(f"模型: ResNet18, 分类数: {NUM_CLASSES}")
    if RESUME_TRAINING:
        log_message(f"继续训练: 从第 {start_epoch} 轮开始")
    log_message("=" * 50)

    # 训练循环
    for epoch in range(start_epoch, EPOCHS):  # 修改为从start_epoch开始
        train_loss, train_acc, train_time = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_time = validate(model, val_loader, criterion, device)
        test_loss, test_acc, test_time = validate(model, test_loader, criterion, device)  # 测试集验证

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
