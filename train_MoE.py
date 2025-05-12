import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models.MoE import MoEResNet18
import numpy as np
import os
import time
import datetime
import glob
import torch.nn.functional as F
# 添加多进程支持
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler
# 添加预取器
from torch.utils.data.dataloader import _utils

# 数据集路径
DATASET_PATH = "./data/AppClassNet/top200"
RESULTS_PATH = "./results/AppClassNet/top200/MoE/5"

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


# 优化超参数
BATCH_SIZE = 2048  # 增大batch size利用更多GPU内存
EPOCHS = 1000
LEARNING_RATE = 0.001
NUM_CLASSES = 200  # AppClassNet 类别数
NUM_EXPERTS = 3  # MoE专家头数量
ROUTING_TYPE = 'hard'  # 路由类型: 'softmax' 或 'hard'
NUM_WORKERS = 12  # 增加同时协作进程数量
PIN_MEMORY = True  # 确保启用pin_memory
PREFETCH_FACTOR = 8  # 增加预取因子

# 多样性损失权重
DIVERSITY_WEIGHT = 0.1  # 多样性损失权重
LOAD_BALANCING_WEIGHT = 0.01  # 负载均衡损失权重
SPECIALIZATION_WEIGHT = 0.1  # 专业化损失权重

# 自动混合精度训练配置
USE_AMP = True  # 启用自动混合精度训练


# 修改加载数据集函数，添加内存优化和预处理并行化
def load_data(split):
    x = np.load(f"{DATASET_PATH}/{split}_x.npy", mmap_mode='r')  # 使用内存映射加载大型数据
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


# 计算多样性损失
def diversity_loss(expert_outputs):
    """计算专家输出之间的多样性损失
    expert_outputs: [batch_size, num_experts, num_classes]
    返回: 专家输出之间的余弦相似度平均值 (值越低表示多样性越高)
    """
    num_experts = expert_outputs.size(1)
    batch_size = expert_outputs.size(0)

    # 将每个专家的输出归一化
    normalized_outputs = F.normalize(expert_outputs, p=2, dim=2)  # [batch, num_experts, num_classes]

    # 计算专家之间的相似度
    total_similarity = 0.0
    count = 0

    for i in range(num_experts):
        for j in range(i + 1, num_experts):
            # 计算余弦相似度 [batch]
            similarity = torch.sum(normalized_outputs[:, i] * normalized_outputs[:, j], dim=1)
            similarity = torch.abs(similarity)
            total_similarity += torch.sum(similarity)
            count += batch_size

    # 平均相似度 (值越低表示多样性越高)
    mean_similarity = total_similarity / max(count, 1)

    # 返回负相似度作为损失 (最大化多样性)
    return mean_similarity


# 计算专家专业化损失
def expert_specialization_loss(routing_weights, labels, num_classes, num_experts):
    """计算专家专业化损失，鼓励不同专家专注于不同类别
    routing_weights: [batch_size, num_experts]
    labels: [batch_size]
    num_classes: 类别总数
    num_experts: 专家数量
    返回: 专业化损失 (值越小表示专家越专注于特定类别)
    """
    batch_size = routing_weights.size(0)

    # 创建类别-专家关联矩阵 [num_classes, num_experts]
    # 记录每个类别对每个专家的偏好程度
    # 优化版本 - 使用矩阵运算代替循环
    class_expert_matrix = torch.zeros(num_classes, num_experts, device=routing_weights.device)
    class_expert_matrix.index_add_(0, labels, routing_weights)

    # 归一化
    class_sums = class_expert_matrix.sum(dim=1, keepdim=True).clamp(min=1e-8)
    class_expert_matrix = class_expert_matrix / class_sums

    # 计算专家间相关性矩阵
    expert_corr_matrix = torch.mm(class_expert_matrix.t(), class_expert_matrix)
    # 移除对角线(自相关)
    mask = 1.0 - torch.eye(num_experts, device=routing_weights.device)
    return (expert_corr_matrix * mask).sum() / (num_experts * (num_experts - 1))


# 优化的训练函数，使用自动混合精度
def train_one_epoch_moe(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_div_loss = 0.0
    running_spec_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()  # 开始计时

    for inputs, labels in loader:
        # 使用non_blocking=True加速CPU->GPU传输
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        total += labels.size(0)

        optimizer.zero_grad(set_to_none=True)  # 更快地重置梯度

        # 使用混合精度训练
        if USE_AMP and scaler is not None:
            with autocast():
                # 获取模型输出
                combined_logits, routing_weights, stacked_experts, _ = model(inputs)

                # 计算损失
                cls_loss = criterion(combined_logits, labels)
                div_loss = diversity_loss(stacked_experts)
                spec_loss = expert_specialization_loss(routing_weights, labels, NUM_CLASSES, model.num_experts)
                loss = cls_loss + DIVERSITY_WEIGHT * div_loss + SPECIALIZATION_WEIGHT * spec_loss

            # 使用scaler进行反向传播和优化
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 获取模型输出
            combined_logits, routing_weights, stacked_experts, _ = model(inputs)

            # 计算损失
            cls_loss = criterion(combined_logits, labels)
            div_loss = diversity_loss(stacked_experts)
            spec_loss = expert_specialization_loss(routing_weights, labels, NUM_CLASSES, model.num_experts)
            loss = cls_loss + DIVERSITY_WEIGHT * div_loss + SPECIALIZATION_WEIGHT * spec_loss

            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        running_cls_loss += cls_loss.item()
        running_div_loss += div_loss.item()
        running_spec_loss += spec_loss.item()

        # 计算准确率
        with torch.no_grad():
            _, predicted = combined_logits.max(1)
            correct += predicted.eq(labels).sum().item()

    epoch_time = time.time() - start_time
    epoch_loss = running_loss / len(loader)
    epoch_cls_loss = running_cls_loss / len(loader)
    epoch_div_loss = running_div_loss / len(loader)
    epoch_spec_loss = running_spec_loss / len(loader)
    epoch_acc = 100. * correct / total

    return {
        'loss': epoch_loss,
        'cls_loss': epoch_cls_loss,
        'div_loss': epoch_div_loss,
        'spec_loss': epoch_spec_loss,
        'acc': epoch_acc,
        'time': epoch_time
    }


# 优化的验证函数，添加混合精度支持
def validate_moe(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_div_loss = 0.0
    running_spec_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            total += labels.size(0)

            # 使用混合精度推理
            if USE_AMP:
                with autocast():
                    # 获取模型输出
                    combined_logits, routing_weights, stacked_experts, _ = model(inputs)

                    # 计算各项损失
                    cls_loss = criterion(combined_logits, labels)
                    div_loss = diversity_loss(stacked_experts)
                    spec_loss = expert_specialization_loss(routing_weights, labels, NUM_CLASSES, model.num_experts)
                    loss = cls_loss + DIVERSITY_WEIGHT * div_loss + SPECIALIZATION_WEIGHT * spec_loss
            else:
                # 获取模型输出
                combined_logits, routing_weights, stacked_experts, _ = model(inputs)

                # 计算各项损失
                cls_loss = criterion(combined_logits, labels)
                div_loss = diversity_loss(stacked_experts)
                spec_loss = expert_specialization_loss(routing_weights, labels, NUM_CLASSES, model.num_experts)
                loss = cls_loss + DIVERSITY_WEIGHT * div_loss + SPECIALIZATION_WEIGHT * spec_loss

            running_loss += loss.item()
            running_cls_loss += cls_loss.item()
            running_div_loss += div_loss.item()
            running_spec_loss += spec_loss.item()

            # 计算准确率
            _, predicted = combined_logits.max(1)
            correct += predicted.eq(labels).sum().item()

    epoch_time = time.time() - start_time
    epoch_loss = running_loss / len(loader)
    epoch_cls_loss = running_cls_loss / len(loader)
    epoch_div_loss = running_div_loss / len(loader)
    epoch_spec_loss = running_spec_loss / len(loader)
    epoch_acc = 100. * correct / total

    return {
        'loss': epoch_loss,
        'cls_loss': epoch_cls_loss,
        'div_loss': epoch_div_loss,
        'spec_loss': epoch_spec_loss,
        'acc': epoch_acc,
        'time': epoch_time
    }


# 查找并加载模型检查点
def load_checkpoint(checkpoint_path=None, model=None, optimizer=None):
    """加载模型检查点以继续训练"""
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
    # 设置NUMA绑定和多线程优化
    if torch.cuda.is_available():
        # 启用CUDA性能优化
        torch.backends.cudnn.benchmark = True
        # 如果输入尺寸固定，可以启用cudnn.deterministic以获得更好性能
        # torch.backends.cudnn.deterministic = True

    # 如果使用CPU，则设置OpenMP线程数
    if not torch.cuda.is_available():
        torch.set_num_threads(NUM_WORKERS)  # 设置适当的线程数

    # 记录训练开始信息和配置信息
    log_message(f"=== 训练开始于 {current_time} ===")
    log_message(f"BatchSize: {BATCH_SIZE}, Learning Rate: {LEARNING_RATE}, Epochs: {EPOCHS}")
    log_message(f"数据集路径: {DATASET_PATH}")
    log_message(f"结果保存路径: {RESULTS_PATH}")
    log_message(f"使用设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    log_message(f"自动混合精度: {USE_AMP}")
    log_message(f"工作进程数: {NUM_WORKERS}")
    log_message(f"MoE专家头数量: {NUM_EXPERTS}")
    log_message(f"路由类型: {ROUTING_TYPE}")
    log_message(f"多样性损失权重: {DIVERSITY_WEIGHT}")
    log_message(f"负载均衡损失权重: {LOAD_BALANCING_WEIGHT}")
    log_message(f"专业化损失权重: {SPECIALIZATION_WEIGHT}")

    # 配置数据并行训练，利用多个GPU
    multi_gpu = torch.cuda.device_count() > 1
    log_message(f"使用GPU数量: {torch.cuda.device_count()}")

    train_x, train_y = load_data("train")
    valid_x, valid_y = load_data("valid")
    test_x, test_y = load_data("test")

    # 打印数据形状，用于调试
    log_message(f"Final shapes - Train: {train_x.shape}, Valid: {valid_x.shape}, Test: {test_x.shape}")

    # 优化数据集加载
    # 为了减少内存使用，可以考虑自定义Dataset实现按需加载数据
    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    val_dataset = torch.utils.data.TensorDataset(valid_x, valid_y)
    test_dataset = torch.utils.data.TensorDataset(test_x, test_y)

    # 使用更多工作线程和更大的预取因子
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=True,  # 保持工作进程活跃，减少重启开销
        drop_last=True  # 丢弃不足一个批次的样本，可能提高训练效率
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,  # 验证时可以用更大的batch size
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=True
    )

    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化MoEResNet18模型，使用自定义ResNet作为主干
    # 获取输入数据形状，以正确配置模型
    input_channels = train_x.shape[1] if len(train_x.shape) >= 4 else 1
    input_height = train_x.shape[2] if len(train_x.shape) >= 4 else 1
    input_width = train_x.shape[3] if len(train_x.shape) >= 4 else train_x.shape[-1]

    log_message(f"模型输入形状: 通道={input_channels}, 高度={input_height}, 宽度={input_width}")

    model = MoEResNet18(
        NUM_CLASSES,
        num_experts=NUM_EXPERTS,
        routing_type=ROUTING_TYPE,
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width
    )

    # 创建混合精度训练的scaler
    scaler = GradScaler() if USE_AMP else None

    # 如果有多GPU，使用DataParallel
    if multi_gpu:
        log_message("启用多GPU并行训练")
        model = nn.DataParallel(model)

    model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    # 添加继续训练选项
    RESUME_TRAINING = False
    CHECKPOINT_PATH = None

    start_epoch = 0
    best_val_acc = 0.0

    # 如果继续训练，加载检查点
    if RESUME_TRAINING:
        start_epoch, best_val_acc, model, optimizer = load_checkpoint(CHECKPOINT_PATH, model, optimizer)

    # 记录模型信息
    log_message(f"模型: MoE-ResNet18, 分类数: {NUM_CLASSES}, 专家头数量: {NUM_EXPERTS}")
    if RESUME_TRAINING:
        log_message(f"继续训练: 从第 {start_epoch} 轮开始")
    log_message("=" * 50)

    # 训练循环
    for epoch in range(start_epoch, EPOCHS):
        # 训练
        train_metrics = train_one_epoch_moe(model, train_loader, criterion, optimizer, device, scaler)
        # 验证
        val_metrics = validate_moe(model, val_loader, criterion, device)
        # 测试
        test_metrics = validate_moe(model, test_loader, criterion, device)

        # 更新学习率
        scheduler.step(val_metrics['acc'])

        # 记录日志
        log_message(f"Epoch {epoch + 1}/{EPOCHS}")
        log_message(f"Train - Loss: {train_metrics['loss']:.4f}, CLS: {train_metrics['cls_loss']:.4f}, "
                    f"DIV: {train_metrics['div_loss']:.4f}, SPEC: {train_metrics['spec_loss']:.4f}, "
                    f"Acc: {train_metrics['acc']:.2f}%, Time: {train_metrics['time']:.2f}s")
        log_message(f"Valid - Loss: {val_metrics['loss']:.4f}, CLS: {val_metrics['cls_loss']:.4f}, "
                    f"DIV: {val_metrics['div_loss']:.4f}, SPEC: {val_metrics['spec_loss']:.4f}, "
                    f"Acc: {val_metrics['acc']:.2f}%, Time: {val_metrics['time']:.2f}s")
        log_message(f"Test  - Loss: {test_metrics['loss']:.4f}, CLS: {test_metrics['cls_loss']:.4f}, "
                    f"DIV: {test_metrics['div_loss']:.4f}, SPEC: {test_metrics['spec_loss']:.4f}, "
                    f"Acc: {test_metrics['acc']:.2f}%, Time: {test_metrics['time']:.2f}s")
        log_message("-" * 50)

        # 保存最佳模型
        if val_metrics['acc'] > best_val_acc:
            best_val_acc = val_metrics['acc']
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'val_loss': val_metrics['loss'],
                'train_loss': train_metrics['loss'],
                'diversity_weight': DIVERSITY_WEIGHT,
                'specialization_weight': SPECIALIZATION_WEIGHT,
                'num_experts': NUM_EXPERTS
            }
            torch.save(checkpoint, f"{RESULTS_PATH}/param/best_model_{val_metrics['acc']:.2f}.pth")
            log_message(f"保存新的最佳模型，验证准确率: {val_metrics['acc']:.2f}%")

        # 定期保存模型
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'val_loss': val_metrics['loss'],
                'train_loss': train_metrics['loss'],
                'diversity_weight': DIVERSITY_WEIGHT,
                'specialization_weight': SPECIALIZATION_WEIGHT,
                'num_experts': NUM_EXPERTS
            }
            torch.save(checkpoint, f"{RESULTS_PATH}/param/model_epoch_{epoch + 1}.pth")

    # 训练结束记录
    log_message(f"=== 训练结束于 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    log_message(f"最佳验证准确率: {best_val_acc:.2f}%")
