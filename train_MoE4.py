import os
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from models.MoE import MoE4Model

# 数据集路径
DATASET_PATH = "./data/AppClassNet/top200"
RESULTS_PATH = "./results/AppClassNet/top200/MoE/4"

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
BATCH_SIZE = 2048  # 批次大小
EPOCHS_STAGE1 = 300  # 第一阶段训练轮数
EPOCHS_STAGE2 = 100  # 第二阶段训练轮数
LEARNING_RATE_STAGE1 = 0.001  # 第一阶段学习率
LEARNING_RATE_STAGE2 = 0.0001  # 第二阶段学习率
NUM_CLASSES = 200  # AppClassNet 类别数
NUM_EXPERTS = 3  # MoE专家头数量
ROUTING_TYPE = 'hard'  # 路由类型: 'softmax' 或 'hard'
NUM_WORKERS = 12  # 数据加载的worker数量
PIN_MEMORY = True  # 确保启用pin_memory
PREFETCH_FACTOR = 8  # 增加预取因子

# 自动混合精度训练配置
USE_AMP = True  # 启用自动混合精度训练


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
def get_dataloaders():
    # 加载数据集
    train_x, train_y = load_data("train")
    val_x, val_y = load_data("valid")
    test_x, test_y = load_data("test")

    # 创建按专家类别范围分割的数据集
    class_ranges = [(0, 99), (100, 149), (150, 199)]
    train_subsets = []

    # 将训练数据分割为每个专家负责的子集
    for start_class, end_class in class_ranges:
        indices = torch.where((train_y >= start_class) & (train_y <= end_class - 1))[0]
        subset_x = train_x[indices]
        subset_y = train_y[indices] - start_class  # 调整标签使其从0开始
        train_subsets.append(TensorDataset(subset_x, subset_y))

    # 创建数据加载器
    train_loaders = []
    for subset in train_subsets:
        train_loaders.append(DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True,
                                        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                                        prefetch_factor=PREFETCH_FACTOR, persistent_workers=True))

    # 完整数据集的加载器
    full_train_dataset = TensorDataset(train_x, train_y)
    full_train_loader = DataLoader(full_train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                   num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                                   prefetch_factor=PREFETCH_FACTOR, persistent_workers=True)

    val_dataset = TensorDataset(val_x, val_y)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                            prefetch_factor=PREFETCH_FACTOR, persistent_workers=True)

    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                             prefetch_factor=PREFETCH_FACTOR, persistent_workers=True)

    return train_loaders, full_train_loader, val_loader, test_loader


# 第一阶段训练：单独训练每个专家
def train_stage1(model, train_loaders, val_loader, test_loader, device):
    """
    第一阶段训练：分别训练每个专家
    """
    log_message(f"开始第一阶段训练...")
    writer = SummaryWriter(os.path.join(RESULTS_PATH, 'logs', 'stage1'))

    # 保存初始模型
    initial_state = model.state_dict()

    # 为每个专家创建优化器
    optimizers = []
    for i in range(len(model.experts)):
        # 只优化backbone和当前专家的参数
        params = list(model.backbone.parameters()) + list(model.experts[i].parameters())
        optimizer = optim.Adam(params, lr=LEARNING_RATE_STAGE1)
        optimizers.append(optimizer)

    # 创建混合精度训练的scaler
    scaler = GradScaler() if USE_AMP else None

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS_STAGE1):
        epoch_start_time = time.time()
        log_message(f"第一阶段训练 - Epoch {epoch + 1}/{EPOCHS_STAGE1}")
        model.train()
        total_epoch_train_time = 0

        # 分别训练每个专家
        for expert_idx, (optimizer, train_loader) in enumerate(zip(optimizers, train_loaders)):
            expert_epoch_start_time = time.time()
            log_message(f"    训练专家 {expert_idx + 1}/{len(model.experts)}")
            start_class, end_class = model.class_ranges[expert_idx]
            epoch_loss = 0
            correct = 0
            total = 0
            expert_batch_train_time = 0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                batch_start_time = time.time()
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)  # 更快地重置梯度

                # 使用混合精度训练
                if USE_AMP and scaler is not None:
                    with autocast():
                        # 前向传播
                        features = model.backbone(inputs)
                        outputs = model.experts[expert_idx](features)

                        # 计算损失
                        loss = criterion(outputs, targets)

                    # 使用scaler进行反向传播和优化
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # 前向传播
                    features = model.backbone(inputs)
                    outputs = model.experts[expert_idx](features)

                    # 计算损失
                    loss = criterion(outputs, targets)

                    # 反向传播与优化
                    loss.backward()
                    optimizer.step()

                batch_time_taken = time.time() - batch_start_time
                expert_batch_train_time += batch_time_taken
                # 统计
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader): # 减少日志频率
                #     log_message(
                #         f"    Expert {expert_idx} Batch [{batch_idx + 1}/{len(train_loader)}] Loss: {loss.item():.4f} Acc: {correct / total:.4f} Time: {batch_time_taken:.2f}s")

            # 记录每个专家的训练损失和准确率
            avg_loss = epoch_loss / len(train_loader)
            accuracy = correct / total
            expert_epoch_time_taken = time.time() - expert_epoch_start_time
            total_epoch_train_time += expert_epoch_time_taken
            writer.add_scalar(f'expert{expert_idx}/train_loss', avg_loss, epoch)
            writer.add_scalar(f'expert{expert_idx}/train_accuracy', accuracy, epoch)
            log_message(
                f"  \t专家{expert_idx} - 训练损失: {avg_loss:.4f}, 准确率: {accuracy:.4f}, 耗时: {expert_epoch_time_taken:.2f}s (Avg Batch: {expert_batch_train_time / len(train_loader):.2f}s)")

        epoch_time_taken = time.time() - epoch_start_time
        # log_message(
        #     f"第一阶段训练 - Epoch {epoch + 1}/{EPOCHS_STAGE1} 完成, 总训练耗时: {total_epoch_train_time:.2f}s, 总耗时: {epoch_time_taken:.2f}s")

        # 每个epoch结束后验证
        if (epoch + 1) % 5 == 0 or (epoch + 1) == EPOCHS_STAGE1:
            # 验证集评估
            val_results = validate_experts(model, val_loader, criterion, writer, epoch, device, "val")
            log_message(f"  验证集评估结果:")
            for expert_idx in range(len(model.experts)):
                if expert_idx in val_results['expert_accuracies']:
                    log_message(f"    \t专家{expert_idx} - 验证损失: {val_results['expert_losses'][expert_idx]:.4f}, "
                                f"准确率: {val_results['expert_accuracies'][expert_idx]:.4f}")

            # 测试集评估
            test_results = validate_experts(model, test_loader, criterion, writer, epoch, device, "test")
            log_message(f"  测试集评估结果:")
            for expert_idx in range(len(model.experts)):
                if expert_idx in test_results['expert_accuracies']:
                    log_message(f"    \t专家{expert_idx} - 测试损失: {test_results['expert_losses'][expert_idx]:.4f}, "
                                f"准确率: {test_results['expert_accuracies'][expert_idx]:.4f}")

        # 保存阶段性检查点
        if (epoch + 1) % 10 == 0 or (epoch + 1) == EPOCHS_STAGE1:
            checkpoint_path = os.path.join(RESULTS_PATH, 'param', f'stage1_epoch{epoch + 1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizers': [opt.state_dict() for opt in optimizers],
            }, checkpoint_path)
            log_message(f"已保存检查点到 {checkpoint_path}")

    # 保存最终的第一阶段模型
    final_checkpoint_path = os.path.join(RESULTS_PATH, 'param', 'stage1_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
    }, final_checkpoint_path)
    log_message(f"第一阶段训练完成，模型已保存到 {final_checkpoint_path}")
    writer.close()

    return model


def validate_experts(model, data_loader, criterion, writer, epoch, device, split="val"):
    """验证每个专家的性能"""
    model.eval()
    expert_losses = {i: 0 for i in range(len(model.experts))}
    expert_correct = {i: 0 for i in range(len(model.experts))}
    expert_total = {i: 0 for i in range(len(model.experts))}

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            features = model.backbone(inputs)

            # 对每个专家单独评估
            for expert_idx, expert in enumerate(model.experts):
                start_class, end_class = model.class_ranges[expert_idx]

                # 只评估该专家负责的类别范围内的样本
                mask = (targets >= start_class) & (targets <= end_class - 1)

                if mask.sum() > 0:
                    expert_inputs = features[mask]
                    expert_targets = targets[mask] - start_class  # 调整目标标签

                    # 获取专家输出
                    outputs = expert(expert_inputs)

                    # 计算损失
                    loss = criterion(outputs, expert_targets)
                    expert_losses[expert_idx] += loss.item() * mask.sum().item()

                    # 计算准确率
                    _, predicted = outputs.max(1)
                    expert_total[expert_idx] += mask.sum().item()
                    expert_correct[expert_idx] += predicted.eq(expert_targets).sum().item()

    # 记录每个专家的验证损失和准确率
    expert_accuracies = {}
    for expert_idx in range(len(model.experts)):
        if expert_total[expert_idx] > 0:
            expert_losses[expert_idx] = expert_losses[expert_idx] / expert_total[expert_idx]
            expert_accuracies[expert_idx] = expert_correct[expert_idx] / expert_total[expert_idx]
            writer.add_scalar(f'expert{expert_idx}/{split}_loss', expert_losses[expert_idx], epoch)
            writer.add_scalar(f'expert{expert_idx}/{split}_accuracy', expert_accuracies[expert_idx], epoch)

    return {
        'expert_losses': expert_losses,
        'expert_accuracies': expert_accuracies,
        'expert_correct': expert_correct,
        'expert_total': expert_total
    }


# 第二阶段训练：仅训练路由器
def train_stage2(model, train_loader, val_loader, test_loader, device):
    """
    第二阶段训练：冻结专家参数，只训练路由器
    """
    log_message(f"开始第二阶段训练...")
    writer = SummaryWriter(os.path.join(RESULTS_PATH, 'logs', 'stage2'))

    # 冻结backbone和专家参数
    for param in model.backbone.parameters():
        param.requires_grad = False
    for expert in model.experts:
        for param in expert.parameters():
            param.requires_grad = False

    # 确保路由器参数可训练
    for param in model.router.parameters():
        param.requires_grad = True

    # 创建优化器
    optimizer = optim.Adam(model.router.parameters(), lr=LEARNING_RATE_STAGE2)
    criterion = nn.CrossEntropyLoss()

    # 创建混合精度训练的scaler
    scaler = GradScaler() if USE_AMP else None

    for epoch in range(EPOCHS_STAGE2):
        epoch_start_time = time.time()
        log_message(f"第二阶段训练 - Epoch {epoch + 1}/{EPOCHS_STAGE2}")
        model.train()

        epoch_loss = 0
        correct = 0
        total = 0
        epoch_batch_train_time = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            batch_start_time = time.time()
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # 更快地重置梯度

            # 使用混合精度训练
            if USE_AMP and scaler is not None:
                with autocast():
                    # 前向传播
                    features = model.backbone(inputs)
                    routing_weights, _ = model.router(features)

                    # 计算每个专家的输出
                    expert_outputs = []
                    for expert_idx, expert in enumerate(model.experts):
                        start_class, end_class = model.class_ranges[expert_idx]
                        logits = expert(features)

                        # 将专家输出扩展到全类别空间
                        full_logits = torch.zeros(inputs.size(0), model.total_classes, device=inputs.device)
                        full_logits[:, start_class:end_class] = logits
                        expert_outputs.append(full_logits)

                    # 根据路由权重融合专家输出
                    combined_outputs = torch.zeros_like(expert_outputs[0])
                    for i, output in enumerate(expert_outputs):
                        combined_outputs += output * routing_weights[:, i].unsqueeze(1)

                    # 计算损失
                    loss = criterion(combined_outputs, targets)

                # 使用scaler进行反向传播和优化
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # 前向传播
                features = model.backbone(inputs)
                routing_weights, _ = model.router(features)

                # 计算每个专家的输出
                expert_outputs = []
                for expert_idx, expert in enumerate(model.experts):
                    start_class, end_class = model.class_ranges[expert_idx]
                    logits = expert(features)

                    # 将专家输出扩展到全类别空间
                    full_logits = torch.zeros(inputs.size(0), model.total_classes, device=inputs.device)
                    full_logits[:, start_class:end_class] = logits
                    expert_outputs.append(full_logits)

                # 根据路由权重融合专家输出
                combined_outputs = torch.zeros_like(expert_outputs[0])
                for i, output in enumerate(expert_outputs):
                    combined_outputs += output * routing_weights[:, i].unsqueeze(1)

                # 计算损失
                loss = criterion(combined_outputs, targets)

                # 反向传播和优化
                loss.backward()
                optimizer.step()

            batch_time_taken = time.time() - batch_start_time
            epoch_batch_train_time += batch_time_taken
            # 统计
            epoch_loss += loss.item()
            _, predicted = combined_outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):  # 减少日志频率
                log_message(
                    f"    Batch [{batch_idx + 1}/{len(train_loader)}] Loss: {loss.item():.4f} Acc: {correct / total:.4f} Time: {batch_time_taken:.2f}s")

        # 记录训练损失和准确率
        avg_loss = epoch_loss / len(train_loader)
        accuracy = correct / total
        epoch_time_taken = time.time() - epoch_start_time
        writer.add_scalar('train_loss', avg_loss, epoch)
        writer.add_scalar('train_accuracy', accuracy, epoch)
        log_message(
            f"  训练损失: {avg_loss:.4f}, 准确率: {accuracy:.4f}, 耗时: {epoch_time_taken:.2f}s (Avg Batch: {epoch_batch_train_time / len(train_loader):.2f}s)")

        # 验证集评估
        val_loss, val_accuracy, val_class_accuracies = validate_full_model(model, val_loader, criterion, device, "val")
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('val_accuracy', val_accuracy, epoch)

        # 测试集评估
        test_loss, test_accuracy, test_class_accuracies = validate_full_model(model, test_loader, criterion, device,
                                                                              "test")
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_accuracy', test_accuracy, epoch)

        # 记录每个类别区间的准确率
        for i, (start, end) in enumerate(model.class_ranges):
            writer.add_scalar(f'val_accuracy_classes_{start}-{end}', val_class_accuracies[i], epoch)
            writer.add_scalar(f'test_accuracy_classes_{start}-{end}', test_class_accuracies[i], epoch)

        # 打印验证和测试结果
        log_message(f"  验证损失: {val_loss:.4f}, 准确率: {val_accuracy:.4f}")
        log_message(f"  测试损失: {test_loss:.4f}, 准确率: {test_accuracy:.4f}")

        # 保存检查点
        if (epoch + 1) % 5 == 0 or (epoch + 1) == EPOCHS_STAGE2:
            checkpoint_path = os.path.join(RESULTS_PATH, 'param', f'stage2_epoch{epoch + 1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
            }, checkpoint_path)
            log_message(f"已保存检查点到 {checkpoint_path}")

    # 保存最终模型
    final_checkpoint_path = os.path.join(RESULTS_PATH, 'param', 'stage2_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'val_accuracy': val_accuracy,
        'test_accuracy': test_accuracy,
    }, final_checkpoint_path)
    log_message(f"第二阶段训练完成，模型已保存到 {final_checkpoint_path}")
    writer.close()

    return model


def validate_full_model(model, data_loader, criterion, device, split="val"):
    """验证完整模型性能，并返回每个类别区间的准确率"""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    # 为每个类别区间跟踪准确率
    class_correct = [0] * len(model.class_ranges)
    class_total = [0] * len(model.class_ranges)

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            # 使用混合精度推理
            if USE_AMP:
                with autocast():
                    # 使用推理模式
                    outputs = model.inference(inputs)

                    # 计算损失
                    loss = criterion(outputs, targets)
            else:
                # 使用推理模式
                outputs = model.inference(inputs)

                # 计算损失
                loss = criterion(outputs, targets)

            val_loss += loss.item()

            # 计算准确率
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 计算每个类别区间的准确率
            for i, (start, end) in enumerate(model.class_ranges):
                mask = (targets >= start) & (targets <= end - 1)
                if mask.sum() > 0:
                    class_total[i] += mask.sum().item()
                    class_correct[i] += predicted[mask].eq(targets[mask]).sum().item()

    # 计算平均损失和总体准确率
    avg_loss = val_loss / len(data_loader)
    accuracy = correct / total

    # 计算每个类别区间的准确率
    class_accuracies = [correct / total if total > 0 else 0 for correct, total in zip(class_correct, class_total)]

    log_message(f"{split.capitalize()} 损失: {avg_loss:.4f}, 准确率: {accuracy:.4f}")
    for i, (start, end) in enumerate(model.class_ranges):
        log_message(f"  类别 {start}-{end - 1} 准确率: {class_accuracies[i]:.4f} ({class_correct[i]}/{class_total[i]})")

    return avg_loss, accuracy, class_accuracies


if __name__ == "__main__":
    # 设置NUMA绑定和多线程优化
    if torch.cuda.is_available():
        # 启用CUDA性能优化
        torch.backends.cudnn.benchmark = True

    # 如果使用CPU，则设置OpenMP线程数
    if not torch.cuda.is_available():
        torch.set_num_threads(NUM_WORKERS)  # 设置适当的线程数

    # 记录训练开始信息和配置信息
    log_message(f"=== 训练开始于 {current_time} ===")
    log_message(f"BatchSize: {BATCH_SIZE}")
    log_message(f"第一阶段学习率: {LEARNING_RATE_STAGE1}, 训练轮数: {EPOCHS_STAGE1}")
    log_message(f"第二阶段学习率: {LEARNING_RATE_STAGE2}, 训练轮数: {EPOCHS_STAGE2}")
    log_message(f"数据集路径: {DATASET_PATH}")
    log_message(f"结果保存路径: {RESULTS_PATH}")
    log_message(f"使用设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    log_message(f"自动混合精度: {USE_AMP}")
    log_message(f"工作进程数: {NUM_WORKERS}")
    log_message(f"专家数量: {NUM_EXPERTS}")
    log_message(f"路由类型: {ROUTING_TYPE}")

    # 配置数据并行训练，利用多个GPU
    multi_gpu = torch.cuda.device_count() > 1
    log_message(f"使用GPU数量: {torch.cuda.device_count()}")

    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    train_loaders, full_train_loader, val_loader, test_loader = get_dataloaders()

    # 从训练数据中获取输入形状
    sample_input, _ = next(iter(full_train_loader))
    input_channels = sample_input.shape[1]
    input_height = sample_input.shape[2]
    input_width = sample_input.shape[3]

    log_message(f"模型输入形状: 通道={input_channels}, 高度={input_height}, 宽度={input_width}")

    # 创建模型
    model = MoE4Model(
        total_classes=NUM_CLASSES,
        class_ranges=[(0, 100), (100, 150), (150, 200)],
        routing_type=ROUTING_TYPE,
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width
    )

    # 如果有多GPU，使用DataParallel
    if multi_gpu:
        log_message("启用多GPU并行训练")
        model = nn.DataParallel(model)

    model.to(device)

    # 进行两阶段训练
    model = train_stage1(model, train_loaders, val_loader, test_loader, device)
    model = train_stage2(model, full_train_loader, val_loader, test_loader, device)

    log_message("训练完成！")
