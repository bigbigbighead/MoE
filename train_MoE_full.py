import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import json

from utils.data_loading_mine import get_dataloaders, CLASS_RANGES, NUM_CLASSES, USE_AMP, log_message, RESULTS_PATH

from models.MoE_full import MixtureOfExperts

# 全局训练参数
# 模型参数
MODEL_PARAMS = {
    "dropout_rate": 0.0,  # Dropout率
    "shared_backbone": False,  # 是否使用共享骨干网络
}

# 训练参数 - 粗分类器
COARSE_CLASSIFIER_PARAMS = {
    "num_epochs": 50,  # 最大训练轮数
    "patience": 5,  # 早停耐心值
    "learning_rate": 0.001,  # 学习率
    "weight_decay": 1e-5,  # 权重衰减
    "scheduler_step_size": 20,  # 学习率衰减周期
    "scheduler_gamma": 0.1,  # 学习率衰减因子
}

# 训练参数 - 专家网络
EXPERT_PARAMS = {
    "num_epochs": 100,  # 最大训练轮数
    "patience": 10,  # 早停耐心值
    "learning_rate": 0.001,  # 学习率
    "weight_decay": 1e-5,  # 权重衰减
    "scheduler_step_size": 20,  # 学习率衰减周期
    "scheduler_gamma": 0.1,  # 学习率衰减因子
}

# 训练参数 - 端到端微调
FINETUNE_PARAMS = {
    "num_epochs": 20,  # 最大训练轮数
    "learning_rate": 0.0001,  # 学习率
    "weight_decay": 1e-6,  # 权重衰减
    "alpha": 0.5,  # 粗分类器损失权重
    "beta": 0.5,  # 专家网络损失权重
}


# 检查模型参数文件是否存在
def check_model_exists(file_path):
    """检查模型参数文件是否存在"""
    return os.path.exists(file_path)


# 保存训练状态
def save_training_state(model, optimizer, scheduler, epoch, train_losses, val_accs, test_accs,
                        epoch_times, val_times, test_times, best_val_acc, epochs_no_improve,
                        stage, expert_idx=None):
    """
    保存训练状态，包括模型参数、优化器状态、训练历史等
    
    Args:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        epoch: 当前epoch
        train_losses: 训练损失历史
        val_accs: 验证集准确率历史
        test_accs: 测试集准确率历史
        epoch_times: 训练时间历史
        val_times: 验证时间历史
        test_times: 测试时间历史
        best_val_acc: 最佳验证准确率
        epochs_no_improve: 验证准确率未提升的epoch数
        stage: 训练阶段 ('coarse', 'expert', 'finetune')
        expert_idx: 专家索引 (仅在stage='expert'时需要)
    """
    state = {
        'epoch': epoch + 1,  # 保存下一个要训练的epoch
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_losses': train_losses,
        'val_accs': val_accs,
        'test_accs': test_accs,
        'epoch_times': epoch_times,
        'val_times': val_times,
        'test_times': test_times,
        'best_val_acc': best_val_acc,
        'epochs_no_improve': epochs_no_improve,
    }

    if stage == 'coarse':
        checkpoint_path = f"{RESULTS_PATH}/param/coarse_classifier_checkpoint.pth"
    elif stage == 'expert':
        checkpoint_path = f"{RESULTS_PATH}/param/expert_{expert_idx}_checkpoint.pth"
    elif stage == 'finetune':
        checkpoint_path = f"{RESULTS_PATH}/param/moe_full_checkpoint.pth"
    else:
        raise ValueError(f"未知的训练阶段: {stage}")

    torch.save(state, checkpoint_path)
    log_message(f"已保存训练状态到 {checkpoint_path}")


# 加载训练状态
def load_training_state(model, optimizer, scheduler, stage, expert_idx=None):
    """
    加载训练状态
    
    Args:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        stage: 训练阶段 ('coarse', 'expert', 'finetune')
        expert_idx: 专家索引 (仅在stage='expert'时需要)
    
    Returns:
        tuple: (开始的epoch, 训练损失历史, 验证准确率历史, 测试准确率历史, 
                训练时间历史, 验证时间历史, 测试时间历史, 最佳验证准确率, 未提升的epoch数)
               如果未找到检查点则返回(0, [], [], [], [], [], [], 0.0, 0)
    """
    if stage == 'coarse':
        checkpoint_path = f"{RESULTS_PATH}/param/coarse_classifier_checkpoint.pth"
    elif stage == 'expert':
        checkpoint_path = f"{RESULTS_PATH}/param/expert_{expert_idx}_checkpoint.pth"
    elif stage == 'finetune':
        checkpoint_path = f"{RESULTS_PATH}/param/moe_full_checkpoint.pth"
    else:
        raise ValueError(f"未知的训练阶段: {stage}")

    if not check_model_exists(checkpoint_path):
        log_message(f"未找到训练检查点: {checkpoint_path}，将从头开始训练")
        return 0, [], [], [], [], [], [], 0.0, 0

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch']
    train_losses = checkpoint['train_losses']
    val_accs = checkpoint['val_accs']
    test_accs = checkpoint['test_accs']
    epoch_times = checkpoint['epoch_times']
    val_times = checkpoint['val_times']
    test_times = checkpoint['test_times']
    best_val_acc = checkpoint['best_val_acc']
    epochs_no_improve = checkpoint['epochs_no_improve']

    log_message(f"已加载训练检查点，将从epoch {start_epoch}继续训练")
    log_message(f"检查点信息 - 最佳验证准确率: {best_val_acc:.2f}%, 未提升epoch数: {epochs_no_improve}")

    return start_epoch, train_losses, val_accs, test_accs, epoch_times, val_times, test_times, best_val_acc, epochs_no_improve


def train_coarse_classifier(model, optimizer, criterion, full_train_loader, val_loader, test_loader, device,
                            num_epochs=COARSE_CLASSIFIER_PARAMS["num_epochs"],
                            patience=COARSE_CLASSIFIER_PARAMS["patience"],
                            scheduler=None, use_amp=USE_AMP):
    """
    训练粗分类器
    
    Args:
        model: MixtureOfExperts模型
        optimizer: 优化器
        criterion: 损失函数
        full_train_loader: 完整训练集加载器
        val_loader: 验证集加载器
        test_loader: 测试集加载器
        device: 计算设备
        num_epochs: 训练轮数
        patience: 早停耐心值
        scheduler: 学习率调度器
        use_amp: 是否使用混合精度训练
    """
    # 加载之前的训练状态（如果有）
    start_epoch, train_losses, val_accs, test_accs, epoch_times, val_times, test_times, best_val_acc, epochs_no_improve = load_training_state(
        model, optimizer, scheduler, 'coarse')

    model.train()

    # 混合精度训练
    scaler = GradScaler() if use_amp else None

    log_message("开始训练粗分类器...")

    # 如果从头开始训练，则输出最佳val_acc初始值
    if start_epoch == 0:
        log_message(f"初始最佳验证准确率: {best_val_acc:.2f}%")

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        # 训练循环
        for inputs, targets in tqdm(full_train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} 训练"):
            inputs, targets = inputs.to(device), targets.to(device)

            # 获取粗分类器的目标标签（专家索引）
            expert_targets = model.get_expert_target(targets)

            optimizer.zero_grad()

            if use_amp:
                with autocast():
                    # 只获取粗分类器输出
                    outputs = model(inputs, mode='coarse_only')
                    loss = criterion(outputs, expert_targets)

                # 混合精度训练的反向传播
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # 常规训练
                outputs = model(inputs, mode='coarse_only')
                loss = criterion(outputs, expert_targets)
                loss.backward()
                optimizer.step()

            # 记录精度和损失
            epoch_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(expert_targets).sum().item()

        if scheduler:
            scheduler.step()

        # 计算训练损失和准确率
        train_loss = epoch_loss / len(full_train_loader.dataset)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)

        epoch_train_time = time.time() - epoch_start_time
        epoch_times.append(epoch_train_time)

        # 立即输出训练结果
        log_message(
            f"Epoch {epoch + 1}/{num_epochs} - 训练完成 - "
            f"Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% - "
            f"Time: {epoch_train_time:.1f}s"
        )

        # 在验证集上评估
        val_start_time = time.time()
        val_acc = validate_coarse_classifier(model, val_loader, device, use_amp)
        val_time = time.time() - val_start_time
        val_times.append(val_time)
        val_accs.append(val_acc)

        # 立即输出验证结果
        log_message(
            f"Epoch {epoch + 1}/{num_epochs} - 验证完成 - "
            f"Acc: {val_acc:.2f}% - Time: {val_time:.1f}s"
        )

        # 在测试集上评估
        test_start_time = time.time()
        test_acc = validate_coarse_classifier(model, test_loader, device, use_amp)
        test_time = time.time() - test_start_time
        test_times.append(test_time)
        test_accs.append(test_acc)

        # 立即输出测试结果
        log_message(
            f"Epoch {epoch + 1}/{num_epochs} - 测试完成 - "
            f"Acc: {test_acc:.2f}% - Time: {test_time:.1f}s"
        )

        # 输出总结
        log_message(
            f"Epoch {epoch + 1}/{num_epochs} - 总结 - "
            f"Train: {train_acc:.2f}%, Val: {val_acc:.2f}%, Test: {test_acc:.2f}%"
        )

        # 早停检查
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            # 保存最佳模型
            torch.save(model.state_dict(), f"{RESULTS_PATH}/param/coarse_classifier_best.pth")
            log_message(f"保存新的最佳模型，验证准确率: {val_acc:.2f}%")
        else:
            epochs_no_improve += 1
            log_message(f"验证准确率未提升 {epochs_no_improve}/{patience} epochs")

        # 保存训练状态，以便下次继续训练
        save_training_state(model, optimizer, scheduler, epoch, train_losses, val_accs, test_accs,
                            epoch_times, val_times, test_times, best_val_acc, epochs_no_improve, 'coarse')

        if epochs_no_improve == patience:
            log_message(f"早停！验证准确率 {patience} epochs 未提升")
            break

    # 可视化训练过程
    plt.figure(figsize=(15, 10))
    # 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(train_losses)
    plt.title('Coarse Classifier Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # 精度曲线
    plt.subplot(2, 2, 2)
    plt.plot(val_accs, label='Validation')
    plt.plot(test_accs, label='Test')
    plt.title('Coarse Classifier Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # 耗时曲线
    plt.subplot(2, 2, 3)
    plt.plot(epoch_times, label='Train')
    plt.plot(val_times, label='Validation')
    plt.plot(test_times, label='Test')
    plt.title('Time per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{RESULTS_PATH}/coarse_classifier_training.png")

    return best_val_acc


def validate_coarse_classifier(model, data_loader, device, use_amp=USE_AMP):
    """验证或测试粗分类器性能"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            expert_targets = model.get_expert_target(targets)

            if use_amp:
                with autocast():
                    outputs = model(inputs, mode='coarse_only')
            else:
                outputs = model(inputs, mode='coarse_only')

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(expert_targets).sum().item()

    accuracy = 100. * correct / total
    return accuracy


def train_expert(model, expert_idx, optimizer, criterion, train_loader, val_loader, test_loader, device,
                 num_epochs=EXPERT_PARAMS["num_epochs"],
                 patience=EXPERT_PARAMS["patience"],
                 scheduler=None, use_amp=USE_AMP):
    """
    训练单个专家网络
    
    Args:
        model: MixtureOfExperts模型
        expert_idx: 专家索引
        optimizer: 优化器
        criterion: 损失函数
        train_loader: 专家训练数据加载器
        val_loader: 专家验证数据加载器
        test_loader: 专家测试数据加载器
        device: 计算设备
        num_epochs: 训练轮数
        patience: 早停耐心值
        scheduler: 学习率调度器
        use_amp: 是否使用混合精度训练
    """
    # 加载之前的训练状态（如果有）
    start_epoch, train_losses, val_accs, test_accs, epoch_times, val_times, test_times, best_val_acc, epochs_no_improve = load_training_state(
        model, optimizer, scheduler, 'expert', expert_idx)

    # 混合精度训练
    scaler = GradScaler() if use_amp else None

    log_message(f"开始训练专家 {expert_idx}...")

    # 如果从头开始训练，则输出最佳val_acc初始值
    if start_epoch == 0:
        log_message(f"初始最佳验证准确率: {best_val_acc:.2f}%")

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        # 训练模式
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        # 训练循环
        for inputs, targets in tqdm(train_loader, desc=f"Expert {expert_idx} - Epoch {epoch + 1}/{num_epochs} 训练"):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            if use_amp:
                with autocast():
                    # 对于共享骨干网络，我们需要单独训练每个专家头
                    if model.shared_backbone:
                        features = model.backbone(inputs)
                        outputs = model.expert_heads[expert_idx](features)
                    else:
                        outputs = model.experts[expert_idx](inputs)

                    loss = criterion(outputs, targets)

                # 混合精度训练的反向传播
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # 常规训练
                if model.shared_backbone:
                    features = model.backbone(inputs)
                    outputs = model.expert_heads[expert_idx](features)
                else:
                    outputs = model.experts[expert_idx](inputs)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            # 记录精度和损失
            epoch_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        if scheduler:
            scheduler.step()

        # 计算训练损失和准确率
        train_loss = epoch_loss / len(train_loader.dataset)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)

        epoch_train_time = time.time() - epoch_start_time
        epoch_times.append(epoch_train_time)

        # 立即输出训练结果
        log_message(
            f"Expert {expert_idx} - Epoch {epoch + 1}/{num_epochs} - 训练完成 - "
            f"Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% - "
            f"Time: {epoch_train_time:.1f}s"
        )

        # 在验证集上评估
        val_start_time = time.time()
        val_acc = validate_expert(model, expert_idx, val_loader, device, use_amp)
        val_time = time.time() - val_start_time
        val_times.append(val_time)
        val_accs.append(val_acc)

        # 立即输出验证结果
        log_message(
            f"Expert {expert_idx} - Epoch {epoch + 1}/{num_epochs} - 验证完成 - "
            f"Acc: {val_acc:.2f}% - Time: {val_time:.1f}s"
        )

        # 在测试集上评估
        test_start_time = time.time()
        test_acc = validate_expert(model, expert_idx, test_loader, device, use_amp)
        test_time = time.time() - test_start_time
        test_times.append(test_time)
        test_accs.append(test_acc)

        # 立即输出测试结果
        log_message(
            f"Expert {expert_idx} - Epoch {epoch + 1}/{num_epochs} - 测试完成 - "
            f"Acc: {test_acc:.2f}% - Time: {test_time:.1f}s"
        )

        # 输出总结
        log_message(
            f"Expert {expert_idx} - Epoch {epoch + 1}/{num_epochs} - 总结 - "
            f"Train: {train_acc:.2f}%, Val: {val_acc:.2f}%, Test: {test_acc:.2f}%"
        )

        # 早停检查
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            # 保存最佳模型
            torch.save(model.state_dict(), f"{RESULTS_PATH}/param/expert_{expert_idx}_best.pth")
            log_message(f"保存专家 {expert_idx} 的新最佳模型，验证准确率: {val_acc:.2f}%")
        else:
            epochs_no_improve += 1
            log_message(f"专家 {expert_idx} 验证准确率未提升 {epochs_no_improve}/{patience} epochs")

        # 保存训练状态，以便下次继续训练
        save_training_state(model, optimizer, scheduler, epoch, train_losses, val_accs, test_accs,
                            epoch_times, val_times, test_times, best_val_acc, epochs_no_improve, 'expert', expert_idx)

        if epochs_no_improve == patience:
            log_message(f"专家 {expert_idx} 早停！验证准确率 {patience} epochs 未提升")
            break

    # 可视化训练过程
    plt.figure(figsize=(15, 10))
    # 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(train_losses)
    plt.title(f'Expert {expert_idx} Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # 精度曲线
    plt.subplot(2, 2, 2)
    plt.plot(val_accs, label='Validation')
    plt.plot(test_accs, label='Test')
    plt.title(f'Expert {expert_idx} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # 耗时曲线
    plt.subplot(2, 2, 3)
    plt.plot(epoch_times, label='Train')
    plt.plot(val_times, label='Validation')
    plt.plot(test_times, label='Test')
    plt.title('Time per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{RESULTS_PATH}/expert_{expert_idx}_training.png")

    return best_val_acc


def validate_expert(model, expert_idx, data_loader, device, use_amp=USE_AMP):
    """验证或测试专家网络性能"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            if use_amp:
                with autocast():
                    if model.shared_backbone:
                        features = model.backbone(inputs)
                        outputs = model.expert_heads[expert_idx](features)
                    else:
                        outputs = model.experts[expert_idx](inputs)
            else:
                if model.shared_backbone:
                    features = model.backbone(inputs)
                    outputs = model.expert_heads[expert_idx](features)
                else:
                    outputs = model.experts[expert_idx](inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    return accuracy


def finetune_moe(model, optimizer, criterion, full_train_loader, val_loader, test_loader, device,
                 num_epochs=FINETUNE_PARAMS["num_epochs"], use_amp=USE_AMP,
                 alpha=FINETUNE_PARAMS["alpha"], beta=FINETUNE_PARAMS["beta"]):
    """
    端到端微调混合专家模型
    
    Args:
        model: MixtureOfExperts模型
        optimizer: 优化器
        criterion: 损失函数
        full_train_loader: 完整训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
        device: 计算设备
        num_epochs: 训练轮数
        use_amp: 是否使用混合精度训练
        alpha: 粗分类器损失权重
        beta: 专家网络损失权重
    """
    # 加载之前的训练状态（如果有）
    start_epoch, train_losses, val_accs, test_accs, epoch_times, val_times, test_times, best_val_acc, epochs_no_improve = load_training_state(
        model, optimizer, None, 'finetune')

    # 混合精度训练
    scaler = GradScaler() if use_amp else None

    log_message("开始端到端微调混合专家模型...")

    # 如果从头开始训练，则输出最佳val_acc初始值
    if start_epoch == 0:
        log_message(f"初始最佳验证准确率: {best_val_acc:.2f}%")

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        # 训练循环
        for inputs, targets in tqdm(full_train_loader, desc=f"Finetune Epoch {epoch + 1}/{num_epochs} 训练"):
            inputs, targets = inputs.to(device), targets.to(device)

            # 获取粗分类器的目标标签（专家索引）
            expert_targets = model.get_expert_target(targets)

            optimizer.zero_grad()

            if use_amp:
                with autocast():
                    # 完整的前向传播
                    coarse_output, expert_id, local_preds, global_preds = model(inputs)

                    # 粗分类器损失
                    coarse_loss = criterion(coarse_output, expert_targets)

                    # 专家网络损失（仅考虑选定的专家）
                    batch_size = inputs.size(0)
                    expert_loss = torch.tensor(0.0).to(device)

                    for i in range(batch_size):
                        e_id = expert_id[i].item()
                        local_target = targets[i].item() - CLASS_RANGES[e_id][0]

                        if model.shared_backbone:
                            features = model.backbone(inputs[i:i + 1])
                            expert_output = model.expert_heads[e_id](features)
                        else:
                            expert_output = model.experts[e_id](inputs[i:i + 1])

                        expert_loss += criterion(expert_output, torch.tensor([local_target]).to(device))

                    expert_loss = expert_loss / batch_size

                    # 总损失
                    loss = alpha * coarse_loss + beta * expert_loss

                # 混合精度训练的反向传播
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # 完整的前向传播
                coarse_output, expert_id, local_preds, global_preds = model(inputs)

                # 粗分类器损失
                coarse_loss = criterion(coarse_output, expert_targets)

                # 专家网络损失（仅考虑选定的专家）
                batch_size = inputs.size(0)
                expert_loss = torch.tensor(0.0).to(device)

                for i in range(batch_size):
                    e_id = expert_id[i].item()
                    local_target = targets[i].item() - CLASS_RANGES[e_id][0]

                    if model.shared_backbone:
                        features = model.backbone(inputs[i:i + 1])
                        expert_output = model.expert_heads[e_id](features)
                    else:
                        expert_output = model.experts[e_id](inputs[i:i + 1])

                    expert_loss += criterion(expert_output, torch.tensor([local_target]).to(device))

                expert_loss = expert_loss / batch_size

                # 总损失
                loss = alpha * coarse_loss + beta * expert_loss
                loss.backward()
                optimizer.step()

            # 记录精度和损失
            epoch_loss += loss.item() * inputs.size(0)
            total += targets.size(0)
            correct += global_preds.eq(targets.float()).sum().item()

        # 计算训练损失和准确率
        train_loss = epoch_loss / len(full_train_loader.dataset)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)

        epoch_train_time = time.time() - epoch_start_time
        epoch_times.append(epoch_train_time)

        # 立即输出训练结果
        log_message(
            f"Finetune Epoch {epoch + 1}/{num_epochs} - 训练完成 - "
            f"Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% - "
            f"Time: {epoch_train_time:.1f}s"
        )

        # 在验证集上评估
        val_start_time = time.time()
        val_acc = evaluate_moe(model, val_loader, device, use_amp)
        val_time = time.time() - val_start_time
        val_times.append(val_time)
        val_accs.append(val_acc)

        # 立即输出验证结果
        log_message(
            f"Finetune Epoch {epoch + 1}/{num_epochs} - 验证完成 - "
            f"Acc: {val_acc:.2f}% - Time: {val_time:.1f}s"
        )

        # 在测试集上评估
        test_start_time = time.time()
        test_acc = evaluate_moe(model, test_loader, device, use_amp)
        test_time = time.time() - test_start_time
        test_times.append(test_time)
        test_accs.append(test_acc)

        # 立即输出测试结果
        log_message(
            f"Finetune Epoch {epoch + 1}/{num_epochs} - 测试完成 - "
            f"Acc: {test_acc:.2f}% - Time: {test_time:.1f}s"
        )

        # 输出总结
        log_message(
            f"Finetune Epoch {epoch + 1}/{num_epochs} - 总结 - "
            f"Train: {train_acc:.2f}%, Val: {val_acc:.2f}%, Test: {test_acc:.2f}%"
        )

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{RESULTS_PATH}/param/moe_full_best.pth")
            log_message(f"保存微调后的最佳模型，验证准确率: {val_acc:.2f}%")

        # 保存训练状态，以便下次继续训练
        save_training_state(model, optimizer, None, epoch, train_losses, val_accs, test_accs,
                            epoch_times, val_times, test_times, best_val_acc, epochs_no_improve, 'finetune')

    # 可视化训练过程
    plt.figure(figsize=(15, 10))
    # 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(train_losses)
    plt.title('End-to-end Finetuning Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # 精度曲线
    plt.subplot(2, 2, 2)
    plt.plot(val_accs, label='Validation')
    plt.plot(test_accs, label='Test')
    plt.title('End-to-end Finetuning Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # 耗时曲线
    plt.subplot(2, 2, 3)
    plt.plot(epoch_times, label='Train')
    plt.plot(val_times, label='Validation')
    plt.plot(test_times, label='Test')
    plt.title('Time per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{RESULTS_PATH}/finetuning.png")

    return best_val_acc


def evaluate_moe(model, data_loader, device, use_amp=USE_AMP):
    """评估完整的混合专家模型"""
    model.eval()
    correct = 0
    total = 0

    start_time = time.time()

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            if use_amp:
                with autocast():
                    _, _, _, global_preds = model(inputs)
            else:
                _, _, _, global_preds = model(inputs)

            total += targets.size(0)
            correct += global_preds.eq(targets.float()).sum().item()

    eval_time = time.time() - start_time
    accuracy = 100. * correct / total

    return accuracy


def test_moe(model, test_loader, device, use_amp=USE_AMP):
    """测试混合专家模型"""
    start_time = time.time()
    model.eval()

    all_targets = []
    all_predictions = []
    expert_counts = [0] * len(CLASS_RANGES)

    total_samples = 0
    correct_samples = 0

    log_message("开始最终测试评估...")

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader, desc="测试进行中")):
            inputs, targets = inputs.to(device), targets.to(device)

            if use_amp:
                with autocast():
                    coarse_output, expert_id, _, global_preds = model(inputs)
            else:
                coarse_output, expert_id, _, global_preds = model(inputs)

            # 统计每个专家被选择的次数
            for e_id in expert_id:
                expert_counts[e_id.item()] += 1

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(global_preds.cpu().numpy())

            # 即时统计准确率
            batch_samples = targets.size(0)
            batch_correct = global_preds.eq(targets.float()).sum().item()
            total_samples += batch_samples
            correct_samples += batch_correct

            # 每处理10个批次输出一次当前进度的准确率
            if (batch_idx + 1) % 10 == 0:
                current_acc = 100.0 * correct_samples / total_samples
                log_message(f"测试进度: {batch_idx + 1}/{len(test_loader)} 批次 - 当前准确率: {current_acc:.2f}%")

    # 计算评估指标
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)

    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='macro')
    recall = recall_score(all_targets, all_predictions, average='macro')
    f1 = f1_score(all_targets, all_predictions, average='macro')

    test_time = time.time() - start_time

    # 输出评估结果
    log_message("\n----- 测试结果 -----")
    log_message(f"准确率: {accuracy * 100:.2f}% (即时计算: {100.0 * correct_samples / total_samples:.2f}%)")
    log_message(f"精确率 (Macro): {precision * 100:.2f}%")
    log_message(f"召回率 (Macro): {recall * 100:.2f}%")
    log_message(f"F1分数 (Macro): {f1 * 100:.2f}%")
    log_message(f"测试耗时: {test_time:.2f}秒 (平均每样本 {test_time / total_samples * 1000:.2f}毫秒)")

    # 输出专家分配情况
    total_samples = sum(expert_counts)
    log_message("\n----- 专家分配统计 -----")
    for i, count in enumerate(expert_counts):
        percentage = count / total_samples * 100
        log_message(f"专家 {i}: {count} 样本 ({percentage:.2f}%)")

    # 绘制混淆矩阵（可选，对于大量类别可能不直观）
    if NUM_CLASSES <= 30:  # 仅当类别数量较少时绘制混淆矩阵
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(all_targets, all_predictions)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f"{RESULTS_PATH}/confusion_matrix.png")

    # 统计专家分配的平衡度
    expected_per_expert = total_samples / len(CLASS_RANGES)
    imbalance_scores = [abs(count - expected_per_expert) / expected_per_expert for count in expert_counts]
    avg_imbalance = sum(imbalance_scores) / len(imbalance_scores)
    log_message(f"专家负载不平衡度: {avg_imbalance:.4f} (0为完全平衡)")

    return accuracy, precision, recall, f1


def main():
    # 设置随机种子以便结果可重复
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)

    # 检查CUDA是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_message(f"使用设备: {device}")

    # 记录当前设置的超参数
    log_message("\n----- 超参数配置 -----")
    log_message(f"MODEL_PARAMS: {MODEL_PARAMS}")
    log_message(f"COARSE_CLASSIFIER_PARAMS: {COARSE_CLASSIFIER_PARAMS}")
    log_message(f"EXPERT_PARAMS: {EXPERT_PARAMS}")
    log_message(f"FINETUNE_PARAMS: {FINETUNE_PARAMS}")

    # 获取数据加载器
    log_message("\n正在加载数据...")
    train_loaders, val_loaders, test_loaders, full_train_loader, val_loader, test_loader = get_dataloaders(CLASS_RANGES)
    log_message("数据加载完成")

    # 检查数据形状
    for x, y in full_train_loader:
        log_message(f"输入数据形状: {x.shape}")
        input_channels = x.shape[1]
        input_height = x.shape[2] if len(x.shape) > 3 else 1
        input_width = x.shape[3] if len(x.shape) > 3 else x.shape[2]
        break

    # 创建混合专家模型
    log_message("\n创建混合专家模型...")
    model = MixtureOfExperts(
        class_ranges=CLASS_RANGES,
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        dropout_rate=MODEL_PARAMS["dropout_rate"],
        shared_backbone=MODEL_PARAMS["shared_backbone"]
    )
    model = model.to(device)

    # 统计模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_message(f"模型总参数量: {total_params:,}")
    log_message(f"模型可训练参数量: {trainable_params:,}")

    # 阶段1: 训练粗分类器
    coarse_classifier_path = f"{RESULTS_PATH}/param/coarse_classifier_best.pth"
    coarse_checkpoint_path = f"{RESULTS_PATH}/param/coarse_classifier_checkpoint.pth"

    if check_model_exists(coarse_classifier_path) and not check_model_exists(coarse_checkpoint_path):
        # 如果已经有了最佳模型但没有检查点，说明训练已经完成了
        log_message("\n===== 阶段1: 加载已训练的粗分类器 =====")
        model.load_state_dict(torch.load(coarse_classifier_path), strict=False)
        log_message(f"已加载粗分类器模型: {coarse_classifier_path}")

        # 在验证集和测试集上评估粗分类器性能
        val_acc = validate_coarse_classifier(model, val_loader, device, USE_AMP)
        test_acc = validate_coarse_classifier(model, test_loader, device, USE_AMP)
        log_message(f"加载的粗分类器 - 验证集准确率: {val_acc:.2f}%, 测试集准确率: {test_acc:.2f}%")
    else:
        log_message("\n===== 阶段1: 训练粗分类器 =====")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=COARSE_CLASSIFIER_PARAMS["learning_rate"],
            weight_decay=COARSE_CLASSIFIER_PARAMS["weight_decay"]
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=COARSE_CLASSIFIER_PARAMS["scheduler_step_size"],
            gamma=COARSE_CLASSIFIER_PARAMS["scheduler_gamma"]
        )

        # 如果使用共享骨干网络，需要冻结专家头部分
        if model.shared_backbone:
            for expert_head in model.expert_heads:
                for param in expert_head.parameters():
                    param.requires_grad = False

            log_message("使用共享骨干网络：已冻结所有专家头")
        else:
            for expert in model.experts:
                for param in expert.parameters():
                    param.requires_grad = False

            log_message("使用独立模型：已冻结所有专家网络")

        coarse_val_acc = train_coarse_classifier(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            full_train_loader=full_train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            num_epochs=COARSE_CLASSIFIER_PARAMS["num_epochs"],
            patience=COARSE_CLASSIFIER_PARAMS["patience"],
            scheduler=scheduler,
            use_amp=USE_AMP
        )

        log_message(f"\n粗分类器训练完成，最佳验证准确率: {coarse_val_acc:.2f}%")

    # 阶段2: 训练专家网络
    # 首先确保加载了最佳的粗分类器模型
    if not check_model_exists(coarse_classifier_path):
        log_message("错误：粗分类器模型不存在，无法继续训练专家网络")
        return

    log_message("\n===== 阶段2: 训练专家网络 =====")
    model.load_state_dict(torch.load(coarse_classifier_path), strict=False)
    log_message("加载粗分类器最佳模型...")

    # 如果使用共享骨干网络，需要冻结骨干网络和粗分类器
    if model.shared_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.coarse_head.parameters():
            param.requires_grad = False

        log_message("共享骨干网络：已冻结骨干网络和粗分类器头")
    else:
        for param in model.coarse_classifier.parameters():
            param.requires_grad = False

        log_message("独立模型：已冻结粗分类器")

    # 训练每个专家
    criterion = nn.CrossEntropyLoss()
    expert_val_accs = []

    for expert_idx in range(len(CLASS_RANGES)):
        expert_model_path = f"{RESULTS_PATH}/param/expert_{expert_idx}_best.pth"
        expert_checkpoint_path = f"{RESULTS_PATH}/param/expert_{expert_idx}_checkpoint.pth"

        if check_model_exists(expert_model_path) and not check_model_exists(expert_checkpoint_path):
            # 如果已经有了最佳模型但没有检查点，说明训练已经完成了
            log_message(f"\n----- 加载已训练的专家 {expert_idx} -----")
            model.load_state_dict(torch.load(expert_model_path), strict=False)
            log_message(f"已加载专家 {expert_idx} 模型: {expert_model_path}")

            # 评估已加载专家模型的性能
            val_acc = validate_expert(model, expert_idx, val_loaders[expert_idx], device, USE_AMP)
            test_acc = validate_expert(model, expert_idx, test_loaders[expert_idx], device, USE_AMP)
            log_message(f"专家 {expert_idx} - 验证集准确率: {val_acc:.2f}%, 测试集准确率: {test_acc:.2f}%")
            expert_val_accs.append(val_acc)
        else:
            log_message(f"\n----- 训练专家 {expert_idx} -----")

            # 解冻当前专家网络
            if model.shared_backbone:
                for param in model.expert_heads[expert_idx].parameters():
                    param.requires_grad = True

                log_message(f"已解冻专家 {expert_idx} 的分类头")

                optimizer = optim.Adam(
                    model.expert_heads[expert_idx].parameters(),
                    lr=EXPERT_PARAMS["learning_rate"],
                    weight_decay=EXPERT_PARAMS["weight_decay"]
                )
            else:
                for param in model.experts[expert_idx].parameters():
                    param.requires_grad = True

                log_message(f"已解冻专家 {expert_idx} 的完整网络")

                optimizer = optim.Adam(
                    model.experts[expert_idx].parameters(),
                    lr=EXPERT_PARAMS["learning_rate"],
                    weight_decay=EXPERT_PARAMS["weight_decay"]
                )

            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=EXPERT_PARAMS["scheduler_step_size"],
                gamma=EXPERT_PARAMS["scheduler_gamma"]
            )

            log_message(
                f"开始训练专家 {expert_idx}，类别范围: {CLASS_RANGES[expert_idx]}，类别数: {model.expert_num_classes[expert_idx]}")

            expert_val_acc = train_expert(
                model=model,
                expert_idx=expert_idx,
                optimizer=optimizer,
                criterion=criterion,
                train_loader=train_loaders[expert_idx],
                val_loader=val_loaders[expert_idx],
                test_loader=test_loaders[expert_idx],
                device=device,
                num_epochs=EXPERT_PARAMS["num_epochs"],
                patience=EXPERT_PARAMS["patience"],
                scheduler=scheduler,
                use_amp=USE_AMP
            )

            expert_val_accs.append(expert_val_acc)

            # 再次冻结当前专家网络
            if model.shared_backbone:
                for param in model.expert_heads[expert_idx].parameters():
                    param.requires_grad = False
            else:
                for param in model.experts[expert_idx].parameters():
                    param.requires_grad = False

            log_message(f"专家 {expert_idx} 训练完成，最佳验证准确率: {expert_val_acc:.2f}%")

    log_message("\n所有专家网络训练完成")
    for i, acc in enumerate(expert_val_accs):
        log_message(f"专家 {i} 最佳验证准确率: {acc:.2f}%")

    # 阶段3: 端到端微调
    moe_full_path = f"{RESULTS_PATH}/param/moe_full_best.pth"
    moe_checkpoint_path = f"{RESULTS_PATH}/param/moe_full_checkpoint.pth"

    if check_model_exists(moe_full_path) and not check_model_exists(moe_checkpoint_path):
        # 如果已经有了最佳模型但没有检查点，说明训练已经完成了
        log_message("\n===== 阶段3: 加载已微调的完整模型 =====")
        model.load_state_dict(torch.load(moe_full_path))
        log_message(f"已加载微调后的完整模型: {moe_full_path}")

        # 评估已加载微调模型的性能
        val_acc = evaluate_moe(model, val_loader, device, USE_AMP)
        test_acc = evaluate_moe(model, test_loader, device, USE_AMP)
        log_message(f"微调后模型 - 验证集准确率: {val_acc:.2f}%, 测试集准确率: {test_acc:.2f}%")
    else:
        log_message("\n===== 阶段3: 端到端微调 =====")

        # 加载之前训练的各组件最佳模型，但前提是没有现有的检查点
        if not check_model_exists(moe_checkpoint_path):
            log_message("加载先前训练的最佳模型参数进行微调")
            model.load_state_dict(torch.load(coarse_classifier_path), strict=False)

            # 加载所有已训练的专家模型参数
            for expert_idx in range(len(CLASS_RANGES)):
                expert_path = f"{RESULTS_PATH}/param/expert_{expert_idx}_best.pth"
                if check_model_exists(expert_path):
                    expert_state = torch.load(expert_path)
                    model.load_state_dict(expert_state, strict=False)
                    log_message(f"已加载专家 {expert_idx} 模型参数")
                else:
                    log_message(f"警告：专家 {expert_idx} 模型参数不存在")

        # 解冻所有参数
        for param in model.parameters():
            param.requires_grad = True

        log_message("已解冻所有参数进行端到端微调")

        # 使用较小的学习率进行微调
        optimizer = optim.Adam(
            model.parameters(),
            lr=FINETUNE_PARAMS["learning_rate"],
            weight_decay=FINETUNE_PARAMS["weight_decay"]
        )

        log_message(
            f"开始端到端微调，学习率: {FINETUNE_PARAMS['learning_rate']}, 损失权重: α={FINETUNE_PARAMS['alpha']}, β={FINETUNE_PARAMS['beta']}")

        finetune_val_acc = finetune_moe(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            full_train_loader=full_train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            num_epochs=FINETUNE_PARAMS["num_epochs"],
            use_amp=USE_AMP,
            alpha=FINETUNE_PARAMS["alpha"],
            beta=FINETUNE_PARAMS["beta"]
        )

        log_message(f"\n端到端微调完成，最佳验证准确率: {finetune_val_acc:.2f}%")

    # 最终测试
    log_message("\n===== 最终测试 =====")
    # 确保加载最佳的微调模型
    if check_model_exists(moe_full_path):
        model.load_state_dict(torch.load(moe_full_path))
        log_message("加载最佳微调模型进行最终测试...")
        test_accuracy, test_precision, test_recall, test_f1 = test_moe(
            model=model,
            test_loader=test_loader,
            device=device,
            use_amp=USE_AMP
        )
    else:
        log_message("警告：微调后的模型参数不存在，使用当前模型进行测试")
        test_accuracy, test_precision, test_recall, test_f1 = test_moe(
            model=model,
            test_loader=test_loader,
            device=device,
            use_amp=USE_AMP
        )

    # 清理检查点文件（可选）
    for path in [f"{RESULTS_PATH}/param/coarse_classifier_checkpoint.pth",
                 *[f"{RESULTS_PATH}/param/expert_{i}_checkpoint.pth" for i in range(len(CLASS_RANGES))],
                 f"{RESULTS_PATH}/param/moe_full_checkpoint.pth"]:
        if os.path.exists(path):
            os.remove(path)
            log_message(f"已删除检查点文件: {path}")

    log_message("\n训练和评估完成！")


if __name__ == "__main__":
    main()
