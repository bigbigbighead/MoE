import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import glob
import gc
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from models.MoE import CoarseExpertModel
from models.ResNet import resnet18

from utils.data_loading_mine import log_message, load_data, get_dataloaders, CLASS_RANGES
from utils.load_model import load_checkpoint
from utils.loss_functions import compute_specialized_loss
from utils.visualization import log_metrics_to_tensorboard, plot_loss_curves, plot_accuracy_curves

# 导入分析模块的函数
from analyse_MoE import (
    analyze_model,
    calculate_per_class_accuracy,
    plot_per_class_accuracy,
    create_confusion_matrix,
    analyze_misclassified,
    ANALYSIS_RESULTS_PATH
)

# 数据集路径
DATASET_PATH = "./data/AppClassNet/top200"
RESULTS_PATH = "./results/AppClassNet/top200/MoE/56"

# 确保结果目录存在
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(f"{RESULTS_PATH}/param", exist_ok=True)
os.makedirs(f"{RESULTS_PATH}/logs", exist_ok=True)
os.makedirs(f"{RESULTS_PATH}/analysis", exist_ok=True)  # 确保分析结果目录存在

# 创建日志文件
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"{RESULTS_PATH}/logs/training_log_{current_time}.txt"

# 优化超参数
BATCH_SIZE = 2048  # 批次大小
EPOCHS_COARSE = 20  # 粗分类器训练轮数
EPOCHS_EXPERT = 20  # 专家训练轮数
LEARNING_RATE_COARSE = 0.001  # 粗分类器学习率
LEARNING_RATE_EXPERT = 0.001  # 专家学习率
NUM_CLASSES = 200  # AppClassNet 类别数
NUM_EXPERTS = len(CLASS_RANGES)  # 专家数量
NUM_WORKERS = 2  # 数据加载的worker数量
PIN_MEMORY = True  # 确保启用pin_memory
PREFETCH_FACTOR = 4  # 增加预取因子
# 自动混合精度训练配置
USE_AMP = True  # 启动自动混合精度训练


# 训练粗分类器
def train_coarse_classifier(model, train_loader, val_loader, test_loader, device, resume_training=False):
    """训练粗分类器，将样本分配给正确的专家"""
    log_message("开始训练粗分类器...")
    writer = SummaryWriter(os.path.join(RESULTS_PATH, 'logs', 'coarse_classifier'))

    # 设置模型为粗分类器训练模式
    model.set_training_mode(train_coarse=True)
    model.train()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE_COARSE
    )
    criterion = nn.CrossEntropyLoss()

    # 创建混合精度训练的scaler
    scaler = GradScaler() if USE_AMP else None

    # 恢复训练
    start_epoch = 0
    best_val_acc = 0.0

    if resume_training:
        checkpoint_path = os.path.join(RESULTS_PATH, 'param', 'coarse_classifier_latest.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint['best_val_acc']
            log_message(f"已加载粗分类器检查点，从第{start_epoch}轮继续训练")

    # 记录训练历史
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # 开始训练
    for epoch in range(start_epoch, EPOCHS_COARSE):
        epoch_start_time = time.time()
        log_message(f"粗分类器训练 - Epoch {epoch + 1}/{EPOCHS_COARSE}")

        # 训练一个epoch
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        # 预计算expert_targets映射，避免在循环中重复计算
        class_to_expert_map = torch.tensor([model.class_to_expert[i] for i in range(NUM_CLASSES)],
                                           device=device)

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            # 将类别标签转换为专家索引 - 优化版本
            expert_targets = class_to_expert_map[targets]

            optimizer.zero_grad(set_to_none=True)

            # 使用混合精度训练
            if USE_AMP and scaler is not None:
                with autocast():
                    # 前向传播 - 只获取粗分类器输出
                    coarse_logits, _ = model(inputs)
                    loss = criterion(coarse_logits, expert_targets)

                # 使用scaler进行反向传播和优化
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # 前向传播
                coarse_logits, _ = model(inputs)
                loss = criterion(coarse_logits, expert_targets)

                # 反向传播与优化
                loss.backward()
                optimizer.step()

            # 统计
            epoch_loss += loss.item()
            _, predicted = coarse_logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(expert_targets).sum().item()

        # 计算训练集平均损失和准确率
        avg_loss = epoch_loss / len(train_loader)
        accuracy = correct / total
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        epoch_time_taken = time.time() - epoch_start_time

        # 记录TensorBoard指标
        writer.add_scalar('train/loss', avg_loss, epoch)
        writer.add_scalar('train/accuracy', accuracy, epoch)

        log_message(f"  粗分类器训练 - 损失: {avg_loss:.4f}, 准确率: {accuracy:.4f}, 耗时: {epoch_time_taken:.2f}s")

        # 验证粗分类器
        val_start_time = time.time()
        val_loss, val_accuracy = validate_coarse_classifier(model, val_loader, criterion, device)
        val_time_taken = time.time() - val_start_time
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # 记录验证集指标
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/accuracy', val_accuracy, epoch)
        writer.add_scalar('val/time_taken', val_time_taken, epoch)

        log_message(f"  粗分类器验证 - 损失: {val_loss:.4f}, 准确率: {val_accuracy:.4f}, 耗时: {val_time_taken:.2f}s")

        # 测试粗分类器
        test_start_time = time.time()
        test_loss, test_accuracy = validate_coarse_classifier(model, test_loader, criterion, device)
        test_time_taken = time.time() - test_start_time

        # 记录测试集指标
        writer.add_scalar('test/loss', test_loss, epoch)
        writer.add_scalar('test/accuracy', test_accuracy, epoch)
        writer.add_scalar('test/time_taken', test_time_taken, epoch)

        log_message(
            f"  粗分类器测试 - 损失: {test_loss:.4f}, 准确率: {test_accuracy:.4f}, 耗时: {test_time_taken:.2f}s")

        # 保存最新检查点
        checkpoint_path = os.path.join(RESULTS_PATH, 'param', 'coarse_classifier_latest.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'best_val_acc': max(best_val_acc, val_accuracy)
        }, checkpoint_path)

        # 如果是最佳性能，保存最佳检查点
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_checkpoint_path = os.path.join(RESULTS_PATH, 'param', 'coarse_classifier_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'best_val_acc': best_val_acc
            }, best_checkpoint_path)
            log_message(f"  已保存粗分类器最佳模型，验证准确率: {best_val_acc:.4f}")

    # 绘制损失和准确率曲线
    loss_curve_path = os.path.join(RESULTS_PATH, 'logs', 'coarse_classifier_loss_curve.png')
    acc_curve_path = os.path.join(RESULTS_PATH, 'logs', 'coarse_classifier_accuracy_curve.png')
    plot_loss_curves(train_losses, val_losses, loss_curve_path)
    plot_accuracy_curves(train_accuracies, val_accuracies, acc_curve_path)
    log_message("已保存粗分类器的损失和准确率曲线")

    writer.close()
    return model


def validate_coarse_classifier(model, val_loader, criterion, device):
    """验证粗分类器的性能"""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    # 预计算expert_targets映射，避免在循环中重复计算
    class_to_expert_map = torch.tensor([model.class_to_expert[i] for i in range(NUM_CLASSES)],
                                       device=device)

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # 将类别标签转换为专家索引 - 优化版本
            expert_targets = class_to_expert_map[targets]

            # 前向传播
            coarse_logits, _ = model(inputs)

            # 计算损失
            loss = criterion(coarse_logits, expert_targets)
            val_loss += loss.item()

            # 计算准确率
            _, predicted = coarse_logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(expert_targets).sum().item()

    # 计算平均损失和准确率
    avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
    accuracy = correct / total if total > 0 else 0

    return avg_val_loss, accuracy


# 训练专家
def train_experts(model, expert_train_loaders, expert_val_loaders, expert_test_loaders, device, resume_training=False):
    """训练每个专家"""
    log_message("开始训练专家...")

    # 设置模型为专家训练模式
    model.set_training_mode(train_coarse=False)
    model.train()

    # 创建混合精度训练的scaler
    scaler = GradScaler() if USE_AMP else None

    # 外层循环：每个专家
    for expert_idx in range(len(model.experts)):
        log_message(f"===== 开始训练专家 {expert_idx + 1}/{len(model.experts)} =====")
        start_class, end_class = model.class_ranges[expert_idx]
        log_message(f"专家{expert_idx}负责类别范围: [{start_class}, {end_class}]")

        # 创建专家的TensorBoard写入器
        writer = SummaryWriter(os.path.join(RESULTS_PATH, 'logs', f'expert{expert_idx}'))

        # 只优化当前专家的参数
        optimizer = optim.Adam(model.experts[expert_idx].parameters(), lr=LEARNING_RATE_EXPERT)
        criterion = nn.CrossEntropyLoss()

        # 恢复训练
        start_epoch = 0
        best_val_acc = 0.0

        if resume_training:
            expert_checkpoint_path = os.path.join(RESULTS_PATH, 'param', f'expert{expert_idx}_latest.pth')
            if os.path.exists(expert_checkpoint_path):
                checkpoint = torch.load(expert_checkpoint_path)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    # 只加载当前专家的参数
                    model_dict = model.state_dict()
                    expert_dict = {k: v for k, v in checkpoint['model_state_dict'].items()
                                   if k.startswith(f'experts.{expert_idx}') or k.startswith(
                            f'module.experts.{expert_idx}')}
                    model_dict.update(expert_dict)
                    model.load_state_dict(model_dict)

                    if 'optimizer_state_dict' in checkpoint:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                    if 'epoch' in checkpoint:
                        start_epoch = checkpoint['epoch'] + 1

                    if 'best_val_acc' in checkpoint:
                        best_val_acc = checkpoint['best_val_acc']

                    log_message(f"已加载专家{expert_idx}的检查点，从第{start_epoch}轮继续训练")

        # 获取当前专家的数据加载器
        train_loader = expert_train_loaders[expert_idx]
        val_loader = expert_val_loaders[expert_idx]
        test_loader = expert_test_loaders[expert_idx]

        # 记录训练历史
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        val_times = []  # 记录验证时间
        test_times = []  # 记录测试时间

        # 内层循环：该专家的完整训练周期
        for epoch in range(start_epoch, EPOCHS_EXPERT):
            epoch_start_time = time.time()
            log_message(f"专家{expert_idx} - Epoch {epoch + 1}/{EPOCHS_EXPERT}")

            # 训练一个epoch
            model.train()
            epoch_loss = 0
            correct = 0
            total = 0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                # 使用混合精度训练
                if USE_AMP and scaler is not None:
                    with autocast():
                        # 直接使用专家进行前向传播
                        outputs = model.experts[expert_idx](inputs)
                        loss = criterion(outputs, targets)

                    # 使用scaler进行反向传播和优化
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # 直接使用专家进行前向传播
                    outputs = model.experts[expert_idx](inputs)
                    loss = criterion(outputs, targets)

                    # 反向传播与优化
                    loss.backward()
                    optimizer.step()

                # 统计
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            # 计算训练集平均损失和准确率
            avg_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0
            accuracy = correct / total if total > 0 else 0
            train_losses.append(avg_loss)
            train_accuracies.append(accuracy)
            epoch_time_taken = time.time() - epoch_start_time

            # 记录TensorBoard指标
            writer.add_scalar('train/loss', avg_loss, epoch)
            writer.add_scalar('train/accuracy', accuracy, epoch)
            writer.add_scalar('train/time_taken', epoch_time_taken, epoch)

            log_message(
                f"  专家{expert_idx} - 训练损失: {avg_loss:.4f}, 准确率: {accuracy:.4f}, 耗时: {epoch_time_taken:.2f}s")

            # 验证专家
            val_start_time = time.time()
            val_loss, val_accuracy = validate_expert(model, val_loader, criterion, expert_idx, device)
            val_time_taken = time.time() - val_start_time
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            val_times.append(val_time_taken)

            # 记录验证集指标
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/accuracy', val_accuracy, epoch)
            writer.add_scalar('val/time_taken', val_time_taken, epoch)

            log_message(
                f"  专家{expert_idx} - 验证损失: {val_loss:.4f}, 准确率: {val_accuracy:.4f}, 耗时: {val_time_taken:.2f}s")

            # 测试专家
            test_start_time = time.time()
            test_loss, test_accuracy = validate_expert(model, test_loader, criterion, expert_idx, device)
            test_time_taken = time.time() - test_start_time
            test_times.append(test_time_taken)

            # 记录测试集指标
            writer.add_scalar('test/loss', test_loss, epoch)
            writer.add_scalar('test/accuracy', test_accuracy, epoch)
            writer.add_scalar('test/time_taken', test_time_taken, epoch)

            log_message(
                f"  专家{expert_idx} - 测试损失: {test_loss:.4f}, 准确率: {test_accuracy:.4f}, 耗时: {test_time_taken:.2f}s")

            # 保存最新检查点
            checkpoint_path = os.path.join(RESULTS_PATH, 'param', f'expert{expert_idx}_latest.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'best_val_acc': max(best_val_acc, val_accuracy)
            }, checkpoint_path)

            # 如果是最佳性能，保存最佳检查点
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_checkpoint_path = os.path.join(RESULTS_PATH, 'param', f'expert{expert_idx}_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'best_val_acc': best_val_acc
                }, best_checkpoint_path)
                log_message(f"  已保存专家{expert_idx}的最佳模型，验证准确率: {best_val_acc:.4f}")

        # 绘制损失和准确率曲线
        loss_curve_path = os.path.join(RESULTS_PATH, 'logs', f'expert{expert_idx}_loss_curve.png')
        acc_curve_path = os.path.join(RESULTS_PATH, 'logs', f'expert{expert_idx}_accuracy_curve.png')
        plot_loss_curves(train_losses, val_losses, loss_curve_path)
        plot_accuracy_curves(train_accuracies, val_accuracies, acc_curve_path)
        log_message(f"  已保存专家{expert_idx}的损失和准确率曲线")

        # 保存验证和测试时间统计
        time_data = {
            'epoch': list(range(1, len(val_times) + 1)),
            'validation_time': val_times,
            'test_time': test_times
        }
        pd.DataFrame(time_data).to_csv(os.path.join(RESULTS_PATH, 'logs', f'expert{expert_idx}_time_stats.csv'),
                                       index=False)
        log_message(f"  已保存专家{expert_idx}的验证和测试时间统计")

        writer.close()

    # 所有专家训练完成后，保存完整模型
    final_checkpoint_path = os.path.join(RESULTS_PATH, 'param', 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_ranges': model.class_ranges,
        'total_classes': model.total_classes
    }, final_checkpoint_path)
    log_message(f"所有专家训练完成，完整模型已保存到 {final_checkpoint_path}")

    return model


def validate_expert(model, val_loader, criterion, expert_idx, device):
    """验证单个专家的性能"""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    # 使用torch.no_grad()避免计算梯度，加快推理速度
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # 直接使用专家进行前向传播
            outputs = model.experts[expert_idx](inputs)

            # 计算损失
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            # 计算准确率
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # 计算平均损失和准确率
    avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
    accuracy = correct / total if total > 0 else 0

    return avg_val_loss, accuracy


def evaluate_model(model, test_loader, device):
    """评估整体模型性能"""
    log_message("评估整体模型性能...")
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    total = 0

    # 记录评估开始时间
    eval_start_time = time.time()

    # 使用torch.no_grad()避免计算梯度，加快推理速度
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # 使用模型的推理函数获取最终输出
            final_logits = model.inference(inputs)

            # 计算损失
            loss = criterion(final_logits, targets)
            test_loss += loss.item()

            # 计算准确率
            _, predicted = final_logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # 计算评估总耗时
    eval_time = time.time() - eval_start_time

    # 计算平均损失和准确率
    avg_test_loss = test_loss / len(test_loader) if len(test_loader) > 0 else 0
    test_accuracy = correct / total if total > 0 else 0

    log_message(f"测试损失: {avg_test_loss:.4f}, 准确率: {test_accuracy:.4f}, 总耗时: {eval_time:.2f}s")

    return avg_test_loss, test_accuracy


def analyze_model_performance(model, test_loader, device):
    """分析模型性能，生成混淆矩阵和每个类别的准确率"""
    log_message("分析模型性能...")
    model.eval()

    all_predictions = []
    all_targets = []

    # 记录分析开始时间
    analysis_start_time = time.time()

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # 使用模型的推理函数获取最终输出
            final_logits = model.inference(inputs)

            # 获取预测结果
            _, predicted = final_logits.max(1)

            all_predictions.append(predicted.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # 合并所有批次的结果
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    # 计算每个类别的准确率
    per_class_accuracy, per_class_correct, per_class_total = calculate_per_class_accuracy(
        all_predictions, all_targets, NUM_CLASSES)

    # 保存每个类别的准确率数据
    accuracy_data = {
        'class': np.arange(NUM_CLASSES),
        'accuracy': per_class_accuracy,
        'correct': per_class_correct,
        'total': per_class_total
    }

    pd.DataFrame(accuracy_data).to_csv(f"{RESULTS_PATH}/analysis/per_class_accuracy.csv", index=False)

    # 绘制每个类别的准确率柱状图
    plot_per_class_accuracy(per_class_accuracy, per_class_total, CLASS_RANGES, )

    # 创建混淆矩阵
    cm, cm_normalized = create_confusion_matrix(
        all_predictions, all_targets, NUM_CLASSES, CLASS_RANGES, )

    # 计算总体准确率
    overall_accuracy = np.sum(per_class_correct) / np.sum(per_class_total)
    log_message(f"整体模型准确率: {overall_accuracy:.4%}")

    # 统计每个专家的性能
    expert_accuracies = []
    for expert_idx, (start_class, end_class) in enumerate(CLASS_RANGES):
        # 过滤该专家负责的类别范围的样本
        expert_mask = (all_targets >= start_class) & (all_targets <= end_class)
        expert_preds = all_predictions[expert_mask]
        expert_targets = all_targets[expert_mask]

        # 计算该专家负责类别范围的准确率
        expert_acc = np.mean(expert_preds == expert_targets) if len(expert_targets) > 0 else 0
        expert_accuracies.append(expert_acc)

        log_message(f"专家{expert_idx}[类别{start_class}-{end_class}]准确率: {expert_acc:.4%}")

    # 统计粗分类器的准确率（将样本分配给正确专家的比例）
    correct_expert_assignments = 0
    for i, (pred, target) in enumerate(zip(all_predictions, all_targets)):
        correct_expert = None
        for expert_idx, (start_class, end_class) in enumerate(CLASS_RANGES):
            if start_class <= target <= end_class:
                correct_expert = expert_idx
                break

        pred_expert = None
        for expert_idx, (start_class, end_class) in enumerate(CLASS_RANGES):
            if start_class <= pred <= end_class:
                pred_expert = expert_idx
                break

        if pred_expert == correct_expert:
            correct_expert_assignments += 1

    expert_assignment_accuracy = correct_expert_assignments / len(all_targets)
    log_message(f"粗分类器准确率(基于预测): {expert_assignment_accuracy:.4%}")

    # 计算分析总耗时
    analysis_time = time.time() - analysis_start_time
    log_message(f"性能分析总耗时: {analysis_time:.2f}s")
    return overall_accuracy, expert_accuracies


if __name__ == "__main__":
    # 设置NUMA绑定和多线程优化
    if torch.cuda.is_available():
        # 启用CUDA性能优化
        torch.backends.cudnn.benchmark = True
        # 确定性计算 - 如果需要精确复现结果，取消下面的注释
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.enabled = True

    # 如果使用CPU，则设置OpenMP线程数
    if not torch.cuda.is_available():
        torch.set_num_threads(NUM_WORKERS)  # 设置适当的线程数

    # 记录训练开始信息和配置信息
    log_message(f"=== 训练开始于 {current_time} ===")
    log_message(f"BatchSize: {BATCH_SIZE}")
    log_message(f"粗分类器学习率: {LEARNING_RATE_COARSE}, 训练轮数: {EPOCHS_COARSE}")
    log_message(f"专家学习率: {LEARNING_RATE_EXPERT}, 训练轮数: {EPOCHS_EXPERT}")
    log_message(f"数据集路径: {DATASET_PATH}")
    log_message(f"结果保存路径: {RESULTS_PATH}")
    log_message(f"使用设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    log_message(f"自动混合精度: {USE_AMP}")
    log_message(f"工作进程数: {NUM_WORKERS}")
    log_message(f"专家数量: {NUM_EXPERTS}")
    log_message(f"类别范围: {CLASS_RANGES}")

    # 记录PyTorch和CUDA版本
    log_message(f"PyTorch版本: {torch.__version__}")
    if torch.cuda.is_available():
        log_message(f"CUDA版本: {torch.version.cuda}")
        log_message(f"GPU型号: {torch.cuda.get_device_name(0)}")

    # 配置数据并行训练，利用多个GPU
    multi_gpu = torch.cuda.device_count() > 1
    log_message(f"使用GPU数量: {torch.cuda.device_count()}")

    # 添加继续训练选项
    RESUME_TRAINING = False  # 设置是否从检查点继续训练

    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    expert_train_loaders, expert_val_loaders, expert_test_loaders, full_train_loader, full_val_loader, full_test_loader = get_dataloaders(
        CLASS_RANGES)

    # 从训练数据中获取输入形状
    sample_input, _ = next(iter(full_train_loader))
    input_channels = sample_input.shape[1]
    input_height = sample_input.shape[2]
    input_width = sample_input.shape[3]

    log_message(f"模型输入形状: 通道={input_channels}, 高度={input_height}, 宽度={input_width}")

    # 创建模型 - 从头开始训练
    model = CoarseExpertModel(
        total_classes=NUM_CLASSES,
        class_ranges=CLASS_RANGES,
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        dropout_rate=0.1
    )

    log_message("已创建模型，将从头开始训练（不加载预训练参数）")

    # 如果有多GPU，使用DataParallel
    if multi_gpu:
        log_message("启用多GPU并行训练")
        model = nn.DataParallel(model)

    model.to(device)

    # 训练粗分类器
    log_message("\n===== 第一阶段：训练粗分类器 =====")
    model = train_coarse_classifier(model, full_train_loader, full_val_loader, full_test_loader, device,
                                    resume_training=RESUME_TRAINING)

    # 清理内存
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 创建专家专用的数据加载器
    log_message("\n===== 创建专家专用数据加载器 =====")
    # 训练专家
    log_message("\n===== 第二阶段：训练专家 =====")
    model = train_experts(model, expert_train_loaders, expert_val_loaders, expert_test_loaders, device,
                          resume_training=RESUME_TRAINING)

    # 清理内存
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 评估整体模型性能
    log_message("\n===== 评估最终模型性能 =====")
    test_loss, test_accuracy = evaluate_model(model, full_test_loader, device)

    # 分析模型性能
    log_message("\n===== 分析模型性能 =====")
    overall_accuracy, expert_accuracies = analyze_model_performance(model, full_test_loader, device)

    # 记录最终结果
    log_message(f"\n===== 训练完成 =====")
    log_message(f"最终测试准确率: {test_accuracy:.4f}")
    for expert_idx, acc in enumerate(expert_accuracies):
        start_class, end_class = CLASS_RANGES[expert_idx]
        log_message(f"专家{expert_idx}[类别{start_class}-{end_class}]准确率: {acc:.4f}")

    log_message(f"完整模型已保存到 {os.path.join(RESULTS_PATH, 'param', 'final_model.pth')}")
    log_message(f"=== 训练结束于 {datetime.datetime.now().strftime('%Y.%m.%d_%H:%M:%S')} ===")
    sys.exit(0)
