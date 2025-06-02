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
from models.MoE import MoE4Model
from models.ResNet import resnet18

from utils.data_loading_mine import log_message, load_data, get_dataloaders, CLASS_RANGES
from utils.load_model import load_pretrained_weights, load_checkpoint
from utils.loss_functions import compute_specialized_loss, combine_expert_outputs
from utils.visualization import log_metrics_to_tensorboard, plot_loss_curves, plot_accuracy_curves, \
    plot_expert_logits_histograms
from validate_model import validate_expert, validate_full_model, validate_router_accuracy, get_expert_indices, \
    compare_model_parameters

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
RESULTS_PATH = "./results/AppClassNet/top200/MoE/55"
# 预训练模型路径
PRETRAINED_RESNET18_PATH = "./results/AppClassNet/top200/ResNet/1/param/model_epoch_800.pth"  # 预训练ResNet18模型路径
# 确保结果目录存在
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(f"{RESULTS_PATH}/param", exist_ok=True)
os.makedirs(f"{RESULTS_PATH}/logs", exist_ok=True)
os.makedirs(f"{RESULTS_PATH}/analysis", exist_ok=True)  # 确保分析结果目录存在

# 缓存文件路径
CACHE_PATH = os.path.join(RESULTS_PATH, "param", "cached_computations.pth")

# 创建日志文件
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"{RESULTS_PATH}/logs/training_log_{current_time}.txt"

# 优化超参数
BATCH_SIZE = 2048  # 批次大小
EPOCHS = 40  # 端到端训练的总轮数
LEARNING_RATE = 0.0005  # 端到端训练的学习率
NUM_CLASSES = 200  # AppClassNet 类别数
NUM_EXPERTS = len(CLASS_RANGES)  # MoE专家头数量
NUM_WORKERS = 2  # 数据加载的worker数量
PIN_MEMORY = True  # 确保启用pin_memory
PREFETCH_FACTOR = 4  # 增加预取因子
# 自动混合精度训练配置
USE_AMP = True  # 启动自动混合精度训练


# 新的端到端训练函数，将所有组件一起训练
def train_end_to_end(model, train_loader, val_loader, test_loader, device, resume_training=False):
    """
    端到端训练整个模型：backbone + 专家 + 整合层同时训练
    不再分阶段训练，而是从一开始就训练整个网络架构
    """
    log_message(f"开始端到端训练...")
    writer = SummaryWriter(os.path.join(RESULTS_PATH, 'logs', 'end_to_end'))

    # 创建混合精度训练的scaler
    scaler = GradScaler() if USE_AMP else None

    # 训练所有参数，不冻结backbone
    # 为不同组件设置不同的学习率
    backbone_params = {'params': model.backbone.parameters(), 'lr': LEARNING_RATE * 0.1}  # backbone使用较小的学习率
    experts_params = {'params': [p for expert in model.experts for p in expert.parameters()], 'lr': LEARNING_RATE}
    integrator_params = {'params': model.integrator.parameters(), 'lr': LEARNING_RATE * 1.5}  # 整合层使用略大的学习率

    # 创建优化器，同时优化所有参数
    optimizer = optim.Adam([backbone_params, experts_params, integrator_params])

    # 使用余弦退火学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 添加恢复训练功能
    start_epoch = 0
    best_val_acc = 0.0

    if resume_training:
        checkpoint_path = os.path.join(RESULTS_PATH, 'param', 'end_to_end_latest.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_val_acc = checkpoint['best_val_acc']
                log_message(f"已加载检查点，从第{start_epoch}轮继续训练，当前最佳验证准确率: {best_val_acc:.4f}")

    # 记录训练历史
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # 开始训练
    for epoch in range(start_epoch, EPOCHS):
        epoch_start_time = time.time()
        log_message(f"端到端训练 - Epoch {epoch + 1}/{EPOCHS}")
        model.train()  # 所有组件都设为训练模式

        # 启用第二阶段模式，这样在前向传播时会使用整合层
        model.stage2_mode = True

        # 训练一个epoch
        epoch_loss = 0
        correct = 0
        total = 0
        batch_train_time = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            batch_start_time = time.time()
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # 更快地重置梯度

            # 使用混合精度训练
            if USE_AMP and scaler is not None:
                with autocast():
                    # 前向传播
                    _, expert_outputs, logits = model(inputs)

                    # 主损失：整体预测的交叉熵损失
                    main_loss = criterion(logits, targets)

                    # 额外的专家单独训练损失
                    expert_losses = 0
                    for expert_idx, expert in enumerate(model.experts):
                        start_class, end_class = model.class_ranges[expert_idx]
                        # 筛选出属于该专家负责范围的样本
                        mask = (targets >= start_class) & (targets <= end_class)
                        if mask.sum() > 0:
                            # 提取相关样本和标签
                            expert_inputs = inputs[mask]
                            expert_targets = targets[mask]
                            # expert_targets = targets[mask] - start_class  # 相对于专家范围的标签

                            # 计算专家的输出和损失
                            if len(expert_inputs) > 0:
                                features = model.backbone(expert_inputs)
                                expert_output = expert(features)
                                cls_loss, reg_loss, total_loss = model.compute_loss(expert_output, expert_targets,
                                                                                    expert_idx)
                                expert_losses += total_loss * 0.2  # 权重调整为主损失的0.2倍

                    # 总损失 = 主损失 + 额外的专家损失
                    loss = main_loss + expert_losses

                # 使用scaler进行反向传播和优化
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # 前向传播
                _, expert_outputs, logits = model(inputs)

                # 主损失：整体预测的交叉熵损失
                main_loss = criterion(logits, targets)

                # 额外的专家单独训练损失
                expert_losses = 0
                for expert_idx, expert in enumerate(model.experts):
                    start_class, end_class = model.class_ranges[expert_idx]
                    # 筛选出属于该专家负责范围的样本
                    mask = (targets >= start_class) & (targets <= end_class)
                    if mask.sum() > 0:
                        # 提取相关样本和标签
                        expert_inputs = inputs[mask]
                        expert_targets = targets[mask] - start_class  # 相对于专家范围的标签

                        # 计算专家的输出和损失
                        if len(expert_inputs) > 0:
                            features = model.backbone(expert_inputs)
                            expert_output = expert(features)
                            cls_loss, reg_loss, total_loss = model.compute_loss(expert_output, expert_targets,
                                                                                expert_idx)
                            expert_losses += total_loss * 0.2  # 权重调整为主损失的0.2倍

                # 总损失 = 主损失 + 额外的专家损失
                loss = main_loss + expert_losses

                # 反向传播与优化
                loss.backward()
                optimizer.step()

            batch_time_taken = time.time() - batch_start_time
            batch_train_time += batch_time_taken

            # 统计
            epoch_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # 更新学习率
        scheduler.step()

        # 计算训练集平均损失和准确率
        avg_loss = epoch_loss / len(train_loader)
        accuracy = correct / total
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        epoch_time_taken = time.time() - epoch_start_time

        # 记录TensorBoard指标
        writer.add_scalar('train/loss', avg_loss, epoch)
        writer.add_scalar('train/accuracy', accuracy, epoch)

        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/learning_rate', current_lr, epoch)

        log_message(
            f"  训练损失: {avg_loss:.4f}, 准确率: {accuracy:.4f}, 学习率: {current_lr:.6f}, 耗时: {epoch_time_taken:.2f}s")

        # 验证模型
        val_start_time = time.time()
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # 确保在stage2_mode下进行推理
                _, _, logits = model(inputs)

                loss = criterion(logits, targets)
                val_loss += loss.item()
                _, predicted = logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        # 计算验证集平均损失和准确率
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        val_time_taken = time.time() - val_start_time

        # 记录TensorBoard指标
        writer.add_scalar('val/loss', avg_val_loss, epoch)
        writer.add_scalar('val/accuracy', val_accuracy, epoch)

        log_message(f"  验证损失: {avg_val_loss:.4f}, 准确率: {val_accuracy:.4f}, 耗时: {val_time_taken:.2f}s")

        # 测试模型
        test_start_time = time.time()
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # 确保在stage2_mode下进行推理
                _, _, logits = model(inputs)

                loss = criterion(logits, targets)
                test_loss += loss.item()
                _, predicted = logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        # 计算测试集平均损失和准确率
        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = correct / total
        test_time_taken = time.time() - test_start_time

        # 记录TensorBoard指标
        writer.add_scalar('test/loss', avg_test_loss, epoch)
        writer.add_scalar('test/accuracy', test_accuracy, epoch)

        log_message(f"  测试损失: {avg_test_loss:.4f}, 准确率: {test_accuracy:.4f}, 耗时: {test_time_taken:.2f}s")

        # 保存最新检查点
        checkpoint_path = os.path.join(RESULTS_PATH, 'param', 'end_to_end_latest.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': avg_val_loss,
            'val_accuracy': val_accuracy,
            'best_val_acc': max(best_val_acc, val_accuracy)
        }, checkpoint_path)

        # 如果是最佳性能，保存最佳检查点
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_checkpoint_path = os.path.join(RESULTS_PATH, 'param', 'end_to_end_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy,
                'best_val_acc': best_val_acc
            }, best_checkpoint_path)
            log_message(f"  已保存最佳模型，验证准确率: {best_val_acc:.4f}")

        # 定期保存阶段性检查点
        if (epoch + 1) % 5 == 0 or (epoch + 1) == EPOCHS:
            epoch_checkpoint_path = os.path.join(RESULTS_PATH, 'param', f'end_to_end_epoch{epoch + 1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy,
                'best_val_acc': best_val_acc
            }, epoch_checkpoint_path)
            log_message(f"  已保存Epoch {epoch + 1}检查点")

    # 绘制损失和准确率曲线
    loss_curve_path = os.path.join(RESULTS_PATH, 'logs', 'end_to_end_loss_curve.png')
    acc_curve_path = os.path.join(RESULTS_PATH, 'logs', 'end_to_end_accuracy_curve.png')
    plot_loss_curves(train_losses, val_losses, loss_curve_path)
    plot_accuracy_curves(train_accuracies, val_accuracies, acc_curve_path)
    log_message(f"已保存端到端训练的损失和准确率曲线")

    # 预计算专家缩放因子
    log_message("训练完成，预计算专家缩放因子...")
    if hasattr(model, 'module'):  # 处理DataParallel包装的模型
        model.module.precompute_scaling_factors()
    else:
        model.precompute_scaling_factors()

    writer.close()

    return model


# 优化评估函数，确保正确使用推理方法
def evaluate_ensemble_model(model, val_loader, test_loader, device, FLAG=True):
    """
    评估整体模型性能
    采用Model design.md中定义的推理输出方式：直接组合各专家的输出
    """
    log_message("开始评估整体模型性能...")

    # 确保模型已经预计算了缩放因子
    if not model.has_cached_computation:
        if hasattr(model, 'module'):  # 处理DataParallel包装的模型
            log_message("预计算专家缩放因子...")
            model.module.precompute_scaling_factors()
        else:
            log_message("预计算专家缩放因子...")
            model.precompute_scaling_factors()

    criterion = nn.CrossEntropyLoss()
    # 评估验证集
    val_start_time = time.time()
    val_loss, val_accuracy = validate_full_model(model, val_loader, criterion, device, RESULTS_PATH, FLAG)
    val_time = time.time() - val_start_time
    log_message(f"验证集性能 - 损失: {val_loss:.4f}, 准确率: {val_accuracy:.4f}, 耗时: {val_time:.2f}s")

    # 评估测试集
    test_start_time = time.time()
    test_loss, test_accuracy = validate_full_model(model, test_loader, criterion, device, RESULTS_PATH, FLAG)
    test_time = time.time() - test_start_time
    log_message(f"测试集性能 - 损失: {test_loss:.4f}, 准确率: {test_accuracy:.4f}, 耗时: {test_time:.2f}s")

    return val_accuracy, test_accuracy


def run_model_analysis(model, test_loader, device):
    """
    运行模型分析，调用analyse_MoE.py中的函数
    """
    log_message("开始运行模型分析...")

    # 确保模型已经预计算了缩放因子
    if not model.has_cached_computation:
        if hasattr(model, 'module'):  # 处理DataParallel包装的模型
            log_message("预计算专家缩放因子...")
            model.module.precompute_scaling_factors()
        else:
            log_message("预计算专家缩放因子...")
            model.precompute_scaling_factors()

    # 分析模型
    predictions, targets = analyze_model(model, test_loader, device)

    # 计算每个类别的准确率
    per_class_accuracy, per_class_correct, per_class_total = calculate_per_class_accuracy(
        predictions, targets, NUM_CLASSES)

    # 保存每个类别的准确率数据
    accuracy_data = {
        'class': np.arange(NUM_CLASSES),
        'accuracy': per_class_accuracy,
        'correct': per_class_correct,
        'total': per_class_total
    }

    pd.DataFrame(accuracy_data).to_csv(f"{ANALYSIS_RESULTS_PATH}/per_class_accuracy.csv", index=False)

    # 绘制每个类别的准确率柱状图
    plot_per_class_accuracy(per_class_accuracy, per_class_total, CLASS_RANGES)

    # 创建混淆矩阵
    cm, cm_normalized = create_confusion_matrix(predictions, targets, NUM_CLASSES, CLASS_RANGES)

    # 分析错分情况
    class_report, expert_report = analyze_misclassified(cm, cm_normalized, CLASS_RANGES)

    # 计算总体准确率
    overall_accuracy = np.sum(per_class_correct) / np.sum(per_class_total)
    log_message(f"整体模型准确率: {overall_accuracy:.4%}")

    # 打印专家性能汇总
    log_message("\n专家性能汇总:")
    log_message(expert_report.to_string())

    # 打印前10个错误率最高的类别
    log_message("\n错误率最高的10个类别:")
    log_message(class_report.head(10).to_string())

    log_message(f"\n分析结果保存至: {ANALYSIS_RESULTS_PATH}")


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
    log_message(f"学习率: {LEARNING_RATE}, 训练轮数: {EPOCHS}")
    log_message(f"数据集路径: {DATASET_PATH}")
    log_message(f"结果保存路径: {RESULTS_PATH}")
    log_message(f"使用设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    log_message(f"自动混合精度: {USE_AMP}")
    log_message(f"工作进程数: {NUM_WORKERS}")
    log_message(f"专家数量: {NUM_EXPERTS}")
    log_message(f"类别范围: {CLASS_RANGES}")
    log_message(f"预训练模型路径: {PRETRAINED_RESNET18_PATH}")

    # 配置数据并行训练，利用多个GPU
    multi_gpu = torch.cuda.device_count() > 1
    log_message(f"使用GPU数量: {torch.cuda.device_count()}")

    # 添加继续训练选项
    RESUME_TRAINING = True  # 设置是否从检查点继续训练

    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据 - 只需要完整数据集的数据加载器
    _, _, _, full_train_loader, full_val_loader, full_test_loader = get_dataloaders(CLASS_RANGES)

    # 从训练数据中获取输入形状
    sample_input, _ = next(iter(full_train_loader))
    input_channels = sample_input.shape[1]
    input_height = sample_input.shape[2]
    input_width = sample_input.shape[3]

    log_message(f"模型输入形状: 通道={input_channels}, 高度={input_height}, 宽度={input_width}")

    # 创建模型
    model = MoE4Model(
        total_classes=NUM_CLASSES,
        class_ranges=CLASS_RANGES,
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width
    )

    # 加载预训练ResNet18权重
    model = load_pretrained_weights(model, PRETRAINED_RESNET18_PATH)

    # 如果有多GPU，使用DataParallel
    if multi_gpu:
        log_message("启用多GPU并行训练")
        model = nn.DataParallel(model)

    model.to(device)

    # 端到端训练模型
    log_message("开始端到端训练...")
    model = train_end_to_end(model, full_train_loader, full_val_loader, full_test_loader, device,
                             resume_training=RESUME_TRAINING)

    # 保存最终模型
    final_model_path = os.path.join(RESULTS_PATH, 'param', 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_ranges': model.class_ranges,
        'total_classes': model.total_classes,
        'stage2_mode': True,
        'has_cached_computation': True,
        'cached_scaling_factors': model.cached_scaling_factors if not hasattr(model,
                                                                              'module') else model.module.cached_scaling_factors
    }, final_model_path)
    log_message(f"完整模型已保存到 {final_model_path}")

    # 评估最终模型
    log_message("评估最终模型性能...")
    val_accuracy, test_accuracy = evaluate_ensemble_model(model, full_val_loader, full_test_loader, device, False)

    # 记录最终结果
    log_message(f"训练完成！")
    log_message(f"验证集准确率: {val_accuracy:.4f}")
    log_message(f"测试集准确率: {test_accuracy:.4f}")

    # 运行模型分析
    log_message("开始对模型进行详细分析...")
    try:
        # 运行分析
        run_model_analysis(model, full_test_loader, device)
        log_message("模型分析完成!")
    except Exception as e:
        log_message(f"运行模型分析时出错: {e}")
        import traceback

        log_message(traceback.format_exc())

    log_message(f"=== 训练结束于 {datetime.datetime.now().strftime('%Y.%m.%d_%H:%M:%S')} ===")
    sys.exit(0)
