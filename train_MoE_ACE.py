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

from models.MoE_ACE import MoE_ACE
from models.ResNet import resnet18

from utils.data_loading_mine import log_message, load_data, get_dataloaders, CLASS_RANGES, RESULTS_PATH
from utils.load_model import load_pretrained_weights, load_checkpoint
from utils.loss_functions import compute_specialized_loss, combine_expert_outputs
from utils.visualization import log_metrics_to_tensorboard, plot_loss_curves, plot_accuracy_curves, \
    plot_expert_logits_histograms
from validate_model import validate_expert, validate_full_model

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
RESULTS_PATH = "./results/AppClassNet/top200/MoE-ACE/1"
# 预训练模型路径
PRETRAINED_RESNET18_PATH = "./results/AppClassNet/top200/ResNet/1/param/model_epoch_800.pth"

# 确保结果目录存在
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(f"{RESULTS_PATH}/param", exist_ok=True)
os.makedirs(f"{RESULTS_PATH}/logs", exist_ok=True)
os.makedirs(f"{RESULTS_PATH}/analysis", exist_ok=True)

# 缓存文件路径
CACHE_PATH = os.path.join(RESULTS_PATH, "param", "cached_computations.pth")

# 创建日志文件
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"{RESULTS_PATH}/logs/training_log_{current_time}.txt"

# 优化超参数
BATCH_SIZE = 2048  # 批次大小
EPOCHS = 60  # 训练轮数
LEARNING_RATE = 0.001  # 学习率
NUM_CLASSES = 200  # AppClassNet 类别数
NUM_EXPERTS = len(CLASS_RANGES)  # MoE专家头数量
# 自动混合精度训练配置
USE_AMP = True  # 启动自动混合精度训练


def train_moe_ace(model, train_loaders, val_loaders, test_loaders, device, resume_training=False):
    """
    训练MoE-ACE模型
    
    按照MoE-ACE.md:
    - 每个专家有自己的目标类范围
    - 每个专家只会接收标签属于自己目标类的样本来进行训练
    - 专家的损失函数包含分类损失和正则化项
    - 每个专家只训练自己独享的深层，除第一个专家外
    """
    log_message(f"开始MoE-ACE模型训练...")
    writer = SummaryWriter(os.path.join(RESULTS_PATH, 'logs'))

    log_message("根据MoE-ACE.md要求配置模型:")
    log_message(f"- 共{len(model.experts)}个专家")
    for i, (start, end) in enumerate(model.class_ranges):
        log_message(f"- 专家{i}负责类别范围: [{start}, {end}]")

    # 创建混合精度训练的scaler
    scaler = GradScaler() if USE_AMP else None

    # 按照说明文档要求，每个专家只能训练自己独享的深层，除第一个专家外
    # 第一个专家（负责所有类的专家）可以训练共享的浅层和独享的深层

    # 外层循环：每个专家
    for expert_idx in range(len(model.experts)):
        log_message(f"===== 开始训练专家 {expert_idx + 1}/{len(model.experts)} =====")
        start_class, end_class = model.class_ranges[expert_idx]
        log_message(f"专家{expert_idx}负责类别范围: [{start_class}, {end_class}]")

        # 冻结所有参数
        for param in model.parameters():
            param.requires_grad = False

        # 对于第一个专家（负责所有类别的专家），启用共享层和它的深层
        if expert_idx == 0:
            log_message("专家0负责所有类别，训练共享浅层和独享深层")
            # 启用共享浅层
            for param in model.shared_backbone.parameters():
                param.requires_grad = True
            # 启用当前专家的深层
            for param in model.expert_backbones[expert_idx].parameters():
                param.requires_grad = True
            # 启用当前专家的分类器
            for param in model.experts[expert_idx].parameters():
                param.requires_grad = True
        else:
            log_message(f"专家{expert_idx}只训练独享的深层和分类器")
            # 只启用当前专家的深层
            for param in model.expert_backbones[expert_idx].parameters():
                param.requires_grad = True
            # 启用当前专家的分类器
            for param in model.experts[expert_idx].parameters():
                param.requires_grad = True

        # 创建优化器，只优化当前可训练的参数
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

        # 添加恢复训练功能
        start_epoch = 0
        best_val_acc = 0.0

        if resume_training:
            expert_checkpoint_path = os.path.join(RESULTS_PATH, 'param', f'expert{expert_idx}_latest.pth')
            if os.path.exists(expert_checkpoint_path):
                checkpoint = torch.load(expert_checkpoint_path)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch'] + 1
                if 'best_val_acc' in checkpoint:
                    best_val_acc = checkpoint['best_val_acc']
                log_message(
                    f"已加载专家{expert_idx}的检查点，从第{start_epoch}轮继续训练，当前最佳准确率: {best_val_acc:.2f}")

        # 获取当前专家的数据加载器
        train_loader = train_loaders[expert_idx]
        val_loader = val_loaders[expert_idx]
        test_loader = test_loaders[expert_idx]

        # 记录训练历史
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        # 内层循环：该专家的训练周期
        for epoch in range(start_epoch, EPOCHS):
            epoch_start_time = time.time()
            log_message(f"专家{expert_idx} - Epoch {epoch + 1}/{EPOCHS}")

            # 训练模式
            model.train()

            # 训练一个epoch
            epoch_cls_loss = 0
            epoch_reg_loss = 0
            epoch_total_loss = 0
            correct = 0
            total = 0
            batch_times = []

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                batch_start_time = time.time()
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                # 使用混合精度训练
                if USE_AMP and scaler is not None:
                    with autocast():
                        # 前向传播
                        expert_feats, expert_outputs, _ = model(inputs)

                        # 计算当前专家的损失
                        cls_loss, reg_loss, total_loss = model.compute_loss(expert_outputs[expert_idx], targets,
                                                                            expert_idx)

                    # 使用scaler进行反向传播和优化
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # 前向传播
                    expert_feats, expert_outputs, _ = model(inputs)

                    # 计算当前专家的损失
                    cls_loss, reg_loss, total_loss = model.compute_loss(expert_outputs[expert_idx], targets, expert_idx)

                    # 反向传播与优化
                    total_loss.backward()
                    optimizer.step()

                batch_time_taken = time.time() - batch_start_time
                batch_times.append(batch_time_taken)

                # 统计
                epoch_cls_loss += cls_loss.item()
                epoch_reg_loss += reg_loss.item()
                epoch_total_loss += total_loss.item()
                _, predicted = expert_outputs[expert_idx].max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                    log_message(f"  批次 {batch_idx + 1}/{len(train_loader)}, "
                                f"损失: {total_loss.item():.4f}, "
                                f"准确率: {100. * correct / total:.2f}%, "
                                f"平均批次耗时: {sum(batch_times) / len(batch_times):.2f}s")

            # 记录专家的训练损失和准确率
            avg_cls_loss = epoch_cls_loss / len(train_loader) if len(train_loader) > 0 else 0
            avg_reg_loss = epoch_reg_loss / len(train_loader) if len(train_loader) > 0 else 0
            avg_total_loss = epoch_total_loss / len(train_loader) if len(train_loader) > 0 else 0
            accuracy = correct / total if total > 0 else 0
            epoch_time_taken = time.time() - epoch_start_time

            train_losses.append(avg_total_loss)
            train_accuracies.append(accuracy)

            # 记录TensorBoard指标
            metrics = {
                'cls_loss': avg_cls_loss,
                'reg_loss': avg_reg_loss,
                'total_loss': avg_total_loss,
                'accuracy': accuracy
            }
            log_metrics_to_tensorboard(writer, metrics, epoch, prefix=f'expert{expert_idx}/train_')

            log_message(
                f"  专家{expert_idx} - 训练损失: 分类={avg_cls_loss:.4f}, 正则={avg_reg_loss:.4f}, 总计={avg_total_loss:.4f}, "
                f"准确率: {accuracy:.4f}, 耗时: {epoch_time_taken:.2f}s")

            # 验证当前专家
            model.eval()
            val_loss, val_accuracy = validate_expert_ace(model, val_loader, expert_idx, device)

            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            # 记录验证集指标
            val_metrics = {
                'loss': val_loss,
                'accuracy': val_accuracy
            }
            log_metrics_to_tensorboard(writer, val_metrics, epoch, prefix=f'expert{expert_idx}/val_')

            log_message(f"  专家{expert_idx} - 验证损失: {val_loss:.4f}, 准确率: {val_accuracy:.4f}")

            # 测试当前专家
            test_loss, test_accuracy = validate_expert_ace(model, test_loader, expert_idx, device)

            # 记录测试集指标
            test_metrics = {
                'loss': test_loss,
                'accuracy': test_accuracy
            }
            log_metrics_to_tensorboard(writer, test_metrics, epoch, prefix=f'expert{expert_idx}/test_')

            log_message(f"  专家{expert_idx} - 测试损失: {test_loss:.4f}, 准确率: {test_accuracy:.4f}")

            # 保存当前专家的最新检查点
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

            # 定期保存阶段性检查点
            if (epoch + 1) % 5 == 0 or (epoch + 1) == EPOCHS:
                epoch_checkpoint_path = os.path.join(RESULTS_PATH, 'param', f'expert{expert_idx}_epoch{epoch + 1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'best_val_acc': best_val_acc
                }, epoch_checkpoint_path)
                log_message(f"  已保存专家{expert_idx}的Epoch {epoch + 1}检查点")

        # 绘制该专家的损失和准确率曲线
        loss_curve_path = os.path.join(RESULTS_PATH, 'logs', f'expert{expert_idx}_loss_curve.png')
        acc_curve_path = os.path.join(RESULTS_PATH, 'logs', f'expert{expert_idx}_accuracy_curve.png')
        plot_loss_curves(train_losses, val_losses, loss_curve_path)
        plot_accuracy_curves(train_accuracies, val_accuracies, acc_curve_path)

        # 清理该专家的数据加载器
        del train_loader, val_loader, test_loader
        torch.cuda.empty_cache()
        gc.collect()

        # 该专家训练完成
        log_message(f"专家{expert_idx}训练完成，最佳验证准确率: {best_val_acc:.4f}")

    # 所有专家训练完成后，保存完整模型
    final_model_path = os.path.join(RESULTS_PATH, 'param', 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_ranges': model.class_ranges,
        'total_classes': model.total_classes
    }, final_model_path)
    log_message(f"所有专家训练完成，完整模型已保存到 {final_model_path}")

    writer.close()
    return model


def validate_expert_ace(model, val_loader, expert_idx, device):
    """验证MoE-ACE模型中单个专家的性能"""
    model.eval()
    val_loss = 0
    cls_loss_sum = 0
    reg_loss_sum = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播
            expert_feats, expert_outputs, _ = model(inputs)

            # 计算损失
            cls_loss, reg_loss, total_loss = model.compute_loss(expert_outputs[expert_idx], targets, expert_idx)

            # 累积损失
            val_loss += total_loss.item()
            cls_loss_sum += cls_loss.item()
            reg_loss_sum += reg_loss.item()

            # 计算准确率
            _, predicted = expert_outputs[expert_idx].max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # 计算平均损失和准确率
    avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
    accuracy = correct / total if total > 0 else 0

    return avg_val_loss, accuracy


def evaluate_moe_ace_model(model, val_loader, test_loader, device):
    """评估整体MoE-ACE模型性能"""
    log_message("开始评估MoE-ACE模型整体性能...")

    # 确保模型已经预计算了缩放因子
    if not model.has_cached_computation:
        log_message("预计算专家缩放因子...")
        model.precompute_scaling_factors()

    criterion = nn.CrossEntropyLoss()

    # 评估验证集
    val_start_time = time.time()
    val_loss, val_accuracy = validate_full_ace_model(model, val_loader, criterion, device)
    val_time = time.time() - val_start_time
    log_message(f"验证集性能 - 损失: {val_loss:.4f}, 准确率: {val_accuracy:.4f}, 耗时: {val_time:.2f}s")

    # 评估测试集
    test_start_time = time.time()
    test_loss, test_accuracy = validate_full_ace_model(model, test_loader, criterion, device)
    test_time = time.time() - test_start_time
    log_message(f"测试集性能 - 损失: {test_loss:.4f}, 准确率: {test_accuracy:.4f}, 耗时: {test_time:.2f}s")

    return val_accuracy, test_accuracy


def validate_full_ace_model(model, data_loader, criterion, device):
    """验证完整MoE-ACE模型性能"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # 使用模型的inference方法进行推理
            _, combined_logits = model.inference(inputs)

            # 计算损失
            loss = criterion(combined_logits, targets)
            test_loss += loss.item()

            # 计算准确率
            _, predicted = combined_logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # 计算平均损失和准确率
    avg_test_loss = test_loss / len(data_loader)
    accuracy = correct / total if total > 0 else 0

    return avg_test_loss, accuracy


def run_model_analysis(model, test_loader, device):
    """运行模型分析"""
    log_message("开始运行MoE-ACE模型分析...")

    # 确保模型已经预计算了缩放因子
    if not model.has_cached_computation:
        log_message("预计算专家缩放因子...")
        model.precompute_scaling_factors()

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            _, logits = model.inference(inputs)
            _, predicted = logits.max(1)

            all_preds.append(predicted.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # 将预测和标签合并
    predictions = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

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

    os.makedirs(ANALYSIS_RESULTS_PATH, exist_ok=True)
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

    return overall_accuracy


if __name__ == "__main__":
    # 设置性能优化
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # 记录训练开始信息和配置信息
    log_message(f"=== MoE-ACE模型训练开始于 {current_time} ===")
    log_message(f"BatchSize: {BATCH_SIZE}")
    log_message(f"学习率: {LEARNING_RATE}, 训练轮数: {EPOCHS}")
    log_message(f"数据集路径: {DATASET_PATH}")
    log_message(f"结果保存路径: {RESULTS_PATH}")
    log_message(f"使用设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    log_message(f"自动混合精度: {USE_AMP}")
    log_message(f"专家数量: {NUM_EXPERTS}")
    log_message(f"类别范围: {CLASS_RANGES}")
    log_message(f"预训练模型路径: {PRETRAINED_RESNET18_PATH}")

    # 配置数据并行训练
    multi_gpu = torch.cuda.device_count() > 1
    log_message(f"使用GPU数量: {torch.cuda.device_count()}")

    # 是否从检查点继续训练
    RESUME_TRAINING = False

    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    train_loaders, val_loaders, test_loaders, full_train_loader, full_val_loader, full_test_loader = get_dataloaders(
        CLASS_RANGES)

    # 从训练数据中获取输入形状
    sample_input, _ = next(iter(full_train_loader))
    input_channels = sample_input.shape[1]
    input_height = sample_input.shape[2]
    input_width = sample_input.shape[3]

    log_message(f"模型输入形状: 通道={input_channels}, 高度={input_height}, 宽度={input_width}")

    # 创建MoE-ACE模型
    model = MoE_ACE(
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

    # 训练MoE-ACE模型
    model = train_moe_ace(model, train_loaders, val_loaders, test_loaders, device, resume_training=RESUME_TRAINING)

    # 预计算缩放因子
    log_message("训练完成，预计算专家缩放因子...")
    if hasattr(model, 'module'):  # 处理DataParallel包装的模型
        model.module.precompute_scaling_factors()
    else:
        model.precompute_scaling_factors()

    # 保存带有预计算结果的完整模型
    log_message("保存带有预计算缩放因子的完整模型...")
    final_model_path = os.path.join(RESULTS_PATH, 'param', 'final_model_with_scaling.pth')

    if hasattr(model, 'module'):
        scaling_factors = model.module.cached_scaling_factors
        has_cached = model.module.has_cached_computation
    else:
        scaling_factors = model.cached_scaling_factors
        has_cached = model.has_cached_computation

    torch.save({
        'model_state_dict': model.state_dict(),
        'class_ranges': CLASS_RANGES,
        'total_classes': NUM_CLASSES,
        'has_cached_computation': has_cached,
        'cached_scaling_factors': scaling_factors
    }, final_model_path)
    log_message(f"带有预计算缩放因子的完整模型已保存到 {final_model_path}")

    # 评估最终模型
    log_message("评估MoE-ACE模型性能...")
    val_accuracy, test_accuracy = evaluate_moe_ace_model(model, full_val_loader, full_test_loader, device)

    # 记录最终结果
    log_message(f"MoE-ACE模型训练完成！")
    log_message(f"验证集准确率: {val_accuracy:.4f}")
    log_message(f"测试集准确率: {test_accuracy:.4f}")

    # 运行模型分析
    log_message("开始对MoE-ACE模型进行详细分析...")
    try:
        # 运行分析
        final_accuracy = run_model_analysis(model, full_test_loader, device)
        log_message(f"模型分析完成! 最终准确率: {final_accuracy:.4f}")
    except Exception as e:
        log_message(f"运行模型分析时出错: {e}")
        import traceback

        log_message(traceback.format_exc())

    log_message(f"=== MoE-ACE模型训练结束于 {datetime.datetime.now().strftime('%Y.%m.%d_%H:%M:%S')} ===")
    sys.exit(0)
