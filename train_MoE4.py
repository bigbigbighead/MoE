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
RESULTS_PATH = "./results/AppClassNet/top200/MoE/32"
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
EPOCHS_STAGE1 = 10  # 第一阶段训练轮数
EPOCHS_EXPERT0 = 10
EPOCH_EXPERTS = 20
LEARNING_RATE_STAGE1 = 0.001  # 第一阶段学习率
NUM_CLASSES = 200  # AppClassNet 类别数
NUM_EXPERTS = len(CLASS_RANGES)  # MoE专家头数量
NUM_WORKERS = 2  # 数据加载的worker数量
PIN_MEMORY = True  # 确保启用pin_memory
PREFETCH_FACTOR = 4  # 增加预取因子
# 自动混合精度训练配置
USE_AMP = True  # 启动自动混合精度训练


# 优化训练过程，确保在训练和记录指标时的一致性
def train_stage1(model, train_loaders, val_loaders, test_loaders, device, resume_training=False):
    """
    第一阶段训练：分别训练每个专家
    
    按照Model design.md:
    - 每个专家有自己的目标类范围
    - 每个专家只会接收标签属于自己目标类的样本来进行训练
    - 专家的损失函数包含分类损失和正则化项
    """
    log_message(f"开始专家训练...")
    writer = SummaryWriter(os.path.join(RESULTS_PATH, 'logs', 'stage1'))

    # 冻结backbone参数
    for param in model.backbone.parameters():
        param.requires_grad = False

    log_message("已冻结backbone参数，仅训练专家分类器")

    # 创建混合精度训练的scaler
    scaler = GradScaler() if USE_AMP else None

    # 外层循环：每个专家
    for expert_idx in range(len(model.experts)):
        log_message(f"===== 开始训练专家 {expert_idx + 1}/{len(model.experts)} =====")
        start_class, end_class = model.class_ranges[expert_idx]
        log_message(f"专家{expert_idx}负责类别范围: [{start_class}, {end_class}]")

        # 只优化当前专家的参数
        optimizer = optim.Adam(model.experts[expert_idx].parameters(), lr=LEARNING_RATE_STAGE1)

        # 添加恢复训练功能
        start_epoch = 0
        best_val_acc = 0.0

        if resume_training:
            final_model_path = os.path.join(RESULTS_PATH, 'param', 'final_model.pth')
            if start_epoch != 0:
                checkpoint = torch.load(final_model_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                log_message(f"成功加载final完整模型参数：{final_model_path}")
            else:
                expert_checkpoint_path = os.path.join(RESULTS_PATH, 'param', f'stage1_expert{expert_idx}_latest.pth')
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

                        log_message(
                            f"已加载专家{expert_idx}的检查点，从第{start_epoch}轮继续训练，当前最佳验证准确率: {best_val_acc:.2f}")

        # 获取当前专家的数据加载器 - train_loaders[expert_idx]只包含该专家负责类别的样本
        train_loader = train_loaders[expert_idx]
        val_loader_expert = val_loaders[expert_idx]
        test_loader_expert = test_loaders[expert_idx]

        # 记录训练历史
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        if expert_idx == 0:
            EPOCHS_STAGE1 = EPOCHS_EXPERT0
        else:
            EPOCHS_STAGE1 = EPOCH_EXPERTS
        # 内层循环：该专家的完整训练周期
        for epoch in range(start_epoch, EPOCHS_STAGE1):
            epoch_start_time = time.time()
            log_message(f"专家{expert_idx} - Epoch {epoch + 1}/{EPOCHS_STAGE1}")
            model.train()
            model.backbone.eval()  # 确保backbone在评估模式下
            # 训练一个epoch
            epoch_cls_loss = 0
            epoch_reg_loss = 0
            epoch_total_loss = 0
            correct = 0
            total = 0
            batch_train_time = 0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                batch_start_time = time.time()
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

                # 确保targets是相对于当前专家负责范围的
                # train_loaders已经确保这一点，这里只是进行确认
                # 注意：targets值已经是相对于专家范围的，例如专家1负责[100-149]，则targets值范围为[0-49]

                optimizer.zero_grad(set_to_none=True)  # 更快地重置梯度

                # 使用混合精度训练
                if USE_AMP and scaler is not None:
                    with autocast():
                        # 前向传播
                        features = model.backbone(inputs)
                        outputs = model.experts[expert_idx](features)

                        # 使用损失函数计算 - 包含分类损失和正则化项
                        cls_loss, reg_loss, total_loss = model.compute_loss(outputs, targets, expert_idx)

                    # 使用scaler进行反向传播和优化
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # 前向传播
                    features = model.backbone(inputs)
                    outputs = model.experts[expert_idx](features)

                    # 使用损失函数计算
                    cls_loss, reg_loss, total_loss = model.compute_loss(outputs, targets, expert_idx)

                    # 反向传播与优化
                    total_loss.backward()
                    optimizer.step()

                batch_time_taken = time.time() - batch_start_time
                batch_train_time += batch_time_taken

                # 统计
                epoch_cls_loss += cls_loss.item()
                epoch_reg_loss += reg_loss.item()
                epoch_total_loss += total_loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            # 记录每个专家的训练损失和准确率
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

            # 使用专家对应的验证集验证当前专家的性能
            valid_start_time = time.time()
            val_loss, val_accuracy = validate_expert(model, val_loader_expert, None, expert_idx, device)
            valid_end_time = time.time()

            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            # 记录验证集指标
            val_metrics = {
                'loss': val_loss,
                'accuracy': val_accuracy
            }
            log_metrics_to_tensorboard(writer, val_metrics, epoch, prefix=f'expert{expert_idx}/val_')

            log_message(
                f"  专家{expert_idx} - 验证损失: {val_loss:.4f}, 准确率: {val_accuracy:.4f}, 耗时: {valid_end_time - valid_start_time:.2f}s")

            # 使用专家对应的测试集测试当前专家的性能
            test_start_time = time.time()
            test_loss, test_accuracy = validate_expert(model, test_loader_expert, None, expert_idx, device)
            test_end_time = time.time()

            # 记录测试集指标
            test_metrics = {
                'loss': test_loss,
                'accuracy': test_accuracy
            }
            log_metrics_to_tensorboard(writer, test_metrics, epoch, prefix=f'expert{expert_idx}/test_')

            log_message(
                f"  专家{expert_idx} - 测试损失: {test_loss:.4f}, 准确率: {test_accuracy:.4f}, 耗时: {test_end_time - test_start_time:.2f}s")

            # 保存当前专家的最新检查点
            checkpoint_path = os.path.join(RESULTS_PATH, 'param', f'stage1_expert{expert_idx}_latest.pth')
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
                best_checkpoint_path = os.path.join(RESULTS_PATH, 'param', f'stage1_expert{expert_idx}_best.pth')
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
            if (epoch + 1) % 10 == 0 or (epoch + 1) == EPOCHS_STAGE1:
                epoch_checkpoint_path = os.path.join(RESULTS_PATH, 'param',
                                                     f'stage1_expert{expert_idx}_epoch{epoch + 1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'best_val_acc': best_val_acc
                }, epoch_checkpoint_path)
                log_message(f"  已保存专家{expert_idx}的Epoch {epoch + 1}检查点")

        # 绘制损失和准确率曲线
        loss_curve_path = os.path.join(RESULTS_PATH, 'logs', f'expert{expert_idx}_loss_curve.png')
        acc_curve_path = os.path.join(RESULTS_PATH, 'logs', f'expert{expert_idx}_accuracy_curve.png')
        plot_loss_curves(train_losses, val_losses, loss_curve_path)
        plot_accuracy_curves(train_accuracies, val_accuracies, acc_curve_path)
        log_message(f"  已保存专家{expert_idx}的损失和准确率曲线")

        # 关闭该专家的数据加载器
        del train_loader, val_loader_expert, test_loader_expert
        torch.cuda.empty_cache()
        gc.collect()
        # 该专家训练完成
        log_message(f"专家{expert_idx}训练完成，最佳验证准确率: {best_val_acc:.4f}")

    # 所有专家训练完成后，保存完整模型
    final_checkpoint_path = os.path.join(RESULTS_PATH, 'param', 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_ranges': model.class_ranges,
        'total_classes': model.total_classes
    }, final_checkpoint_path)
    log_message(f"所有专家训练完成，完整模型已保存到 {final_checkpoint_path}")
    writer.close()

    return model


# 验证专家函数，确保正确处理相对类别标签
def validate_expert(model, val_loader, criterion, expert_idx, device):
    """
    验证单个专家的性能
    注意：
    1. 专家的输入标签已经是相对于该专家负责类别范围的
    2. 按照Model design.md，损失函数包含分类损失和正则化项
    """
    model.eval()
    val_loss = 0
    cls_loss_sum = 0
    reg_loss_sum = 0
    correct = 0
    total = 0

    # 获取专家的类别范围
    if hasattr(model, 'module'):
        start_class, end_class = model.module.class_ranges[expert_idx]
    else:
        start_class, end_class = model.class_ranges[expert_idx]

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # 获取特征和专家输出
            if hasattr(model, 'module'):
                features = model.module.backbone(inputs)
                outputs = model.module.experts[expert_idx](features)

                # 计算损失 - 使用model内部的compute_loss方法
                cls_loss, reg_loss, total_loss = model.module.compute_loss(outputs, targets, expert_idx)
            else:
                features = model.backbone(inputs)
                outputs = model.experts[expert_idx](features)

                # 计算损失 - 使用model内部的compute_loss方法
                cls_loss, reg_loss, total_loss = model.compute_loss(outputs, targets, expert_idx)

            # 累积损失
            val_loss += total_loss.item()
            cls_loss_sum += cls_loss.item()
            reg_loss_sum += reg_loss.item()

            # 计算准确率
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # 计算平均损失和准确率
    avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
    avg_cls_loss = cls_loss_sum / len(val_loader) if len(val_loader) > 0 else 0
    avg_reg_loss = reg_loss_sum / len(val_loader) if len(val_loader) > 0 else 0
    accuracy = correct / total if total > 0 else 0

    return avg_val_loss, accuracy


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
    log_message(f"学习率: {LEARNING_RATE_STAGE1}, 训练轮数: {EPOCHS_STAGE1}")
    log_message(f"数据集路径: {DATASET_PATH}")
    log_message(f"结果保存路径: {RESULTS_PATH}")
    log_message(f"使用设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    log_message(f"自动混合精度: {USE_AMP}")
    log_message(f"工作进程数: {NUM_WORKERS}")
    log_message(f"专家数量: {NUM_EXPERTS}")
    # log_message(f"路由类型: {ROUTING_TYPE}")
    log_message(f"类别范围: {CLASS_RANGES}")
    log_message(f"预训练模型路径: {PRETRAINED_RESNET18_PATH}")

    # 配置数据并行训练，利用多个GPU
    multi_gpu = torch.cuda.device_count() > 1
    log_message(f"使用GPU数量: {torch.cuda.device_count()}")

    # 添加继续训练选项
    RESUME_TRAINING = True  # 设置是否从检查点继续训练

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

    # 训练专家
    model = train_stage1(model, train_loaders, val_loaders, test_loaders, device, resume_training=RESUME_TRAINING)

    # 在评估前预计算
    log_message("训练完成，预计算专家缩放因子...")
    if hasattr(model, 'module'):  # 处理DataParallel包装的模型
        model.module.precompute_scaling_factors()
    else:
        model.precompute_scaling_factors()

    # 保存带有预计算结果的完整模型
    log_message("保存带有预计算缩放因子的完整模型...")
    final_checkpoint_path = os.path.join(RESULTS_PATH, 'param', 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_ranges': model.class_ranges,
        'total_classes': model.total_classes,
        'has_cached_computation': True,
        'cached_scaling_factors': model.cached_scaling_factors if not hasattr(model, 'module') else model.module.cached_scaling_factors
    }, final_checkpoint_path)
    log_message(f"带有预计算缩放因子的完整模型已保存到 {final_checkpoint_path}")

    # 评估整体模型
    val_accuracy, test_accuracy = evaluate_ensemble_model(model, full_val_loader, full_test_loader, device, False)

    # 记录最终结果
    log_message(f"训练完成！")
    log_message(f"最终验证集准确率: {val_accuracy:.4f}")
    log_message(f"最终测试集准确率: {test_accuracy:.4f}")

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
