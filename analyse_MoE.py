import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from models.MoE import MoE4Model
from utils.data_loading_mine import load_data, get_dataloaders, RESULTS_PATH, NUM_CLASSES, log_message, CLASS_RANGES
import torch.nn as nn
import datetime

# 设置字体，使用通用字体以避免中文字体问题
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = True

# 分析配置
MODEL_PATH = f"{RESULTS_PATH}/param/final_model.pth"
ANALYSIS_RESULTS_PATH = f"{RESULTS_PATH}/analysis"
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#f1c40f', '#e67e22', '#95a5a6',
          '#d35400', '#8e44ad']  # 蓝色、绿色、红色、橙色、紫色、青色、深蓝色、黄色、橙红色、灰色、深橙色、深紫色

# 创建分析日志文件
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
ANALYSIS_LOG_FILE = f"{RESULTS_PATH}/analysis/analysis_log_{current_time}.txt"

# 确保分析结果目录存在
os.makedirs(ANALYSIS_RESULTS_PATH, exist_ok=True)


def load_model(model_path):
    """Load trained model"""
    log_message(f"Loading model: {model_path}", ANALYSIS_LOG_FILE)

    # 确保文件存在
    if not os.path.exists(model_path):
        log_message(f"Error: Model file {model_path} does not exist", ANALYSIS_LOG_FILE)
        return None

    # 获取输入样本的形状
    sample_x, _ = load_data("test")
    input_channels = sample_x.shape[1]
    input_height = sample_x.shape[2]
    input_width = sample_x.shape[3]

    # 创建模型实例
    model = MoE4Model(
        total_classes=NUM_CLASSES,
        class_ranges=CLASS_RANGES,
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width
    )

    # 加载保存的模型权重
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            log_message(f"Successfully loaded model weights", ANALYSIS_LOG_FILE)
        else:
            log_message(f"Warning: Checkpoint does not contain model_state_dict field", ANALYSIS_LOG_FILE)
            return None
    except Exception as e:
        log_message(f"Error loading model: {e}", ANALYSIS_LOG_FILE)
        return None

    return model


def analyze_model(model, test_loader, device):
    """Analyze model performance on test set"""
    log_message("Starting model analysis...", ANALYSIS_LOG_FILE)

    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    # 收集所有预测和真实标签
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # 使用模型预测
            _, logits = model.inference(inputs)
            _, predictions = torch.max(logits, 1)

            all_preds.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # 转换为numpy数组
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    return all_preds, all_targets


def calculate_per_class_accuracy(preds, targets, num_classes):
    """Calculate accuracy for each class"""
    log_message("Calculating per-class accuracy...", ANALYSIS_LOG_FILE)

    per_class_correct = np.zeros(num_classes)
    per_class_total = np.zeros(num_classes)

    for i in range(len(targets)):
        per_class_total[targets[i]] += 1
        if preds[i] == targets[i]:
            per_class_correct[targets[i]] += 1

    # 计算每个类别的准确率
    per_class_accuracy = np.zeros(num_classes)
    for i in range(num_classes):
        if per_class_total[i] > 0:
            per_class_accuracy[i] = per_class_correct[i] / per_class_total[i]

    return per_class_accuracy, per_class_correct, per_class_total


def plot_per_class_accuracy(per_class_accuracy, per_class_total, class_ranges):
    """Plot bar chart of per-class accuracy"""
    log_message("Plotting per-class accuracy bar chart...", ANALYSIS_LOG_FILE)

    plt.figure(figsize=(20, 8))

    # # 每个专家负责的类别用不同颜色表示
    # colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']  # 蓝色、绿色、红色、橙色

    for i, (start_class, end_class) in enumerate(class_ranges):
        class_range = list(range(start_class, end_class + 1))
        plt.bar(class_range,
                per_class_accuracy[start_class:end_class + 1],
                color=colors[i],
                label=f'Expert {i + 1} ({start_class}-{end_class})')

    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Per-class Classification Accuracy')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 保存图像
    plt.savefig(f"{ANALYSIS_RESULTS_PATH}/per_class_accuracy.png", dpi=300, bbox_inches='tight')

    # 另外绘制一个权重图，样本数量多的类别的柱子更宽
    plt.figure(figsize=(20, 8))

    max_samples = np.max(per_class_total)
    bar_width = 0.8  # 基础宽度

    for i in range(NUM_CLASSES):
        # 根据样本数量调整柱子宽度
        width = bar_width * (0.3 + 0.7 * per_class_total[i] / max_samples)

        # 确定颜色
        color_idx = 0
        for j, (start_class, end_class) in enumerate(class_ranges):
            if start_class <= i <= end_class:
                color_idx = j
                break

        plt.bar(i, per_class_accuracy[i], width=width, color=colors[color_idx], alpha=0.8)

    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Per-class Classification Accuracy (Bar width represents sample count)')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 添加专家范围的图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[i], label=f'Expert {i + 1} ({start}-{end})')
        for i, (start, end) in enumerate(class_ranges)
    ]
    plt.legend(handles=legend_elements)

    # 保存图像
    plt.savefig(f"{ANALYSIS_RESULTS_PATH}/per_class_accuracy_weighted.png", dpi=300, bbox_inches='tight')


def create_confusion_matrix(preds, targets, num_classes, class_ranges=CLASS_RANGES):
    """Create and visualize confusion matrix"""
    log_message("Creating confusion matrix...", ANALYSIS_LOG_FILE)

    # 计算混淆矩阵
    cm = confusion_matrix(targets, preds, labels=range(num_classes))

    # 计算归一化的混淆矩阵（按行归一化，表示每个真实类别的样本预测分布）
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # 处理可能的除零问题

    # 由于类别数量较多，可以尝试分块可视化混淆矩阵
    # 1. 首先保存完整的混淆矩阵
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm_normalized, cmap="YlGnBu", vmin=0, vmax=1)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Normalized Confusion Matrix (Full)')
    plt.savefig(f"{ANALYSIS_RESULTS_PATH}/confusion_matrix_full.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 然后按照专家负责的类别范围分块可视化
    for i, (start_class, end_class) in enumerate(class_ranges):
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm_normalized[start_class:end_class + 1, :], cmap="YlGnBu", vmin=0, vmax=0.5)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix for Expert {i + 1} Classes ({start_class}-{end_class})')
        plt.savefig(f"{ANALYSIS_RESULTS_PATH}/confusion_matrix_expert{i + 1}.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 保存原始混淆矩阵数据以便进一步分析
    np.save(f"{ANALYSIS_RESULTS_PATH}/confusion_matrix.npy", cm)
    np.save(f"{ANALYSIS_RESULTS_PATH}/confusion_matrix_normalized.npy", cm_normalized)

    return cm, cm_normalized


def analyze_misclassified(cm, cm_normalized, class_ranges):
    """Analyze misclassification patterns"""
    log_message("Analyzing misclassification patterns...", ANALYSIS_LOG_FILE)

    # 创建错分分析报告
    report = []

    # 1. 计算每个类别的错分率及主要错分去向
    for c in range(NUM_CLASSES):
        # 找到哪个专家负责这个类别
        responsible_expert = -1
        for i, (start, end) in enumerate(class_ranges):
            if start <= c <= end:
                responsible_expert = i
                break

        # 计算错误率
        error_rate = 1.0 - cm_normalized[c, c]

        # 找出前3个最常错分的类别
        other_indices = [i for i in range(NUM_CLASSES) if i != c]
        top_confused = sorted(other_indices, key=lambda i: cm_normalized[c, i], reverse=True)[:3]
        top_confused_str = ", ".join([f"Class {i} ({cm_normalized[c, i]:.2%})" for i in top_confused[:3]])

        report.append({
            "Class": c,
            "Responsible Expert": responsible_expert + 1,  # 专家编号从1开始
            "Total Samples": cm[c].sum(),
            "Correct Classifications": cm[c, c],
            "Accuracy": cm_normalized[c, c],
            "Error Rate": error_rate,
            "Top Misclassified As": top_confused_str
        })

    # 将报告转换为DataFrame
    df_report = pd.DataFrame(report)

    # 按错误率排序
    df_report = df_report.sort_values("Error Rate", ascending=False)

    # 保存到CSV
    df_report.to_csv(f"{ANALYSIS_RESULTS_PATH}/misclassification_report.csv", index=False)

    # 2. 绘制按专家分组的错误率柱状图
    fig, ax = plt.subplots(figsize=(12, 8))

    # 设置不同专家的颜色
    # colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']  # 蓝色、绿色、红色、橙色

    # 按专家分组并绘制
    for i, (start, end) in enumerate(class_ranges):
        expert_classes = range(start, end + 1)
        error_rates = [1.0 - cm_normalized[c, c] for c in expert_classes]
        ax.bar(expert_classes, error_rates, color=colors[i],
               label=f'Expert {i + 1} ({start}-{end})')

    ax.set_xlabel('Class')
    ax.set_ylabel('Error Rate')
    ax.set_title('Classification Error Rate by Class')
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig(f"{ANALYSIS_RESULTS_PATH}/error_rates_by_class.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 计算每个专家的平均准确率和最差表现类别
    expert_summary = []
    for i, (start, end) in enumerate(class_ranges):
        expert_classes = list(range(start, end + 1))
        accuracies = [cm_normalized[c, c] for c in expert_classes]

        # 找出表现最差的5个类别
        worst_classes = sorted(zip(expert_classes, accuracies), key=lambda x: x[1])[:5]
        worst_classes_str = ", ".join([f"Class {c} ({acc:.2%})" for c, acc in worst_classes])

        expert_summary.append({
            "Expert": i + 1,
            "Class Range": f"{start}-{end}",
            "Number of Classes": len(expert_classes),
            "Average Accuracy": np.mean(accuracies),
            "Max Accuracy": np.max(accuracies),
            "Min Accuracy": np.min(accuracies),
            "Worst Performing Classes": worst_classes_str
        })

    # 将专家汇总转换为DataFrame
    df_expert = pd.DataFrame(expert_summary)

    # 保存到CSV
    df_expert.to_csv(f"{ANALYSIS_RESULTS_PATH}/expert_performance_summary.csv", index=False)

    # 4. 分析专家间的错误"流动"
    expert_flow = np.zeros((len(class_ranges), len(class_ranges)))

    for true_class in range(NUM_CLASSES):
        # 找出真实类别的负责专家
        true_expert = -1
        for i, (start, end) in enumerate(class_ranges):
            if start <= true_class <= end:
                true_expert = i
                break

        for pred_class in range(NUM_CLASSES):
            if true_class == pred_class:
                continue  # 跳过正确分类

            # 找出预测类别的负责专家
            pred_expert = -1
            for i, (start, end) in enumerate(class_ranges):
                if start <= pred_class <= end:
                    pred_expert = i
                    break

            # 累加从true_expert到pred_expert的错误流动
            expert_flow[true_expert, pred_expert] += cm[true_class, pred_class]

    # 归一化专家流动矩阵（按行）
    expert_flow_norm = expert_flow / expert_flow.sum(axis=1)[:, np.newaxis]
    expert_flow_norm = np.nan_to_num(expert_flow_norm)

    # 可视化专家间的错误流动
    plt.figure(figsize=(10, 8))
    sns.heatmap(expert_flow_norm, annot=True, fmt=".2%", cmap="YlGnBu",
                xticklabels=[f"Expert {i + 1}" for i in range(len(class_ranges))],
                yticklabels=[f"Expert {i + 1}" for i in range(len(class_ranges))])
    plt.xlabel('Predicted Expert')
    plt.ylabel('True Expert')
    plt.title('Error Flow Between Experts (Normalized)')
    plt.savefig(f"{ANALYSIS_RESULTS_PATH}/expert_error_flow.png", dpi=300, bbox_inches='tight')
    plt.close()

    return df_report, df_expert


def main():
    log_message("======= MoE Model Analysis =======", ANALYSIS_LOG_FILE)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_message(f"Using device: {device}", ANALYSIS_LOG_FILE)

    # 加载模型
    model = load_model(MODEL_PATH)
    if model is None:
        log_message("Model loading failed, exiting analysis", ANALYSIS_LOG_FILE)
        return
    CLASS_RANGES = model.class_ranges  # 获取模型的类别范围
    # 加载测试数据
    _, _, _, _, _, test_loader = get_dataloaders(CLASS_RANGES)

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
    log_message(f"Overall accuracy: {overall_accuracy:.4%}", ANALYSIS_LOG_FILE)

    # 打印专家性能汇总
    log_message("\nExpert Performance Summary:", ANALYSIS_LOG_FILE)
    log_message(expert_report.to_string(), ANALYSIS_LOG_FILE)

    # 打印前10个错误率最高的类别
    log_message("\nTop 10 classes with highest error rates:", ANALYSIS_LOG_FILE)
    log_message(class_report.head(10).to_string(), ANALYSIS_LOG_FILE)

    log_message(f"\nAnalysis results saved to: {ANALYSIS_RESULTS_PATH}", ANALYSIS_LOG_FILE)


if __name__ == "__main__":
    main()
