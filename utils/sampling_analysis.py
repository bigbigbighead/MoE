import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
import logging
import argparse
from tqdm import tqdm
import time
from typing import Dict, List, Tuple, Optional, Union, Any

# 设置matplotlib中文支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 添加项目根目录到系统路径，以便导入models包
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入模型定义
from models.ResNet import resnet18

# 导入自定义的重采样函数
from data_loading_downsample import load_data

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sampling_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

# 指定两个模型的路径（全局变量）
BASELINE_MODEL_PATH = "./results/AppClassNet/top200/ResNet/1/param/best_model_81.94.pth"
SAMPLED_MODEL_PATH = "./results/AppClassNet/top200/ResNet/3/param/best_model_83.21.pth"
SAMPLING_STRATEGY = "tiered_importance"  # 采样模型使用的策略名称

# 定义常量
DEFAULT_SAVE_DIR = "./results/sampling_analysis"
DEFAULT_DATASET_PATH = "./data/AppClassNet/top200"

# 定义类别层级
TIER_COLORS = {
    (0, 49): ('blue', '前50类'),
    (50, 99): ('green', '中前50类'),
    (100, 149): ('orange', '中后50类'),
    (150, 199): ('red', '后50类')
}


def setup_directories(*dirs) -> None:
    """创建多个目录"""
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"确保目录存在: {dir_path}")


def evaluate_model(model: torch.nn.Module,
                  dataloader: torch.utils.data.DataLoader,
                  device: torch.device,
                  num_classes: int = 200) -> Dict[str, Any]:
    """
    评估模型在给定数据集上的性能

    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        num_classes: 类别数量

    Returns:
        包含评估结果的字典
    """
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="评估模型"):
            # 将数据移至��备
            inputs, labels = inputs.to(device), labels.to(device)

            # 模型推理
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # 移回CPU以进行评估
            predicted_cpu = predicted.cpu().numpy()
            labels_cpu = labels.cpu().numpy()

            # 记录总体统计信息
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 收集所有预测和标签用于后续分析
            all_preds.extend(predicted_cpu)
            all_labels.extend(labels_cpu)

            # 记录每个类别的准确率
            for i in range(len(labels_cpu)):
                label = labels_cpu[i]
                class_total[label] += 1
                if predicted_cpu[i] == label:
                    class_correct[label] += 1

    # 计算总体准确率
    accuracy = 100 * correct / total

    # 计算每个类别的准确率
    class_accuracies = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            class_accuracies[i] = 100 * class_correct[i] / class_total[i]
        else:
            class_accuracies[i] = 0.0

    # 计算每个层级的准确率
    tier_accuracies = {}
    for (start, end), (_, label) in TIER_COLORS.items():
        tier_classes = [c for c in range(start, end + 1) if class_total[c] > 0]
        if tier_classes:
            tier_correct = sum(class_correct[c] for c in tier_classes)
            tier_total = sum(class_total[c] for c in tier_classes)
            tier_accuracies[label] = 100 * tier_correct / tier_total
        else:
            tier_accuracies[label] = 0.0

    # 组织返回结果
    results = {
        'accuracy': accuracy,
        'class_accuracies': class_accuracies,
        'tier_accuracies': tier_accuracies,
        'class_totals': {i: class_total[i] for i in range(num_classes) if class_total[i] > 0}
    }

    return results


def load_model(model_path: str,
              num_classes: int = 200,
              device: Union[str, torch.device] = 'cuda') -> Optional[torch.nn.Module]:
    """
    加载模型权重

    Args:
        model_path: 模型路径
        num_classes: 类别数量
        device: 设备

    Returns:
        加载了权重的模型，如果加载失败则返回None
    """
    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')

    model = resnet18(num_classes=num_classes)

    try:
        checkpoint = torch.load(model_path, map_location=device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"加载模型权重成功 (使用model_state_dict): {model_path}")
        else:
            model.load_state_dict(checkpoint)
            logger.info(f"加载模型权重成功: {model_path}")

        return model.to(device)

    except Exception as e:
        logger.error(f"加载模型权重失败: {str(e)}")
        return None


def prepare_test_data(dataset_path: str,
                     num_classes: int = 200,
                     batch_size: int = 512) -> torch.utils.data.DataLoader:
    """
    准备测试数据

    Args:
        dataset_path: 数据集路径
        num_classes: 类别数量
        batch_size: 批次大小

    Returns:
        测试数据的DataLoader
    """
    # 加载测试数据
    logger.info(f"加载测试数据: {dataset_path}")
    test_x, test_y = load_data("test", dataset_path=dataset_path)
    original_size = len(test_y)

    # 筛选类别（如果需要）
    if num_classes < 200:
        mask = test_y < num_classes
        test_x = test_x[mask]
        test_y = test_y[mask]
        logger.info(f"筛选前{num_classes}类: {original_size} -> {len(test_y)} 样本")

    # 创建数据集和加载器
    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(8, os.cpu_count() or 4),
        pin_memory=True
    )

    logger.info(f"测试数据准备完成: {len(test_dataset)} 样本, {len(test_loader)} 批次")
    return test_loader


def plot_accuracy_comparison(baseline_results: Dict[str, Any],
                           sampled_results: Dict[str, Any],
                           strategy_name: str,
                           save_path: Optional[str] = None,
                           show_plot: bool = True) -> None:
    """
    绘制两个模型的准确率对比图

    Args:
        baseline_results: 基准模型的评估结果
        sampled_results: 采样模型的评估结果
        strategy_name: 采样策略名称
        save_path: 图表保存路径
        show_plot: 是否显示图表
    """
    # 1. 总体准确率对比
    plt.figure(figsize=(10, 6))
    models = ['基准模型', f'{strategy_name}采样模型']
    accuracies = [baseline_results['accuracy'], sampled_results['accuracy']]

    bars = plt.bar(models, accuracies, color=['blue', 'green'])

    # 添���数值标签
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=12)

    # 计算准确率差异
    acc_diff = sampled_results['accuracy'] - baseline_results['accuracy']
    diff_color = 'green' if acc_diff > 0 else 'red'
    plt.title(f"模型总体准确率对比 (差异: {acc_diff:+.2f}%)", fontsize=14)
    plt.ylabel("准确率 (%)", fontsize=12)
    plt.ylim(min(accuracies) - 5, max(accuracies) + 5)
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    # 保存图表
    if save_path:
        plt.savefig(f"{save_path}_overall.png", dpi=300, bbox_inches='tight')
        logger.info(f"总体准确率对比图已保存到 {save_path}_overall.png")

    if show_plot:
        plt.show()
    else:
        plt.close()

    # 2. 类别层级准确率对比
    plt.figure(figsize=(12, 7))

    # 获取层级名称和准确率
    tier_names = list(baseline_results['tier_accuracies'].keys())
    baseline_tier_accs = [baseline_results['tier_accuracies'][tier] for tier in tier_names]
    sampled_tier_accs = [sampled_results['tier_accuracies'][tier] for tier in tier_names]

    # 计算差异
    diff_tier_accs = [s - b for s, b in zip(sampled_tier_accs, baseline_tier_accs)]

    # 设置x位置
    x = np.arange(len(tier_names))
    width = 0.35

    # 绘制层级准确率对比图
    plt.bar(x - width/2, baseline_tier_accs, width, label='基准模型', color='blue', alpha=0.7)
    plt.bar(x + width/2, sampled_tier_accs, width, label=f'{strategy_name}采样模型', color='green', alpha=0.7)

    # 添加数值标签
    for i, (b_acc, s_acc) in enumerate(zip(baseline_tier_accs, sampled_tier_accs)):
        plt.text(i - width/2, b_acc + 1, f'{b_acc:.1f}%', ha='center', va='bottom', fontsize=10)
        plt.text(i + width/2, s_acc + 1, f'{s_acc:.1f}%', ha='center', va='bottom', fontsize=10)

    # 添加差异箭头
    for i, diff in enumerate(diff_tier_accs):
        color = 'green' if diff > 0 else 'red'
        plt.annotate(f"{diff:+.1f}%",
                    xy=(i, max(baseline_tier_accs[i], sampled_tier_accs[i]) + 3),
                    xytext=(i, max(baseline_tier_accs[i], sampled_tier_accs[i]) + 6),
                    arrowprops=dict(facecolor=color, shrink=0.05, width=1.5),
                    ha='center', va='bottom', color=color, fontsize=10)

    plt.xlabel("类别层级", fontsize=12)
    plt.ylabel("准确率 (%)", fontsize=12)
    plt.title(f"各类别层级准确率对比", fontsize=14)
    plt.xticks(x, tier_names)
    plt.legend(loc='upper right')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    # 保存图表
    if save_path:
        plt.savefig(f"{save_path}_tiers.png", dpi=300, bbox_inches='tight')
        logger.info(f"层级准确率对比图已保存到 {save_path}_tiers.png")

    if show_plot:
        plt.show()
    else:
        plt.close()

    # 3. 每个类别准确率差异
    plt.figure(figsize=(20, 8))

    # 获取所有类别ID
    class_ids = sorted(set(baseline_results['class_accuracies'].keys()) |
                      set(sampled_results['class_accuracies'].keys()))

    # 计算每个类别的准确率差异
    acc_diffs = []
    for class_id in class_ids:
        base_acc = baseline_results['class_accuracies'].get(class_id, 0)
        samp_acc = sampled_results['class_accuracies'].get(class_id, 0)
        acc_diffs.append(samp_acc - base_acc)

    # 使用颜色区分正负差异
    colors = ['green' if diff >= 0 else 'red' for diff in acc_diffs]

    # 绘制条形图
    plt.bar(class_ids, acc_diffs, color=colors, alpha=0.7)

    # 添加水平参考线（0差异线）
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # 为不同层级添加背景色
    for (start, end), (color, label) in TIER_COLORS.items():
        plt.axvspan(start - 0.5, end + 0.5, alpha=0.1, color=color, label=label if start == 0 else None)
        # 添加层级分隔线
        if start > 0:
            plt.axvline(x=start - 0.5, color='black', linestyle='--', alpha=0.3)

    # 计算每个层级的平均差异
    tier_diff_avgs = {}
    for (start, end), (_, label) in TIER_COLORS.items():
        tier_diffs = [acc_diffs[class_ids.index(c)] for c in class_ids if start <= c <= end]
        if tier_diffs:
            tier_diff_avgs[label] = np.mean(tier_diffs)

            # 在每个层级中间显示平均差异
            mid_point = (start + end) / 2
            plt.text(mid_point, max(acc_diffs) - 5 if max(acc_diffs) > 0 else min(acc_diffs) + 5,
                    f"平均: {tier_diff_avgs[label]:+.2f}%",
                    ha='center', va='center', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))

    plt.xlabel("类别ID", fontsize=12)
    plt.ylabel("准确率差异 (百分点)", fontsize=12)
    plt.title(f"{strategy_name}采样模型 vs 基准模型: 各类别准确率差异", fontsize=14)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.legend(loc='upper right')

    # 添加全局平均差异
    avg_diff = np.mean(acc_diffs)
    plt.text(0.02, 0.95, f"总体平均差异: {avg_diff:+.2f}%",
            transform=plt.gca().transAxes, fontsize=12,
            bbox=dict(facecolor='yellow' if avg_diff > 0 else 'lightcoral', alpha=0.8, boxstyle='round,pad=0.5'))

    plt.tight_layout()

    # 保存图表
    if save_path:
        plt.savefig(f"{save_path}_class_diffs.png", dpi=300, bbox_inches='tight')
        logger.info(f"类别准确率差异图已保存到 {save_path}_class_diffs.png")

    if show_plot:
        plt.show()
    else:
        plt.close()

    # 4. 散点图 - 比较两个模型在每个类别上的准确率
    plt.figure(figsize=(10, 10))

    # 准备数据
    x_accs = []
    y_accs = []
    sizes = []  # 点的大小将基于样本数量

    for class_id in class_ids:
        if class_id in baseline_results['class_accuracies'] and class_id in sampled_results['class_accuracies']:
            x_accs.append(baseline_results['class_accuracies'][class_id])
            y_accs.append(sampled_results['class_accuracies'][class_id])
            # 样本数量决定点的大小
            sample_count = baseline_results['class_totals'].get(class_id, 0)
            sizes.append(np.sqrt(sample_count) * 0.5)  # 调整以便可视化

    # 为不同层级的类别设置不同颜色
    colors = []
    for class_id in class_ids:
        if class_id in baseline_results['class_accuracies'] and class_id in sampled_results['class_accuracies']:
            for (start, end), (color, _) in TIER_COLORS.items():
                if start <= class_id <= end:
                    colors.append(color)
                    break
            else:
                colors.append('gray')

    # 绘制散���图
    plt.scatter(x_accs, y_accs, c=colors, s=sizes, alpha=0.7)

    # 添加对角线（y=x线）
    max_val = max(max(x_accs), max(y_accs))
    min_val = min(min(x_accs), min(y_accs))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

    # 添加图例
    for (_, end), (color, label) in TIER_COLORS.items():
        plt.scatter([], [], c=color, s=50, label=label)

    # 添加统计信息
    info_text = f"类别数: {len(x_accs)}\n"
    info_text += f"采样模型更好: {above_diagonal} 类 ({above_diagonal/len(x_accs):.1%})\n"
    info_text += f"基准模型更好: {below_diagonal} 类 ({below_diagonal/len(x_accs):.1%})\n"
    info_text += f"性能相同: {on_diagonal} 类 ({on_diagonal/len(x_accs):.1%})"

    plt.text(0.02, 0.95, info_text, transform=plt.gca().transAxes, fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    plt.xlabel("基准模型准确率 (%)", fontsize=12)
    plt.ylabel(f"{strategy_name}采样模型准确率 (%)", fontsize=12)
    plt.title("两个模型在各类别上的准确率对比", fontsize=14)
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend(loc='lower right')

    # 保存图表
    if save_path:
        plt.savefig(f"{save_path}_scatter.png", dpi=300, bbox_inches='tight')
        logger.info(f"准确率散点图已保存到 {save_path}_scatter.png")

    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    """主函数：执行模型对比分析"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='模型性能对比分析工具')
    parser.add_argument('--baseline', type=str, default=BASELINE_MODEL_PATH,
                        help='基准模型路径')
    parser.add_argument('--sampled', type=str, default=SAMPLED_MODEL_PATH,
                        help='采样模型路径')
    parser.add_argument('--strategy', type=str, default=SAMPLING_STRATEGY,
                        help='采样策略名称')
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET_PATH,
                        help='数据集路径')
    parser.add_argument('--save-dir', type=str, default=DEFAULT_SAVE_DIR,
                        help='保存结果的目录')
    parser.add_argument('--no-plots', action='store_true',
                        help='不显示图表（仅保存）')
    parser.add_argument('--num-classes', type=int, default=200,
                        help='类别数量')
    args = parser.parse_args()

    logger.info("====== 开始模型性能对比分析 ======")
    logger.info(f"基准模型路径: {args.baseline}")
    logger.info(f"采样模型路径: {args.sampled}")
    logger.info(f"采样策略: {args.strategy}")
    logger.info(f"数据集路径: {args.dataset}")

    # 创建保存目录
    setup_directories(args.save_dir)

    try:
        # 准备测试数据
        test_loader = prepare_test_data(
            dataset_path=args.dataset,
            num_classes=args.num_classes,
            batch_size=512
        )

        # 确定运行设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {device}")

        # 加载并评估基准模型
        logger.info("开始评估基准模型...")
        baseline_model = load_model(args.baseline, num_classes=args.num_classes, device=device)
        if baseline_model is None:
            logger.error("基准模型加载失败，无法继续比较")
            return

        baseline_results = evaluate_model(baseline_model, test_loader, device, num_classes=args.num_classes)
        logger.info(f"基准模型评估完成，总体准确率: {baseline_results['accuracy']:.2f}%")

        # 加载并评估采样模型
        logger.info("开始评估采样模型...")
        sampled_model = load_model(args.sampled, num_classes=args.num_classes, device=device)
        if sampled_model is None:
            logger.error("采样模型加载失败，无法继续比较")
            return

        sampled_results = evaluate_model(sampled_model, test_loader, device, num_classes=args.num_classes)
        logger.info(f"采样模型评估完成，总体准确率: {sampled_results['accuracy']:.2f}%")

        # 释放模型内存
        del baseline_model
        del sampled_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 计算准确率差异
        acc_diff = sampled_results['accuracy'] - baseline_results['accuracy']
        logger.info(f"采样模型相比基准模型准确率差异: {acc_diff:+.2f}%")

        # 计算层级准确率差异
        logger.info("各类别层级准确率对比:")
        for tier_name in baseline_results['tier_accuracies']:
            base_acc = baseline_results['tier_accuracies'][tier_name]
            samp_acc = sampled_results['tier_accuracies'][tier_name]
            diff = samp_acc - base_acc
            logger.info(f"  {tier_name}: 基准 {base_acc:.2f}%, 采样 {samp_acc:.2f}%, 差异 {diff:+.2f}%")

        # 绘制对比图表
        save_path = os.path.join(args.save_dir, f"model_comparison_{args.strategy}")
        plot_accuracy_comparison(
            baseline_results=baseline_results,
            sampled_results=sampled_results,
            strategy_name=args.strategy,
            save_path=save_path,
            show_plot=not args.no_plots
        )

        logger.info(f"模型对比分析完成！图表已保存到 {args.save_dir} 目录")

    except Exception as e:
        logger.error(f"分析过程中发生错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

    logger.info("====== 分析过程结束 ======")


if __name__ == "__main__":
    main()
