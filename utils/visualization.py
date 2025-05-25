import matplotlib

try:
    # 设置为非交互式后端，避免tostring_rgb错误
    matplotlib.use('Agg')  # 改为Agg后端，这是一个非交互式后端，适合保存图表到文件
except ImportError:
    print("警告: 尝试设置 Matplotlib 后端为 'Agg' 失败。")

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.tensorboard import SummaryWriter


def plot_expert_logits_histograms(expert_outputs, RESULTS_PATH):
    """
    为每个样本绘制每个专家的logits输出，横轴为类别索引，纵轴为logits值。

    Args:
        expert_outputs: 列表，包含每个专家的输出logits张量。
                        例如: [expert1_logits_tensor, expert2_logits_tensor, ...],
                        其中 expertN_logits_tensor 的形状为 (batch_size, num_classes_for_expertN).
        RESULTS_PATH: 结果保存路径
    """
    if not expert_outputs:
        print("No expert outputs to plot.")
        return

    num_experts = len(expert_outputs)
    if num_experts == 0:
        print("Zero experts provided.")
        return

    # 确保输出目录存在
    save_dir = os.path.join(RESULTS_PATH, "expert_logits_plots")
    os.makedirs(save_dir, exist_ok=True)

    # 假设所有专家的批处理大小相同
    batch_size = expert_outputs[0].shape[0]

    if batch_size == 0:
        print("Batch size is zero, nothing to plot.")
        return

    # 为了避免创建太多图表，最多只处理前5个样本
    max_samples_to_plot = min(5, batch_size)

    for sample_idx in range(max_samples_to_plot):
        # 为每个样本创建一个新的图形，包含num_experts个子图
        fig, axes = plt.subplots(nrows=1, ncols=num_experts,
                                 figsize=(6 * num_experts, 5), squeeze=False)

        fig.suptitle(f"Logits Distribution for Sample {sample_idx + 1}", fontsize=16)

        for expert_idx in range(num_experts):
            # 获取当前样本、当前专家的logits
            current_expert_logits_for_sample = expert_outputs[expert_idx][sample_idx]

            # 从计算图中分离，移至CPU，并转换为NumPy数组
            logits_np = current_expert_logits_for_sample.detach().cpu().numpy()

            ax = axes[0, expert_idx]  # 获取子图对象

            # 获取类别数量
            num_classes = logits_np.shape[0]
            class_indices = np.arange(num_classes)

            # 找出最大值及其索引
            max_val = np.max(logits_np)
            max_idx = np.argmax(logits_np)

            # 绘制折线图，横轴为类别索引，纵轴为logits值
            ax.plot(class_indices, logits_np, marker='o', linestyle='-', markersize=4, color=f'C{expert_idx}',
                    alpha=0.7)

            # 特别标记出最大值点
            ax.plot(max_idx, max_val, marker='*', markersize=15, color='red',
                    label=f'Max: {max_val:.4f} at class {max_idx}')

            # 添加网格线以提高可读性
            ax.grid(True, linestyle='--', alpha=0.7)

            # 设置标题和轴标签
            ax.set_title(f"Expert {expert_idx + 1}", fontsize=14)
            ax.set_xlabel("Class Index", fontsize=12)
            ax.set_ylabel("Logit Value", fontsize=12)

            # 微调x轴，使其更清晰
            if num_classes <= 50:  # 如果类别数量较少，显示所有刻度
                ax.set_xticks(class_indices)
            else:  # 否则只显示部分刻度以避免拥挤
                step = max(1, num_classes // 10)
                ax.set_xticks(class_indices[::step])

            # 添加水平参考线，表示零点
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.3)

            # 添加一些统计信息到子图，包括最大值的索引
            mean_val = np.mean(logits_np)
            std_val = np.std(logits_np)
            min_val = np.min(logits_np)

            stats_text = f"Mean: {mean_val:.4f}\nStd: {std_val:.4f}\nMax: {max_val:.4f} (idx: {max_idx})\nMin: {min_val:.4f}"
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # 添加图例
            ax.legend(loc='upper left')

        # 调整布局以防止标题和标签重叠
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # rect=[left, bottom, right, top]

        # 保存图形到文件而不是显示
        save_path = os.path.join(save_dir, f"sample_{sample_idx + 1}_logits_plot.png")
        plt.savefig(save_path, dpi=100)
        plt.close(fig)  # 关闭图形以释放内存

        print(f"已保存样本 {sample_idx + 1} 的logits折线图到 {save_path}")


def plot_loss_curves(train_losses, val_losses, save_path):
    """
    绘制训练和验证损失曲线
    
    Args:
        train_losses: 训练损失历史
        val_losses: 验证损失历史
        save_path: 保存图像的路径
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_accuracy_curves(train_accuracies, val_accuracies, save_path):
    """
    绘制训练和验证准确率曲线
    
    Args:
        train_accuracies: 训练准确率历史
        val_accuracies: 验证准确率历史
        save_path: 保存图像的路径
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def log_metrics_to_tensorboard(writer, metrics, epoch, prefix=''):
    """
    将指标记录到TensorBoard
    
    Args:
        writer: TensorBoard的SummaryWriter对象
        metrics: 指标字典，例如 {'loss': 0.5, 'accuracy': 0.8}
        epoch: 当前的epoch
        prefix: 指标名称的前缀，例如 'train/' 或 'val/'
    """
    for name, value in metrics.items():
        if isinstance(value, torch.Tensor):
            value = value.item()
        writer.add_scalar(f'{prefix}{name}', value, epoch)


def visualize_expert_assignments(model, val_loader, device, save_path):
    """
    可视化专家分配情况
    
    Args:
        model: 模型对象
        val_loader: 验证数据加载器
        device: 计算设备
        save_path: 保存图像的路径
    """
    model.eval()

    # 收集每个类别被分配到每个专家的样本数量
    class_to_expert_count = {}

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # 确定每个样本应该由哪个专家处理
            target_classes = targets.cpu().numpy()

            # 对于每个类别，确定其负责的专家
            for class_idx in target_classes:
                class_idx = int(class_idx)
                if class_idx not in class_to_expert_count:
                    class_to_expert_count[class_idx] = {}

                # 找出负责该类别的专家
                for expert_idx, (start, end) in enumerate(model.class_ranges):
                    if start <= class_idx <= end:
                        if expert_idx not in class_to_expert_count[class_idx]:
                            class_to_expert_count[class_idx][expert_idx] = 0
                        class_to_expert_count[class_idx][expert_idx] += 1

    # 创建可视化
    plt.figure(figsize=(12, 8))

    # 对于每个类别，绘制其被分配到各个专家的比例
    classes = sorted(class_to_expert_count.keys())
    x = np.arange(len(classes))
    width = 0.8 / model.num_experts  # 柱状图宽度

    # 绘制每个专家的分配情况
    for expert_idx in range(model.num_experts):
        expert_vals = []
        for class_idx in classes:
            total_samples = sum(class_to_expert_count[class_idx].values())
            expert_count = class_to_expert_count[class_idx].get(expert_idx, 0)
            expert_ratio = expert_count / total_samples if total_samples > 0 else 0
            expert_vals.append(expert_ratio)

        plt.bar(x + expert_idx * width, expert_vals, width, label=f'Expert {expert_idx}')

    plt.xlabel('Class')
    plt.ylabel('Expert Assignment Ratio')
    plt.title('Class to Expert Assignment')
    plt.xticks(x + width * model.num_experts / 2 - width / 2, classes)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
