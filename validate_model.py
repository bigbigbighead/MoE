import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.loss_functions import compute_specialized_loss, combine_expert_outputs
from utils.data_loading_mine import log_message
from utils.visualization import plot_expert_logits_histograms


# 计算专家分类器最后一层权重的L2范数
def calculate_expert_weight_norms(model):
    """
    计算每个专家分类器最后一层权重的L2范数

    Args:
        model: MoE模型

    Returns:
        list: 每个专家最后一层权重的L2范数
    """
    expert_weights = []

    # 处理可能的DataParallel封装
    if hasattr(model, 'module'):
        experts = model.module.experts
    else:
        experts = model.experts

    # 计算每个专家的权重平方范数
    for i, expert in enumerate(experts):
        weight = expert.get_last_layer_weights()
        if weight is not None:
            # 计算权重的平方范数 ||w||²
            tao = 1
            weight_norm_squared = torch.norm(weight, p=2, dim=1).pow(tao).mean()
            log_message(f"专家 {i} 的{tao}次幂权重平方范数: {weight_norm_squared.item():.4f}")
            expert_weights.append(weight_norm_squared)
        else:
            # 如果无法获取权重，则使用默认值1.0
            log_message(f"专家 {i} 无法获取权重，使用默认值1.0")
            expert_weights.append(torch.tensor(1.0, device=weight.device if weight is not None else 'cpu'))

    return expert_weights


# 验证单个专家的性能
def validate_expert(model, val_loader, criterion, expert_idx, device):
    """
    验证单个专家的性能

    按照Model design.md:
    1. 每个专家有自己的目标类范围
    2. 损失函数为分类损失和干扰类别正则项之和
    3. 专家只处理属于自己目标类的样本

    注意：val_loader中的标签已经是相对于专家负责类别范围的
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
        responsible_classes = set(range(start_class, end_class + 1))
        total_classes = model.module.total_classes
    else:
        start_class, end_class = model.class_ranges[expert_idx]
        responsible_classes = set(range(start_class, end_class + 1))
        total_classes = model.total_classes

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if hasattr(model, 'module'):
                features = model.module.backbone(inputs)
                outputs = model.module.experts[expert_idx](features)

                # 使用模型内部的损失计算函数
                cls_loss, reg_loss, total_loss = model.module.compute_loss(outputs, targets, expert_idx)
            else:
                features = model.backbone(inputs)
                outputs = model.experts[expert_idx](features)

                # 使用模型内部的损失计算函数
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
    avg_val_loss = val_loss / len(val_loader)
    avg_cls_loss = cls_loss_sum / len(val_loader)
    avg_reg_loss = reg_loss_sum / len(val_loader)
    accuracy = correct / total

    return avg_val_loss, accuracy


# 验证完整模型的性能
def validate_full_model(model, val_loader, criterion, device, result_path=None, FLAG=True, alpha1=0.0, alpha2=0.0,
                        output_stats=True, output_weight_norms=True):
    """
    验证完整模型的性能

    Args:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 计算设备
        result_path: 结果保存路径
        FLAG: 是否打印详细信息和保存可视化结果
        alpha1: 专家1的权重系数
        alpha2: 专家2的权重系数
        output_stats: 是否输出专家logits统计信息
        output_weight_norms: 是否输出专家权重范数

    Returns:
        tuple: (平均损失, 准确率)
    """
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    batch_count = 0

    # 预测结果和真实标签存储
    all_preds = []
    all_targets = []

    # 为每个专家初始化统计信息收集器 - 使用列表存储而非嵌套字典，提高效率
    num_experts = len(model.experts) if not hasattr(model, 'module') else len(model.module.experts)
    expert_means = [[] for _ in range(num_experts)]
    expert_stds = [[] for _ in range(num_experts)]
    expert_maxs = [[] for _ in range(num_experts)]
    expert_mins = [[] for _ in range(num_experts)]
    expert_all_logits = [[] for _ in range(num_experts)]

    # 批处理预测和统计
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            batch_count += 1
            inputs, targets = inputs.to(device), targets.to(device)

            # 使用指定的α1和α2进行推理
            if hasattr(model, 'module'):
                expert_outputs, combined_outputs = model.module.inference(inputs, alpha1, alpha2)
            else:
                expert_outputs, combined_outputs = model.inference(inputs, alpha1, alpha2)

            # 高效统计每个专家logits的信息
            if output_stats:
                for i, logits in enumerate(expert_outputs):
                    # 转移到CPU并转换为NumPy一次性完成
                    logits_np = logits.detach().cpu().numpy()

                    # 批量计算统计值
                    mean_val = np.mean(logits_np)
                    std_val = np.std(logits_np)
                    max_val = np.max(logits_np)
                    min_val = np.min(logits_np)

                    # 收集统计数据
                    expert_means[i].append(mean_val)
                    expert_stds[i].append(std_val)
                    expert_maxs[i].append(max_val)
                    expert_mins[i].append(min_val)

                    # 只收集需要的数据，避免内存占用过大
                    if FLAG and result_path is not None:
                        # 对于大批量，可以考虑采样而不是存储所有值
                        if logits_np.size > 100000:
                            indices = np.random.choice(logits_np.size, 10000, replace=False)
                            expert_all_logits[i].extend(logits_np.flatten()[indices])
                        else:
                            expert_all_logits[i].extend(logits_np.flatten())

            # 计算损失
            loss = criterion(combined_outputs, targets)
            val_loss += loss.item()

            # 计算准确率
            _, predicted = combined_outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 收集预测结果和真实标签（如果需要绘制混淆矩阵）
            if FLAG and result_path is not None:
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

    # 计算平均损失和准确率
    avg_loss = val_loss / batch_count
    accuracy = correct / total

    # 输出总体统计信息
    if output_stats:
        log_message("\n=== 专家 Logits 总体统计信息 ===")
        for i in range(num_experts):
            overall_mean = np.mean(expert_means[i])
            overall_std = np.mean(expert_stds[i])
            overall_max = np.max(expert_maxs[i])
            overall_min = np.min(expert_mins[i])

            log_message(f"Expert {i} overall stats: mean={overall_mean:.4f}, std={overall_std:.4f}, "
                        f"max={overall_max:.4f}, min={overall_min:.4f}")

            # 只有在收集了logits数据时才计算
            if expert_all_logits[i]:
                all_logits_mean = np.mean(expert_all_logits[i])
                all_logits_std = np.std(expert_all_logits[i])
                log_message(f"Expert {i} all logits: mean={all_logits_mean:.4f}, std={all_logits_std:.4f}")

    # 计算并输出每个专家最后一层权重的L2范数
    if output_weight_norms:
        log_message("\n=== 专家最后一层权重范数 ===")
        expert_weights = calculate_expert_weight_norms(model)
    else:
        expert_weights = []

    # 如果提供了结果路径，保存直方图
    if result_path is not None and FLAG and output_stats and any(expert_all_logits[0]):
        try:
            import matplotlib.pyplot as plt

            # 为每个专家绘制logits分布直方图
            for i in range(num_experts):
                if not expert_all_logits[i]:
                    continue

                plt.figure(figsize=(10, 6))
                plt.hist(expert_all_logits[i], bins=50, alpha=0.7)
                plt.title(f'Expert {i} Logits Distribution (α1={alpha1:.2f}, α2={alpha2:.2f})')
                plt.xlabel('Logit Values')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)

                # 确保保存路径存在
                os.makedirs(result_path, exist_ok=True)
                plt.savefig(os.path.join(result_path, f'expert{i}_logits_hist_a1_{alpha1:.2f}_a2_{alpha2:.2f}.png'))
                plt.close()

            # 绘制专家权重范数条形图
            if output_weight_norms and expert_weights:
                plt.figure(figsize=(10, 6))
                x = np.arange(num_experts)
                weight_values = [w.item() for w in expert_weights]

                plt.bar(x, weight_values, width=0.6)
                plt.xlabel('Expert Index')
                plt.ylabel('Weight Norm (squared)')
                plt.title(f'Expert Weight Norms (α1={alpha1:.2f}, α2={alpha2:.2f})')
                plt.xticks(x, [f'Expert {i}' for i in range(num_experts)])
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(result_path, f'expert_weight_norms_a1_{alpha1:.2f}_a2_{alpha2:.2f}.png'))
                plt.close()

        except Exception as e:
            print(f"Error generating visualizations: {str(e)}")

    # 如果需要，绘制混淆矩阵
    if result_path is not None and FLAG and all_preds and all_targets:
        try:
            from sklearn.metrics import confusion_matrix
            import matplotlib.pyplot as plt
            import seaborn as sns

            # 确保有足够的预测结果再绘制混淆矩阵
            if len(all_preds) > 100:  # 设置一个最小阈值
                cm = confusion_matrix(all_targets, all_preds)
                plt.figure(figsize=(10, 8))
                # 对于大型混淆矩阵，不显示具体数字
                sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title(f'Confusion Matrix (α1={alpha1:.2f}, α2={alpha2:.2f})')

                # 确保保存路径存在
                os.makedirs(result_path, exist_ok=True)
                plt.savefig(os.path.join(result_path, f'confusion_matrix_a1_{alpha1:.2f}_a2_{alpha2:.2f}.png'))
                plt.close()
        except Exception as e:
            print(f"Error generating confusion matrix: {str(e)}")

    return avg_loss, accuracy


# 获取专家索引
def get_expert_indices(targets, class_ranges):
    """
    根据目标类别确定对应的专家索引
    """
    expert_indices = torch.zeros_like(targets)

    for i, (start_class, end_class) in enumerate(class_ranges):
        mask = (targets >= start_class) & (targets <= end_class)
        expert_indices[mask] = i

    return expert_indices


# 路由器准确性验证函数保持不变
def validate_router_accuracy(model, val_loader, device):
    """验证路由器的准确率（如果模型包含路由器）"""
    # 这个函数可能不适用于MoE4Model，但保留以供参考
    pass


def compare_model_parameters(model1, model2, tolerance=1e-6, verbose=True):
    """
    比较两个模型的参数是否完全相同。

    Args:
        model1: 第一个模型
        model2: 第二个模型
        tolerance: 浮点数比较的容差值，默认为1e-6
        verbose: 是否输出详细信息，默认为True

    Returns:
        bool: 如果所有参数都相同返回True，否则返回False
    """
    # 获取两个模型的状态字典
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    # 检查参数键是否相同
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())

    # 检查键集合是否一致
    if keys1 != keys2:
        if verbose:
            print(f"模型参数键不匹配!")
            print(f"模型1独有的键: {keys1 - keys2}")
            print(f"模型2独有的键: {keys2 - keys1}")
        return False

    # 检查每个参数是否相同
    all_match = True
    for key in keys1:
        if not torch.allclose(state_dict1[key], state_dict2[key], rtol=tolerance, atol=tolerance):
            all_match = False
            if verbose:
                # 计算差异
                diff = (state_dict1[key] - state_dict2[key]).abs()
                max_diff = diff.max().item()
                avg_diff = diff.float().mean().item()
                print(f"参数'{key}'不匹配: 最大差异={max_diff:.8f}, 平均差异={avg_diff:.8f}")

                # 如果是较小的张量，打印出���体值进行比较
                if state_dict1[key].numel() < 10:
                    print(f"  模型1: {state_dict1[key]}")
                    print(f"  模型2: {state_dict2[key]}")

    if verbose:
        if all_match:
            print("所有参数完全匹配！")
        else:
            print("参数不完全匹配！")

    return all_match
