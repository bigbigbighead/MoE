import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.loss_functions import compute_specialized_loss, combine_expert_outputs
from utils.data_loading_mine import log_message
from utils.visualization import plot_expert_logits_histograms


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

            # 获取特征
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
def validate_full_model(model, val_loader, criterion, device, RESULTS_PATH, FLAG=True):
    """
    验证完整模型的性能

    按照Model design.md，推理输出公式为:
    o^c = (1/|S_c|) * sum(z_i)，其中z_i是专家i对类别c的输出
    在本模型中，由于专家责任范围互不重叠，每个类别只由一个专家负责，公式简化为直接取专家输出
    """
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    # 统计每个专家的logits范围
    expert_logits_stats = []
    for i in range(len(model.experts) if not hasattr(model, 'module') else len(model.module.experts)):
        expert_logits_stats.append({
            'min': float('inf'),
            'max': float('-inf'),
            'mean': [],
            'std': []
        })

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # 使用inference方法获取模型预测（确保使用正确的推理逻辑）
            _, expert_outputs, logits = model(inputs)

            # 只对第一个batch的数据进行可视化
            if batch_idx == 0:
                plot_expert_logits_histograms(expert_outputs, RESULTS_PATH)

            # 统计每个专家的logits范围
            for i, expert_output in enumerate(expert_outputs):
                expert_logits = expert_output.detach().cpu().numpy()
                expert_logits_stats[i]['min'] = min(expert_logits_stats[i]['min'], np.min(expert_logits))
                expert_logits_stats[i]['max'] = max(expert_logits_stats[i]['max'], np.max(expert_logits))
                expert_logits_stats[i]['mean'].append(np.mean(expert_logits))
                expert_logits_stats[i]['std'].append(np.std(expert_logits))

            if FLAG and batch_idx == 0:
                log_message(f"sample:{inputs[0]}")
                log_message(f"logits:{logits[0]}")
                FLAG = False

            # 计算损失
            loss = criterion(logits, targets)

            val_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # 计算并记录每个专家的logits统计信息
    log_message("\n专家logits统计信息:")
    for i, stats in enumerate(expert_logits_stats):
        # 计算全局平均值和标准差
        global_mean = np.mean(stats['mean'])
        global_std = np.mean(stats['std'])
        log_message(f"专家{i + 1}: 最小值={stats['min']:.4f}, 最大值={stats['max']:.4f}, "
                    f"平均值={global_mean:.4f}, 标准差={global_std:.4f}")

    return val_loss / len(val_loader), correct / total


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

                # 如果是较小的张量，打印出具体值进行比较
                if state_dict1[key].numel() < 10:
                    print(f"  模型1: {state_dict1[key]}")
                    print(f"  模型2: {state_dict2[key]}")

    if verbose:
        if all_match:
            print("所有参数完全匹配！")
        else:
            print("参数不完全匹配！")

    return all_match
