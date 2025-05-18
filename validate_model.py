import torch
from torch.cuda.amp import autocast, GradScaler
from utils.data_loading_mine import log_message

# 自动混合精度训练配置
USE_AMP = True  # 启动自动混合精度训练


def validate_expert(model, data_loader, criterion, expert_idx, device):
    """验证单个专家的性能"""
    model.eval()
    start_class, end_class = model.class_ranges[expert_idx]
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # 只筛选该专家负责的类别范围内的样本
            mask = (targets >= start_class) & (targets <= end_class)
            if mask.sum() > 0:
                expert_inputs = inputs[mask]
                expert_targets = targets[mask]  # 调整目标标签范围

                # 前向传播
                features = model.backbone(expert_inputs)
                outputs = model.experts[expert_idx](features)

                # 计算损失
                loss = criterion(outputs, expert_targets)
                val_loss += loss.item() * mask.sum().item()

                # 计算准确率
                _, predicted = outputs.max(1)
                total += expert_targets.size(0)
                correct += predicted.eq(expert_targets).sum().item()

    # 确保避免除零错误
    if total == 0:
        return 0, 0

    return val_loss / total, correct / total


def validate_full_model(model, data_loader, criterion, device, split="valid"):
    """验证完整模型性能，并返回每个类别区间的准确率"""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    # 为每个类别区间跟踪准确率
    class_correct = [0] * len(model.class_ranges)
    class_total = [0] * len(model.class_ranges)

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            # 使用混合精度推理
            if USE_AMP:
                with autocast():
                    # 使用推理模式
                    outputs = model.inference(inputs)

                    # 计算损失
                    loss = criterion(outputs, targets)
            else:
                # 使用推理模式
                outputs = model.inference(inputs)

                # 计算损失
                loss = criterion(outputs, targets)

            val_loss += loss.item()

            # 计算准确率
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 计算每个类别区间的准确率
            for i, (start, end) in enumerate(model.class_ranges):
                mask = (targets >= start) & (targets <= end)
                if mask.sum() > 0:
                    class_total[i] += mask.sum().item()
                    class_correct[i] += predicted[mask].eq(targets[mask]).sum().item()

    # 计算平均损失和总体准确率
    avg_loss = val_loss / len(data_loader)
    accuracy = correct / total

    # 计算每个类别区间的准确率
    class_accuracies = [correct / total if total > 0 else 0 for correct, total in zip(class_correct, class_total)]

    log_message(f"\t{split.capitalize()} loss: {avg_loss:.4f}, accuracy: {accuracy:.4f}")
    for i, (start, end) in enumerate(model.class_ranges):
        log_message(
            f"\t  类别 {start}-{end} 准确率: {class_accuracies[i]:.4f} ({class_correct[i]}/{class_total[i]})")

    return avg_loss, accuracy, class_accuracies


def validate_router_accuracy(model, data_loader, device):
    """
    验证路由器的分类准确率
    
    Args:
        model: 模型
        data_loader: 数据加载器
        device: 设备
        
    Returns:
        float: 路由器的分类准确率
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # 获取每个样本应该属于哪个专家
            expert_targets = get_expert_indices(targets, model.class_ranges).to(device)

            # 使用混合精度推理
            if USE_AMP:
                with autocast():
                    # 获取backbone特征
                    features = model.backbone(inputs)
                    # 获取路由器的输出
                    routing_weights, router_logits = model.router(features)
            else:
                # 获取backbone特征
                features = model.backbone(inputs)
                # 获取路由器的输出
                routing_weights, router_logits = model.router(features)

            # 获取路由器预测的专家
            _, predicted_experts = router_logits.max(1)

            # 统计正确分类的样本数
            total += targets.size(0)
            correct += predicted_experts.eq(expert_targets).sum().item()

    # 计算路由器准确率
    router_accuracy = correct / total if total > 0 else 0

    log_message(f"\t路由器分类准确率: {router_accuracy:.4f} ({correct}/{total})")

    return router_accuracy


# 辅助函数：将原始类别标签转换为对应的专家索引
def get_expert_indices(targets, class_ranges):
    """
    将原始类别标签映射到对应的专家索引

    Args:
        targets: 原始类别标签
        class_ranges: 专家负责的类别范围列表 [(start1, end1), (start2, end2), ...]

    Returns:
        专家索引张量
    """
    expert_indices = torch.zeros_like(targets)
    for expert_idx, (start_class, end_class) in enumerate(class_ranges):
        expert_indices[(targets >= start_class) & (targets <= end_class)] = expert_idx
    return expert_indices
