import torch
import torch.nn as nn
import torch.nn.functional as F

# 验证单个专家的性能
def validate_expert(model, val_loader, criterion, expert_idx, device):
    """验证单个专家的性能"""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    start_class, end_class = model.module.class_ranges[expert_idx] if hasattr(model, 'module') else model.class_ranges[expert_idx]
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 获取特征
            features = model.module.backbone(inputs) if hasattr(model, 'module') else model.backbone(inputs)
            
            # 获取专家输出
            outputs = model.module.experts[expert_idx](features) if hasattr(model, 'module') else model.experts[expert_idx](features)
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return val_loss / len(val_loader), correct / total

# 验证完整模型的性能
def validate_full_model(model, val_loader, criterion, device):
    """验证完整模型的性能，使用模型的inference方法确保使用新的推理逻辑"""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 使用inference方法获取模型预测（确保使用拼接逻辑）
            logits = model.inference(inputs)
            
            loss = criterion(logits, targets)
            
            val_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return val_loss / len(val_loader), correct / total

# 其他验证函数
def validate_router_accuracy(model, val_loader, device):
    """验证路由器的准确率（如果模型包含路由器）"""
    # 这个函数可能不适用于MoE4Model，但保留以供参考
    pass

def get_expert_indices(targets, class_ranges):
    """根据目标类别确定对应的专家索引"""
    expert_indices = torch.zeros_like(targets)
    
    for i, (start_class, end_class) in enumerate(class_ranges):
        mask = (targets >= start_class) & (targets <= end_class)
        expert_indices[mask] = i
    
    return expert_indices
