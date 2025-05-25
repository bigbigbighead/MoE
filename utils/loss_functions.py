import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_specialized_loss(logits, targets, responsible_classes, total_classes):
    """
    计算专家的专用损失函数，根据Model design.md的定义
    
    按照设计文档：
    L_cls = 交叉熵损失 (目标类别)
    L_com = 干扰类别logits的L2范数平方和
    L = L_cls + L_com
    
    注意：根据设计，每个专家只接收属于自己目标类的样本进行训练，
    并且专家只输出其负责的类别logits
    
    Args:
        logits: 专家输出的logits，形状为 [batch_size, expert_num_classes]
        targets: 相对于该专家负责类别范围的标签，形状为 [batch_size]
        responsible_classes: 该专家负责的类别集合
        total_classes: 总类别数
        
    Returns:
        cls_loss: 分类损失 (L_cls)
        reg_loss: 正则化损失 (L_com)
        total_loss: 总损失 (L = L_cls + L_com)
    """
    # 1. 计算分类损失 L_cls (交叉熵损失)
    criterion = nn.CrossEntropyLoss()
    cls_loss = criterion(logits, targets)

    # 2. 计算正则化损失 L_com
    # 在设计文档中，L_com是干扰类别logits的L2范数平方和
    # 由于我们的专家只输出自己负责的类别(即目标类)，
    # 而干扰类别的logits由于架构设计不会被专家输出
    # 所以L_com实际上为0
    reg_loss = torch.tensor(0.0, device=logits.device)

    # 3. 总损失 L = L_cls + L_com
    total_loss = cls_loss + reg_loss

    return cls_loss, reg_loss, total_loss


def combine_expert_outputs(expert_outputs, class_ranges, num_classes):
    """
    按照Model design.md中的公式合并专家输出

    o^c = (1/|S_c|) * sum(z_i)，其中z_i是专家i对类别c的输出
    
    在本模型中，由于每个类别只由一个专家负责(互不重叠)，|S_c| = 1
    所以直接将各专家输出拼接即可
    
    Args:
        expert_outputs: 各专家的输出 list[tensor]
        class_ranges: 各专家负责的类别范围 [(start1, end1), (start2, end2), ...]
        num_classes: 总类别数
        
    Returns:
        combined_logits: 合并后的logits
    """
    if not expert_outputs:
        return None

    batch_size = expert_outputs[0].shape[0]
    device = expert_outputs[0].device

    # 创建输出tensor
    combined_logits = torch.zeros(batch_size, num_classes, device=device)

    # 将每个专家的输出放到对应的类别位置
    for i, (expert_output, (start_class, end_class)) in enumerate(zip(expert_outputs, class_ranges)):
        # 检查专家输出维度与其负责的类别范围是否匹配
        range_width = end_class - start_class + 1
        assert expert_output.size(1) == range_width, \
            f"Expert {i} output width ({expert_output.size(1)}) != range width ({range_width})"

        # 将专家输出放到对应位置
        combined_logits[:, start_class:end_class + 1] = expert_output

    return combined_logits
