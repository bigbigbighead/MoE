import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ResNet import resnet18_backbone, resnet18


class ExpertHead(nn.Module):
    """专家网络的分类头"""

    def __init__(self, in_features, num_classes):
        super(ExpertHead, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)


class MixtureOfExperts(nn.Module):
    def __init__(self, class_ranges, input_channels=1, input_height=1, input_width=1024,
                 dropout_rate=0.1, shared_backbone=True):
        """
        初始化混合专家模型
        
        Args:
            class_ranges: 专家负责的类别范围列表，格式为[(start1, end1), (start2, end2), ...]
            input_channels: 输入通道数
            input_height: 输入高度
            input_width: 输入宽度
            dropout_rate: Dropout率
            shared_backbone: 是否使用共享骨干网络
        """
        super(MixtureOfExperts, self).__init__()
        self.class_ranges = class_ranges
        self.num_experts = len(class_ranges)
        self.shared_backbone = shared_backbone

        # 计算每个专家负责的类别数
        self.expert_num_classes = [end - start + 1 for start, end in class_ranges]
        self.max_expert_classes = max(self.expert_num_classes)

        # 创建粗分类器
        if shared_backbone:
            # 共享骨干网络
            self.backbone = resnet18_backbone(
                input_channels=input_channels,
                input_height=input_height,
                input_width=input_width,
                dropout_rate=dropout_rate
            )
            self.coarse_head = nn.Linear(self.backbone.feat_dim, self.num_experts)

            # 创建每个专家的分类头
            self.expert_heads = nn.ModuleList([
                ExpertHead(self.backbone.feat_dim, num_classes)
                for num_classes in self.expert_num_classes
            ])
        else:
            # 独立模型
            self.coarse_classifier = resnet18(
                num_classes=self.num_experts,
                input_channels=input_channels,
                input_height=input_height,
                input_width=input_width,
                dropout_rate=dropout_rate
            )

            # 创建每个专家网络
            self.experts = nn.ModuleList([
                resnet18(
                    num_classes=num_classes,
                    input_channels=input_channels,
                    input_height=input_height,
                    input_width=input_width,
                    dropout_rate=dropout_rate
                )
                for num_classes in self.expert_num_classes
            ])

    def select_expert(self, coarse_output):
        """根据粗分类器输出选择专家"""
        # 获取最高概率的专家索引
        expert_id = torch.argmax(coarse_output, dim=1)
        return expert_id

    def forward(self, x, mode='inference'):
        """
        前向传播
        
        Args:
            x: 输入张量
            mode: 运行模式，可以是'inference'、'coarse_only'或'experts_only'
            
        Returns:
            如果mode是'inference'，返回(粗分类输出, 专家索引, 本地类别预测, 全局类别预测)
            如果mode是'coarse_only'，返回粗分类器输出
            如果mode是'experts_only'，返回所有专家的输出列表
        """
        batch_size = x.size(0)
        device = x.device

        if self.shared_backbone:
            # 提取共享特征
            shared_features = self.backbone(x)

            # 粗分类头
            coarse_output = self.coarse_head(shared_features)

            if mode == 'coarse_only':
                return coarse_output

            # 选择专家
            expert_id = self.select_expert(coarse_output)

            if mode == 'experts_only':
                # 返回所有专家的输出
                expert_outputs = [head(shared_features) for head in self.expert_heads]
                return expert_outputs

            # 优化的推理模式：更高效地选择专家和处理预测
            # 初始化输出张量
            local_preds = torch.zeros(batch_size, self.max_expert_classes, device=device)
            global_preds = torch.zeros(batch_size, device=device)

            # 对每个专家分别处理
            for e_id in range(self.num_experts):
                # 找出当前批次中由这个专家处理的样本索引
                expert_mask = (expert_id == e_id)
                expert_indices = expert_mask.nonzero(as_tuple=True)[0]

                # 如果没有样本由这个专家处理，跳过
                if len(expert_indices) == 0:
                    continue

                # 获取这些样本的特征
                expert_features = shared_features[expert_indices]

                # 使用当前专家处理这些样本
                expert_output = self.expert_heads[e_id](expert_features)

                # 获取每个样本的本地类别预测（最高概率的类别索引）
                local_class_indices = torch.argmax(expert_output, dim=1)

                # 将本地类别转换为全局类别
                global_class_indices = local_class_indices + self.class_ranges[e_id][0]

                # 保存预测结果
                global_preds[expert_indices] = global_class_indices.float()

                # 保存本地类别预测概率（确保形状匹配）
                probs = F.softmax(expert_output, dim=1)
                for i, idx in enumerate(expert_indices):
                    local_preds[idx, :self.expert_num_classes[e_id]] = probs[i]

            return coarse_output, expert_id, local_preds, global_preds

        else:  # 非共享骨干网络
            # 粗分类器前向传播
            coarse_output = self.coarse_classifier(x)

            if mode == 'coarse_only':
                return coarse_output

            # 选择专家
            expert_id = self.select_expert(coarse_output)

            if mode == 'experts_only':
                # 返回所有专家的输出
                expert_outputs = [expert(x) for expert in self.experts]
                return expert_outputs

            # 优化的推理模式：更高效地选择专家和处理预测
            # 初始化输出张量
            local_preds = torch.zeros(batch_size, self.max_expert_classes, device=device)
            global_preds = torch.zeros(batch_size, device=device)

            # 对每个专家分别处理
            for e_id in range(self.num_experts):
                # 找出当前批次中由这个专家处理的样本索引
                expert_mask = (expert_id == e_id)
                expert_indices = expert_mask.nonzero(as_tuple=True)[0]

                # 如果没有样本由这个专家处理，跳过
                if len(expert_indices) == 0:
                    continue

                # 获取这些样本
                expert_inputs = x[expert_indices]

                # 使用当前专家处理这些样本
                expert_output = self.experts[e_id](expert_inputs)

                # 获取每个样本的本地类别预测（最高概率的类别索引）
                local_class_indices = torch.argmax(expert_output, dim=1)

                # 将本地类别转换为全局类别
                global_class_indices = local_class_indices + self.class_ranges[e_id][0]

                # 保存预测结果
                global_preds[expert_indices] = global_class_indices.float()

                # 保存本地类别预测概率（确保形状匹配）
                probs = F.softmax(expert_output, dim=1)
                for i, idx in enumerate(expert_indices):
                    local_preds[idx, :self.expert_num_classes[e_id]] = probs[i]

            return coarse_output, expert_id, local_preds, global_preds

    def get_expert_target(self, target):
        """
        将原始目标标签转换为专家索引
        
        Args:
            target: 原始类别标签 [batch_size]
            
        Returns:
            专家索引 [batch_size]
        """
        expert_target = torch.zeros_like(target)

        for i, (start, end) in enumerate(self.class_ranges):
            mask = (target >= start) & (target <= end)
            expert_target[mask] = i

        return expert_target

    def get_local_target(self, target, expert_id):
        """
        将全局目标转换为专家内的本地目标
        
        Args:
            target: 原始类别标签 [batch_size]
            expert_id: 专家索引 [batch_size]
            
        Returns:
            本地类别标签 [batch_size]
        """
        local_target = torch.zeros_like(target)

        for i in range(len(target)):
            e_id = expert_id[i].item()
            start_class = self.class_ranges[e_id][0]
            local_target[i] = target[i] - start_class

        return local_target
