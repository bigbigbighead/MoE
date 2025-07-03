import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ResNet_ACE import resnet18_shared_backbone, resnet18_expert_backbone
from utils.data_loading_mine import log_message


class SpecializedExpert(nn.Module):
    """
    专门负责特定类别范围的专家
    直接包含独享的深层网络和全连接分类层
    """

    def __init__(self, feat_dim, num_classes, input_channels=1, input_height=1, input_width=1024, dropout_rate=0):
        """
        Args:
            feat_dim: 特征维度
            num_classes: 总类别数
            input_channels: 输入通道数
            input_height: 输入高度
            input_width: 输入宽度
            dropout_rate: dropout比率
        """
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # 专家独享的深层网络
        self.expert_backbone = resnet18_expert_backbone(
            input_channels=input_channels,
            input_height=input_height,
            input_width=input_width,
            dropout_rate=dropout_rate
        )

        # 分类器 - 输出全部类别的logits
        self.classifier = nn.Sequential(
            nn.Linear(self.expert_backbone.feat_dim, num_classes)
        )

    def forward(self, x):
        # 通过深层网络提取特征
        feat = self.expert_backbone(x)
        # 通过分类器获得logits
        logits = self.classifier(feat)
        return feat, logits

    def get_last_layer_weights(self):
        """获取专家分类器最后一层的权重"""
        # 假设最后一层是线性层
        last_layer = self.classifier[-1]
        if isinstance(last_layer, nn.Linear):
            return last_layer.weight
        return None


class MoE_ACE(nn.Module):
    """
    Mixture of Experts - Additive Cooperative Experts (MoE-ACE)

    按照MoE-ACE.md的设计：
    - 模型的backbone是ResNet-18，由多个专家组成
    - 每个专家共享浅层（前2个残差块）而独享深层（后两个残差块）
    - 每个专家有自己负责的类别范围
    - 第一个专家负责所有类别，其他专家负责特定类别范围
    """

    def __init__(self, total_classes=200, class_ranges=None,
                 input_channels=1, input_height=1, input_width=1024, dropout_rate=0):
        super().__init__()

        # 默认类别范围划分，如果未提供
        if class_ranges is None:
            # 第一个专家负责所有类别，后面的专家分别负责不同的类别范围
            class_ranges = [(0, 199), (0, 99), (100, 149), (150, 199)]

        self.class_ranges = class_ranges
        self.num_experts = len(class_ranges)
        self.total_classes = total_classes
        self.dropout_rate = dropout_rate

        # ResNet18 backbone - 共享部分（前2个残差块）
        self.shared_backbone = resnet18_shared_backbone(
            input_channels=input_channels,
            input_height=input_height,
            input_width=input_width,
            dropout_rate=dropout_rate
        )

        # 确定共享backbone的输出特征维度
        self.shared_feat_dim = self.shared_backbone.feat_dim

        # 创建专家 - 每个专家包含独享的深层网络和分类器
        self.experts = nn.ModuleList()
        for i in range(self.num_experts):
            expert = SpecializedExpert(
                feat_dim=self.shared_feat_dim,
                num_classes=total_classes,
                input_channels=input_channels,
                input_height=input_height,
                input_width=input_width,
                dropout_rate=dropout_rate
            )
            self.experts.append(expert)

        # 为每个类别构建负责的专家索引
        self.class_to_experts = {}
        for c in range(total_classes):
            self.class_to_experts[c] = []
            for i, (start, end) in enumerate(self.class_ranges):
                if start <= c <= end:
                    self.class_to_experts[c].append(i)

        # 缓存计算结果的标志
        self.has_cached_computation = False
        # 使用普通属性来存储缩放因子
        self.cached_scaling_factors = None

    def forward(self, x):
        # 首先通过共享的浅层backbone
        shared_feat = self.shared_backbone(x)

        # 然后通过每个专家的独享深层和分类器
        expert_feats = []
        expert_outputs = []

        for expert in self.experts:
            # 使用专家的前向传播方法处理共享特征
            expert_feat, expert_output = expert(shared_feat)
            expert_feats.append(expert_feat)
            expert_outputs.append(expert_output)

        # 创建一个全零的输出张量，用于存放最终的输出logits
        batch_size = x.size(0)
        combined_logits = torch.zeros(batch_size, self.total_classes, device=x.device)

        # 如果已经预计算了缩放因子，则使用缩放因子
        if self.has_cached_computation:
            # 使用预计算的缩放因子
            for c in range(self.total_classes):
                if c in self.cached_scaling_factors:
                    class_scaling_factors = self.cached_scaling_factors[c]

                    # 收集负责该类别的所有专家的调整后输出
                    adjusted_expert_outputs = []

                    for expert_idx, scaling_factor in class_scaling_factors:
                        # 在这里，scaling_factor是标量值
                        scaling_factor_tensor = torch.tensor(scaling_factor, device=x.device)
                        adjusted_output = expert_outputs[expert_idx][:, c] * scaling_factor_tensor
                        adjusted_expert_outputs.append(adjusted_output)

                    # 对重叠专家的输出取平均值
                    if adjusted_expert_outputs:
                        stacked_outputs = torch.stack(adjusted_expert_outputs, dim=0)
                        mean_output = torch.mean(stacked_outputs, dim=0)
                        combined_logits[:, c] = mean_output
        else:
            # 未预计算缩放因子时，直接相加专家输出
            # 直接将每个专家的输出相加到combined_logits
            for c in range(self.total_classes):
                responsible_experts = self.class_to_experts[c]
                if len(responsible_experts) > 0:
                    expert_logits = [expert_outputs[idx][:, c] for idx in responsible_experts]
                    combined_logits[:, c] = torch.stack(expert_logits).mean(dim=0)

        return expert_feats, expert_outputs, combined_logits

    def compute_loss(self, outputs, targets, expert_idx):
        """
        计算专家的损失函数，按照MoE-ACE.md的设计

        L_cls = 交叉熵损失 (目标类别)
        L_com = 干扰类别logits的L2范数
        L = L_cls + L_com

        Args:
            outputs: 专家输出的logits，维度为 [batch_size, total_classes]
            targets: 相对于该专家负责类别范围的标签，维度为 [batch_size]
            expert_idx: 专家索引

        Returns:
            cls_loss: 分类损失 (L_cls)
            reg_loss: 正则化损失 (L_com)
            total_loss: 总损失 (L = L_cls + L_com)
        """
        # 获取当前专家负责的类别范围
        start_class, end_class = self.class_ranges[expert_idx]

        # 1. 分类损失 (L_cls) - 交叉熵损失
        criterion = nn.CrossEntropyLoss()
        # 只计算专家负责范围内的类别损失
        # 注意：targets是相对于专家类别范围的索引
        cls_loss = criterion(outputs, targets)

        # 2. 正则化损失 (L_com)
        # 根据MoE-ACE.md: L_com^i(B_i)=\sum_{c_j\in \tilde{C_i}}^{C}||z_i^{c_j}||^2
        # 创建干扰类别的掩码
        interference_mask = torch.zeros(self.total_classes, device=outputs.device)

        # 设置干扰类别 (不在专家负责范围内的类别)
        for c in range(self.total_classes):
            if not (start_class <= c <= end_class):
                interference_mask[c] = 1.0

        # 应用掩码，仅保留干扰类别的logits
        batch_size = outputs.size(0)
        interference_mask = interference_mask.expand(batch_size, -1)
        interference_logits = outputs * interference_mask

        # 计算干扰类别logits的L2范数
        reg_loss = torch.norm(interference_logits, dim=1).pow(2).mean()

        # 3. 总损失 = 分类损失 + 正则化损失
        total_loss = cls_loss + reg_loss

        return cls_loss, reg_loss, total_loss

    def precompute_scaling_factors(self):
        """
        预计算所有专家的缩放因子并存储

        根据MoE-ACE.md:
        ẑ_i = (||w_1||²/||w_i||²)·z_i
        w_i为专家i的全连接层权重
        w_1为第一个专家的全连接层权重

        修改：每个专家只计算其负责类别范围内权重的范数平均值
        """
        # 获取各专家最后一层全连接层的权重
        expert_weights = []

        # 计算每个专家的权重范数，只考虑其负责的类别范围
        for i, expert in enumerate(self.experts):
            weight = expert.get_last_layer_weights()
            if weight is not None:
                start_class, end_class = self.class_ranges[i]

                # 只选择专家负责的类别权重
                responsible_weights = weight[start_class:end_class + 1]

                # 计算权重的范数 ||w||
                tao = 1
                weight_norm_squared = torch.norm(weight, p=2, dim=1).pow(tao).mean()
                # log_message(
                #     f"专家 {i} 的{tao}次幂缩放权重范数(只考虑负责类别范围[{start_class}-{end_class}]): {weight_norm_squared.item():.4f}")
                log_message(
                    f"专家 {i} 的{tao}次幂缩放权重范数: {weight_norm_squared.item():.4f}")
                expert_weights.append(weight_norm_squared)
            else:
                # 如果无法获取权重，则使用默认值1.0
                log_message(f"专家 {i} 无法获取权重，使用默认值1.0")
                expert_weights.append(torch.tensor(1.0, device=weight.device if weight is not None else 'cpu'))

        # 使用第一个专家的权重作为参考
        reference_weight_norm = expert_weights[0]

        # 初始化缩放因子字典
        scaling_factors_dict = {}

        # 计算缩放因子
        for c in range(self.total_classes):
            responsible_experts = self.class_to_experts[c]
            if len(responsible_experts) > 0:
                class_scaling_factors = []
                for expert_idx in responsible_experts:
                    # 缩放因子 = ||w_1||²/||w_i||²
                    scaling_factor = expert_weights[expert_idx] / reference_weight_norm
                    class_scaling_factors.append((expert_idx, scaling_factor.item()))
                scaling_factors_dict[c] = class_scaling_factors

        # 存储缩放因子字典
        self.cached_scaling_factors = scaling_factors_dict
        self.has_cached_computation = True

        return scaling_factors_dict

    def inference(self, x):
        """
        模型推理，根据MoE-ACE.md中的公式计算输出

        o^c = (1/|S_c|)∑_{ε_i∈S_c}ẑ_i
        ẑ_i = (||w_1||²/||w_i||²)·z_i
        """
        # 首先通过共享的浅层backbone
        shared_feat = self.shared_backbone(x)

        # 获取每个专家的原始输出
        expert_outputs = []
        for expert in self.experts:
            _, expert_output = expert(shared_feat)
            expert_outputs.append(expert_output)

        # 创建一个全零的输出张量
        batch_size = x.size(0)
        combined_logits = torch.zeros(batch_size, self.total_classes, device=x.device)

        # 使用预计算的缩放因子(如果已预计算)
        if self.has_cached_computation:
            # 使用预计算的缩放因子
            for c in range(self.total_classes):
                if c in self.cached_scaling_factors:
                    class_scaling_factors = self.cached_scaling_factors[c]

                    # 收集负责该类别的所有专家的调整后输出
                    adjusted_expert_outputs = []

                    for expert_idx, scaling_factor in class_scaling_factors:
                        scaling_factor_tensor = torch.tensor(scaling_factor, device=x.device)
                        adjusted_output = expert_outputs[expert_idx][:, c] * scaling_factor_tensor
                        adjusted_expert_outputs.append(adjusted_output)

                    # 对重叠专家的输出取平均值
                    if adjusted_expert_outputs:
                        stacked_outputs = torch.stack(adjusted_expert_outputs, dim=0)
                        mean_output = torch.mean(stacked_outputs, dim=0)
                        combined_logits[:, c] = mean_output
        else:
            # 未预计算时，动态计算缩放因子
            # 获取各专家最后一层全连接层的权重
            expert_weights = []

            # 计算每个专家的权重范数，只考虑其负责的类别范围
            for i, expert in enumerate(self.experts):
                weight = expert.get_last_layer_weights()
                if weight is not None:
                    start_class, end_class = self.class_ranges[i]

                    # 只选择专家负责的类别权重
                    responsible_weights = weight[start_class:end_class + 1]

                    # 计算负责类别范围内权重的平方范数平均值
                    weight_norm_squared = torch.norm(responsible_weights, p=2, dim=1).pow(2).mean()
                    expert_weights.append(weight_norm_squared)
                else:
                    expert_weights.append(torch.tensor(1.0, device=x.device))

            # 使用第一个专家的权重作为参考
            reference_weight_norm = expert_weights[0]

            # 为每个类别计算最终的logits输出
            for c in range(self.total_classes):
                responsible_experts = self.class_to_experts[c]
                if len(responsible_experts) == 0:
                    continue

                adjusted_expert_outputs = []
                for expert_idx in responsible_experts:
                    scaling_factor = reference_weight_norm / expert_weights[expert_idx]
                    adjusted_output = expert_outputs[expert_idx][:, c] * scaling_factor
                    adjusted_expert_outputs.append(adjusted_output)

                if adjusted_expert_outputs:
                    stacked_outputs = torch.stack(adjusted_expert_outputs, dim=0)
                    mean_output = torch.mean(stacked_outputs, dim=0)
                    combined_logits[:, c] = mean_output

        return expert_outputs, combined_logits

    def predict(self, x):
        """返回预测的类别"""
        _, logits = self.inference(x)
        _, predicted = torch.max(logits, 1)
        return predicted
