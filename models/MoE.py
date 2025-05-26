import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ResNet import ResNetBlock, ResNetBackbone, resnet18_backbone
from utils.loss_functions import compute_specialized_loss


class MoE4Router(nn.Module):
    """专为MoE4设计的路由器，根据类别范围分配专家"""

    def __init__(self, feat_dim, num_experts=3, routing_type='softmax', dropout_rate=0.1):
        super().__init__()
        self.routing_type = routing_type
        self.router = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # 添加更高比例的dropout
            nn.Linear(256, num_experts)
        )

    def forward(self, x):
        router_logits = self.router(x)

        if self.routing_type == 'softmax':
            # 软路由：每个样本对所有专家的权重分布
            routing_weights = F.softmax(router_logits, dim=1)
        else:
            # 硬路由：每个样本只选择概率最高的专家
            max_indices = router_logits.argmax(dim=1)
            # 修改这部分，使用带有梯度的伪硬路由方式
            mask = F.one_hot(max_indices, num_classes=router_logits.size(1)).float()

            # 使用STE (Straight-Through Estimator) 技术保持梯度流
            # 前向传播使用硬路由，反向传播使用原始logits的梯度
            routing_weights = mask.detach() + router_logits - router_logits.detach()
        return routing_weights, router_logits


class SpecializedExpert(nn.Module):
    """专门负责特定类别范围的专家"""

    def __init__(self, feat_dim, class_range, dropout_rate=0.2):
        """
        Args:
            feat_dim: 特征维度
            class_range: 该专家负责的类别范围，例如(0,99)表示0-99类
            dropout_rate: dropout比率
        """
        super().__init__()
        self.start_class, self.end_class = class_range
        self.num_classes = self.end_class - self.start_class + 1
        self.dropout_rate = dropout_rate
        # self.dropout_rate = dropout_rate * self.num_classes / 50
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 200)
        )

    def forward(self, x):
        return self.classifier(x)

    def get_last_layer_weights(self):
        """获取专家分类器最后一层的权重"""
        # 假设最后一层是线性层
        last_layer = self.classifier[-1]
        if isinstance(last_layer, nn.Linear):
            return last_layer.weight
        return None


class MoE4Model(nn.Module):
    """按照修改后的设计实现的模型，专家按类别范围分工，输出直接相加"""

    def __init__(self, total_classes=200, class_ranges=None, routing_type='hard',
                 input_channels=1, input_height=1, input_width=1024, dropout_rate=0):
        super().__init__()

        # 默认类别范围划分：0-99, 100-149, 150-199
        if class_ranges is None:
            class_ranges = [(0, 99), (100, 149), (150, 199)]

        self.class_ranges = class_ranges
        self.num_experts = len(class_ranges)
        self.total_classes = total_classes
        self.dropout_rate = dropout_rate

        # ResNet18 backbone - 使用从ResNet.py导入的组件
        self.backbone = resnet18_backbone(
            input_channels=input_channels,
            input_height=input_height,
            input_width=input_width,
            dropout_rate=dropout_rate
        )

        feat_dim = self.backbone.feat_dim  # 特征维度是512

        # 创建专家
        self.experts = nn.ModuleList()
        for class_range in self.class_ranges:
            expert = SpecializedExpert(feat_dim, class_range, dropout_rate=dropout_rate)
            self.experts.append(expert)

        # 为每个类别构建负责的专家索引
        self.class_to_experts = {}
        for c in range(total_classes):
            self.class_to_experts[c] = []
            for i, (start, end) in enumerate(self.class_ranges):
                if start <= c <= end:
                    self.class_to_experts[c].append(i)

    def forward(self, x):
        # 提取特征
        feat = self.backbone(x)

        # 创建一个全零的输出张量，大小为 [batch_size, total_classes]
        batch_size = x.size(0)
        combined_logits = torch.zeros(batch_size, self.total_classes, device=x.device)

        # 获取每个专家的输出
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            # 获取专家的输出
            expert_output = expert(feat)  # [batch_size, expert_num_classes]
            expert_outputs.append(expert_output)

            # 直接将每个专家的输出相加到combined_logits
            combined_logits += expert_output

        # 返回特征、各专家输出和相加后的结果
        return feat, expert_outputs, combined_logits

    def inference(self, x):
        """
        推理模式：根据Model design.md中的公式计算每个类别的输出logits
        使用可学习权重缩放分类器(LWS)调整各专家输出的尺度
        
        对于每个专家i，输出调整后的logits: ̂z_i = (||w_i||²/||w_1||²)·z_i
        其中w_i是专家i的全连接层权重，w_1是第一个专家的权重
        """
        feat = self.backbone(x)
        batch_size = x.size(0)

        # 创建一个全零的输出张量
        combined_logits = torch.zeros(batch_size, self.total_classes, device=x.device)

        # 获取各专家最后一层全连接层的权重
        expert_weights = []
        for expert in self.experts:
            weight = expert.get_last_layer_weights()
            if weight is not None:
                # 计算权重的平方范数 ||w||²
                weight_norm_squared = torch.norm(weight, p=2, dim=1).pow(2).mean()
                expert_weights.append(weight_norm_squared)
            else:
                # 如果无法获取权重，则使用默认值1.0
                expert_weights.append(torch.tensor(1.0, device=x.device))

        # 使用第一个专家的权重作为参考
        reference_weight_norm = expert_weights[0]

        # 获取每个专家的输出并根据权重比例调整后相加
        for i, expert in enumerate(self.experts):
            expert_output = expert(feat)

            if i > 0:  # 第一个专家(i=0)的输出不需要调整
                # 按照公式 ̂z_i = (||w_i||²/||w_1||²)·z_i 进行调整
                scaling_factor = expert_weights[i] / reference_weight_norm
                expert_output = expert_output * scaling_factor

            combined_logits += expert_output

        return combined_logits

    def compute_loss(self, logits, targets, expert_idx):
        """
        计算专家的损失函数，严格按照Model design.md的设计

        按照设计文档定义：
        L_cls = 交叉熵损失 (目标类别)
        L_com = 干扰类别logits的L2范数平方和
        L = L_cls + L_com

        Args:
            logits: 专家输出的logits，维度为 [batch_size, expert_num_classes]
            targets: 相对于该专家负责类别范围的标签，维度为 [batch_size]
            expert_idx: 专家索引

        Returns:
            cls_loss: 分类损失 (L_cls)
            reg_loss: 正则化损失 (L_com)
            total_loss: 总损失 (L = L_cls + L_com)
        """
        # 1. 分类损失 (L_cls) - 交叉熵损失
        criterion = nn.CrossEntropyLoss()
        cls_loss = criterion(logits, targets)

        # 2. 正则化损失 (L_com)
        # 根据设计文档: L_com^i(B_i) = sum_{c_j∈C̃_i}^C ||z_i^{c_j}||^2
        # 获取当前专家负责的类别范围
        start_class, end_class = self.class_ranges[expert_idx]

        # 创建目标类别的one-hot编码掩码，这个掩码表示专家负责的类别范围
        # 首先创建一个全零矩阵
        target_mask = torch.zeros(logits.size(0), self.total_classes, device=logits.device)

        # 然后将专家负责的类别范围设置为1
        target_mask[:, start_class:end_class + 1] = 1.0

        # 计算干扰类别的掩码 (非目标类别)
        interference_mask = 1.0 - target_mask

        # 应用掩码，仅保留干扰类别的logits
        interference_logits = logits * interference_mask

        # 计算每个样本干扰类别logits的L2范数平方和
        # 先计算每个样本的L2范数，再平方，最后求平均
        reg_loss = torch.norm(interference_logits, dim=1).pow(2).mean()

        # 3. 总损失 = 分类损失 + 正则化损失
        total_loss = cls_loss + reg_loss

        return cls_loss, reg_loss, total_loss

    def predict(self, x):
        """返回预测的类别"""
        logits = self.inference(x)
        _, predicted = torch.max(logits, 1)
        return predicted


class Router(nn.Module):
    """专家路由器，决定输入应该由哪个专家处理"""

    def __init__(self, feat_dim, num_experts, routing_type='softmax', dropout_rate=0.2):
        super().__init__()
        self.routing_type = routing_type
        self.dropout = nn.Dropout(dropout_rate)  # 添加dropout
        self.router = nn.Linear(feat_dim, num_experts)

    def forward(self, x):
        x = self.dropout(x)  # 在预测前应用dropout
        router_logits = self.router(x)

        if self.routing_type == 'softmax':
            # 软路由：每个样本对所有专家的权重分布
            routing_weights = F.softmax(router_logits, dim=1)
        else:
            # 硬路由：每个样本只选择概率最高的专家
            max_indices = router_logits.argmax(dim=1)
            routing_weights = F.one_hot(max_indices, num_classes=router_logits.size(1)).float()

        return routing_weights, router_logits


class MoEResNet18(nn.Module):
    def __init__(self, num_classes, num_experts=3, routing_type='softmax', input_channels=1, input_height=1,
                 input_width=1024, dropout_rate=0.2):
        super().__init__()

        # 使用从ResNet.py导入的ResNetBackbone组件
        self.backbone = resnet18_backbone(
            input_channels=input_channels,
            input_height=input_height,
            input_width=input_width,
            dropout_rate=dropout_rate
        )

        feat_dim = self.backbone.feat_dim  # 特征维度是512

        # 设置专家头的数量
        self.num_experts = num_experts
        self.num_classes = num_classes

        # 路由器：决定每个输入应该使用哪个专家
        self.router = Router(feat_dim, num_experts, routing_type, dropout_rate)

        # 创建多个专家头，每个头负责所有类别
        self.heads = nn.ModuleList()
        for _ in range(self.num_experts):
            head = nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),  # 添加dropout
                nn.Linear(256, num_classes)
            )
            self.heads.append(head)

    def forward(self, x):
        # 提取特征
        feat = self.backbone(x)

        # 路由决策
        routing_weights, router_logits = self.router(feat)

        # 所有专家头的输出
        expert_outputs = []
        for i, head in enumerate(self.heads):
            expert_logits = head(feat)  # [batch_size, num_classes]
            expert_outputs.append(expert_logits)

        # 合并专家输出 - 用于计算多样性损失
        stacked_experts = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, num_classes]

        # 根据路由权重加权组合专家输出
        combined_logits = 0
        batch_size = x.size(0)
        for i in range(self.num_experts):
            expert_weight = routing_weights[:, i].view(batch_size, 1)  # [batch_size, 1]
            combined_logits += expert_outputs[i] * expert_weight  # [batch_size, num_classes]

        return combined_logits, routing_weights, stacked_experts, router_logits

    def inference(self, x):
        """推理模式"""
        feat = self.backbone(x)

        # 路由决策
        routing_weights, _ = self.router(feat)

        # 所有专家头的输出
        expert_outputs = []
        for head in self.heads:
            logits = head(feat)
            expert_outputs.append(logits)

        # 根据路由权重加权组合专家输出
        combined_logits = 0
        batch_size = x.size(0)
        for i in range(self.num_experts):
            expert_weight = routing_weights[:, i].view(batch_size, 1)
            combined_logits += expert_outputs[i] * expert_weight

        return combined_logits

    def predict(self, x):
        """返回预测的类别"""
        logits = self.inference(x)
        _, predicted = torch.max(logits, 1)
        return predicted
