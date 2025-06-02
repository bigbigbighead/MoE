import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ResNet import ResNetBlock, ResNetBackbone, resnet18_backbone
from utils.data_loading_mine import log_message
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

        # 缓存计算结果的标志
        self.has_cached_computation = False
        # 使用普通属性而非buffer来存储缩放因子
        self.cached_scaling_factors = None

        # 第二阶段训练使用的整合层
        # 将所有专家的输出拼接起来，输入到全连接层
        total_expert_outputs = self.total_classes * self.num_experts  # 每个专家负责的类别数相同
        self.integrator = nn.Linear(total_expert_outputs, total_classes)

        # 第二阶段训练标志
        self.stage2_mode = False

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

        # 如果处于第二阶段模式，使用整合层
        if self.stage2_mode:
            # 拼接所有专家的输出
            concat_outputs = torch.cat(expert_outputs, dim=1)  # [batch_size, total_expert_outputs]
            # 通过整合层获得最终输出
            combined_logits = self.integrator(concat_outputs)

        # 返回特征、各专家输出和相加后或整合后的结果
        return feat, expert_outputs, combined_logits

    def precompute_scaling_factors(self):
        """
        预计算所有专家的缩放因子并存储在模型的缓存中
        这个方法应该在模型训练完成后调用一次
        """
        # 获取各专家最后一层全连接层的权重
        expert_weights = []

        # 计算每个专家的权重平方范数
        for i, expert in enumerate(self.experts):
            weight = expert.get_last_layer_weights()
            if weight is not None:
                # 计算权重的平方范数 ||w||²
                weight_norm_squared = torch.norm(weight, p=2, dim=1).pow(0.0005).mean()
                log_message(f"专家 {i} 的权重平方范数: {weight_norm_squared.item():.4f}")
                expert_weights.append(weight_norm_squared)
            else:
                # 如果无法获取权重，则使用默认值1.0
                log_message(f"专家 {i} 无法获取权重，使用默认值1.0")
                expert_weights.append(torch.tensor(1.0, device=weight.device if weight is not None else 'cpu'))

        # 使用第一个专家的权重作为参考
        reference_weight_norm = expert_weights[0]

        # 初始化缩放因子字典 - 使用普通字典而非试图注册为buffer
        scaling_factors_dict = {}

        # 计算缩放因子
        for c in range(self.total_classes):
            responsible_experts = self.class_to_experts[c]
            if len(responsible_experts) > 0:
                class_scaling_factors = []
                for expert_idx in responsible_experts:
                    scaling_factor = reference_weight_norm / expert_weights[expert_idx]
                    class_scaling_factors.append((expert_idx, scaling_factor.item()))
                scaling_factors_dict[c] = class_scaling_factors

        # 直接存储字典，不再尝试将其注册为buffer
        self.cached_scaling_factors = scaling_factors_dict
        self.has_cached_computation = True
        return scaling_factors_dict

    def inference(self, x):
        """
        推理模式：根据Model design.md中的公式计算每个类别的输出logits
        使用预计算的缩放因子来提高效率
        第二阶段训练后，使用整合层进行推理
        """
        feat = self.backbone(x)
        batch_size = x.size(0)

        # 获取每个专家的原始输出
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(feat)
            expert_outputs.append(expert_output)

        # 如果处于第二阶段模式，使用整合层
        if self.stage2_mode:
            # 拼接所有专家的输出
            concat_outputs = torch.cat(expert_outputs, dim=1)  # [batch_size, total_expert_outputs]
            # 通过整合层获得最终输出
            combined_logits = self.integrator(concat_outputs)
            return expert_outputs, combined_logits

        # 创建一个全零的输出张量
        combined_logits = torch.zeros(batch_size, self.total_classes, device=x.device)

        # 使用缓存的缩放因子(如果已经预计算)
        if not self.has_cached_computation:
            # 获取各专家最后一层全连接层的权重
            expert_weights = []

            # 计算每个专家的权重平方范数
            for expert in self.experts:
                weight = expert.get_last_layer_weights()
                if weight is not None:
                    # 计算权重的平方范数 ||w||²
                    weight_norm_squared = torch.norm(weight, p=2, dim=1).mean()
                    expert_weights.append(weight_norm_squared)
                else:
                    # 如果无法获取权重，则使用默认值1.0
                    expert_weights.append(torch.tensor(1.0, device=x.device))

            # 使用第一个专家的权重作为参考
            reference_weight_norm = expert_weights[0]

            # 为每个类别计算最终的logits输出
            for c in range(self.total_classes):
                # 获取负责该类别的所有专家索引
                responsible_experts = self.class_to_experts[c]

                if len(responsible_experts) == 0:
                    continue  # 如果没有专家负责该类别，跳过

                # 收集负责该类别的所有专家的调整后输出
                adjusted_expert_outputs = []

                for expert_idx in responsible_experts:
                    # 应用可学习权重缩放: ̂z_i = (||w_i||²/||w_1||²)·z_i
                    scaling_factor = expert_weights[expert_idx] / reference_weight_norm
                    adjusted_output = expert_outputs[expert_idx][:, c] / scaling_factor
                    adjusted_expert_outputs.append(adjusted_output)

                # 对重叠专家的输出取平均值
                if adjusted_expert_outputs:
                    # 将列表中的张量堆叠起来，然后计算均值
                    stacked_outputs = torch.stack(adjusted_expert_outputs, dim=0)
                    mean_output = torch.mean(stacked_outputs, dim=0)
                    combined_logits[:, c] = mean_output
        else:
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

        return expert_outputs, combined_logits

    def enable_stage2_mode(self):
        """启用第二阶段训练模式"""
        self.stage2_mode = True
        # 冻结backbone和所有专家的参数
        for param in self.backbone.parameters():
            param.requires_grad = False

        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad = False

        # 仅允许整��层的参数可训练
        for param in self.integrator.parameters():
            param.requires_grad = True

        log_message("已启用第二阶段训练模式，冻结backbone和专家参数，仅训练整合层")

    def disable_stage2_mode(self):
        """禁用第二阶段训练模式"""
        self.stage2_mode = False
        log_message("已禁用第二阶段训练模式，恢复原始推理方式")

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
        _, logits = self.inference(x)
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
        x = self.dropout(x)  # 在预测前��用dropout
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


class CoarseExpertModel(nn.Module):
    """按照Model design v3实现的模型，包含一个粗分类器和多个专家"""

    def __init__(self, total_classes=200, class_ranges=None,
                 input_channels=1, input_height=1, input_width=1024, dropout_rate=0):
        super().__init__()

        # 默认类别范围划分：0-99, 100-149, 150-199
        if class_ranges is None:
            class_ranges = [(0, 99), (100, 149), (150, 199)]

        self.class_ranges = class_ranges
        self.num_experts = len(class_ranges)
        self.total_classes = total_classes
        self.dropout_rate = dropout_rate

        # 创建粗分类器 - 使用ResNet18作为backbone
        self.coarse_classifier = resnet18_backbone(
            input_channels=input_channels,
            input_height=input_height,
            input_width=input_width,
            dropout_rate=dropout_rate
        )

        # 粗分类器输出层 - 输出专家索引
        self.coarse_fc = nn.Linear(self.coarse_classifier.feat_dim, self.num_experts)

        # 创建多个专家模型 - 每个专家是一个完整的ResNet18
        self.experts = nn.ModuleList()
        for expert_idx, class_range in enumerate(self.class_ranges):
            start_class, end_class = class_range
            num_classes = end_class - start_class + 1

            # 每个专家是一个完整的ResNet18模型
            expert = resnet18_backbone(
                input_channels=input_channels,
                input_height=input_height,
                input_width=input_width,
                dropout_rate=dropout_rate
            )
            # 添加专家的分类层
            expert_classifier = nn.Linear(expert.feat_dim, num_classes)

            # 将backbone和分类器组合为一个模块
            expert_model = nn.Sequential(
                expert,
                expert_classifier
            )

            self.experts.append(expert_model)

        # 为每个类别记录负责的专家索引
        self.class_to_expert = {}
        for c in range(total_classes):
            for i, (start, end) in enumerate(self.class_ranges):
                if start <= c <= end:
                    self.class_to_expert[c] = i
                    break

        # 训练阶段标识
        self.training_coarse = True  # True: 训练粗分类器，False: 训练专家

        # 缓存专家范围，用于加速推理
        self.cached_expert_ranges = [(start, end) for start, end in class_ranges]

    def forward(self, x):
        # 粗分类器输出
        coarse_feat = self.coarse_classifier(x)
        coarse_logits = self.coarse_fc(coarse_feat)

        # 如果是训练粗分类器阶段，只返回粗分类器的输出
        if self.training_coarse and self.training:
            return coarse_logits, None

        # 获取专家预测
        if self.training:
            # 训练时使用teacher forcing，即根据样本的真实标签选择专家
            # 这个逻辑需要在训练脚本中实现，这里仅返回所有专家的输出
            batch_size = x.size(0)
            expert_outputs = []
            for expert in self.experts:
                expert_output = expert(x)
                expert_outputs.append(expert_output)

            return coarse_logits, expert_outputs
        else:
            # 推理时根据粗分类器结果选择专家
            # 获取每个样本最可能的专家索引
            batch_size = x.size(0)
            _, expert_indices = torch.max(coarse_logits, dim=1)

            # 收集每个样本的专家预测结果
            final_logits = torch.zeros(batch_size, self.total_classes, device=x.device)

            # 优化：按专家批量处理，而不是单个样本
            for expert_idx in range(self.num_experts):
                # 找到应该由当前专家处理的样本索引
                expert_samples = (expert_indices == expert_idx).nonzero(as_tuple=True)[0]
                if len(expert_samples) == 0:
                    continue

                # 获取该专家负责的类别范围
                start_class, end_class = self.cached_expert_ranges[expert_idx]

                # 批量处理样本
                expert_output = self.experts[expert_idx](x[expert_samples])

                # 将专家输出填充到最终logits中的对应位置
                final_logits[expert_samples, start_class:end_class + 1] = expert_output

            return coarse_logits, final_logits

    def inference(self, x):
        """优化的推理函数，返回最终的类别预测，批量处理相同专家的样本"""
        # 粗分类器输出
        with torch.no_grad():  # 明确标记不需要梯度计算
            coarse_feat = self.coarse_classifier(x)
            coarse_logits = self.coarse_fc(coarse_feat)

            # 获取每个样本最可能的专家索引
            batch_size = x.size(0)
            _, expert_indices = torch.max(coarse_logits, dim=1)

            # 收集每个样本的专家预测结果
            final_logits = torch.zeros(batch_size, self.total_classes, device=x.device)

            # 按专家批量处理
            for expert_idx in range(self.num_experts):
                # 找到应该由当前专家处理的样本索引
                expert_samples = (expert_indices == expert_idx).nonzero(as_tuple=True)[0]
                if len(expert_samples) == 0:
                    continue

                # 获取该专家负责的类别范围
                start_class, end_class = self.cached_expert_ranges[expert_idx]

                # 批量处理样本
                expert_output = self.experts[expert_idx](x[expert_samples])

                # 将专家输出填充到最终logits中的对应位置
                final_logits[expert_samples, start_class:end_class + 1] = expert_output

        return final_logits

    def set_training_mode(self, train_coarse=True):
        """设置训练模式：训练粗分类器还是专家"""
        self.training_coarse = train_coarse

        # 根据训练模式冻结相应的参数
        if train_coarse:
            # 训练粗分类器，冻结专家参数
            for expert in self.experts:
                for param in expert.parameters():
                    param.requires_grad = False

            # 解冻粗分类器参数
            for param in self.coarse_classifier.parameters():
                param.requires_grad = True
            self.coarse_fc.weight.requires_grad = True
            self.coarse_fc.bias.requires_grad = True
        else:
            # 训练专家，冻结粗分类器参数
            for param in self.coarse_classifier.parameters():
                param.requires_grad = False
            self.coarse_fc.weight.requires_grad = False
            self.coarse_fc.bias.requires_grad = False

            # 解冻专家参数
            for expert in self.experts:
                for param in expert.parameters():
                    param.requires_grad = True

        return self

    def predict(self, x):
        """返回预测的类别"""
        final_logits = self.inference(x)
        _, predicted = torch.max(final_logits, 1)
        return predicted
