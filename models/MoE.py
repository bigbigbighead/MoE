import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = F.relu(out)
        return out


# 实现可自定义层数的ResNetBackbone
class ResNetBackbone(nn.Module):
    def __init__(self, block=ResNetBlock, layers=[2, 2, 2, 2], input_channels=1, input_height=1, input_width=1024):
        """
        可自定义每一层块数的ResNet骨干网络
        
        Args:
            block: 使用的残差块类型
            layers: 每一层的块数列表，例如[2,2,2,2]表示ResNet18, [3,4,6,3]表示ResNet34
            input_channels: 输入通道数
            input_height: 输入高度
            input_width: 输入宽度
        """
        super(ResNetBackbone, self).__init__()
        self.in_channels = 64

        # 根据输入尺寸调整初始卷积层
        if input_height == 1:
            # 使用1x7的卷积核，不改变高度维度
            self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(1, 7), stride=(1, 2), padding=(0, 3), bias=False)
        else:
            # 原始7x7卷积
            self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 调整池化层以适应可能的小高度
        if input_height == 1:
            self.maxpool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 创建网络的四个层，每层的块数由layers参数决定
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 输出特征维度是最后一层的通道数
        self.feat_dim = 512

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class Router(nn.Module):
    """专家路由器，决定输入应该由哪个专家处理"""

    def __init__(self, feat_dim, num_experts, routing_type='softmax'):
        super().__init__()
        self.routing_type = routing_type
        self.router = nn.Linear(feat_dim, num_experts)

    def forward(self, x):
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
                 input_width=1024):
        super().__init__()

        # 使用我们更灵活的ResNetBackbone，默认配置为ResNet18
        self.backbone = ResNetBackbone(
            block=ResNetBlock,
            layers=[2, 2, 2, 2],  # ResNet18的配置
            input_channels=input_channels,
            input_height=input_height,
            input_width=input_width
        )

        feat_dim = self.backbone.feat_dim  # 特征维度是512

        # 设置专家头的数量
        self.num_experts = num_experts
        self.num_classes = num_classes

        # 路由器：决定每个输入应该使用哪个专家
        self.router = Router(feat_dim, num_experts, routing_type)

        # 创建多个专家头，每个头负责所有类别
        self.heads = nn.ModuleList()
        for _ in range(self.num_experts):
            head = nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(),
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
