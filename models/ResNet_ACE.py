import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ResNet import ResNetBlock


class PartialResNetBackbone(nn.Module):
    """
    可定制的ResNet骨干网络，支持部分层的使用
    
    用于实现MoE-ACE中的共享浅层(前2个残差块)和独享深层(后两个残差块)结构
    """

    def __init__(self, block=ResNetBlock, layers=None, input_channels=1, input_height=1, input_width=1024,
                 dropout_rate=0, start_layer=0, num_layers=None, is_partial=False):
        """
        初始化可部分使用的ResNet骨干网络
        
        Args:
            block: 使用的残差块类型，默认为ResNetBlock
            layers: 每一层的块数列表，例如[2,2,2,2]表示ResNet18
            input_channels: 输入通道数
            input_height: 输入高度
            input_width: 输入宽度
            dropout_rate: dropout比率
            start_layer: 从哪一层开始，用于构建特定部分(如0表示从头开始，2表示从第三个残差块开始)
            num_layers: 要使用的层数，如果为None则使用从start_layer开始的所有层
            is_partial: 是否是部分网络(用于深层模式)
        """
        super(PartialResNetBackbone, self).__init__()

        if layers is None:
            layers = [2, 2, 2, 2]  # 默认为ResNet18的配置

        self.in_channels = 64
        self.dropout_rate = dropout_rate
        self.is_partial = is_partial
        self.start_layer = start_layer

        # 根据start_layer调整输入通道数
        in_channels_per_layer = [64, 64, 128, 256]
        if start_layer > 0 and is_partial:
            self.in_channels = in_channels_per_layer[start_layer]

        # 确定要使用的层数
        if num_layers is None:
            self.num_layers = len(layers) - start_layer  # 使用从start_layer开始的所有层
        else:
            self.num_layers = min(num_layers, len(layers) - start_layer)  # 防止越界

        # 计算输出特征维度 - 取决于构建的最后一层
        out_channels_per_layer = [64, 128, 256, 512]
        last_layer_idx = min(start_layer + self.num_layers - 1, len(out_channels_per_layer) - 1)
        self.feat_dim = out_channels_per_layer[last_layer_idx]

        # 如果是从头开始的网络，则包含初始层
        if start_layer == 0:
            # 根据输入尺寸调整初始卷积层
            if input_height == 1:
                # 使用1x7的卷积核，不改变高度维度
                self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(1, 7), stride=(1, 2), padding=(0, 3),
                                       bias=False)
            else:
                # 原始7x7卷积
                self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

            # 调整池化层以适应可能的小高度
            if input_height == 1:
                self.maxpool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
            else:
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 创建需要的ResNet层
        self.layers = nn.ModuleList()

        # 通道数和步长配置
        channels = [64, 128, 256, 512]
        strides = [1, 2, 2, 2]

        # 只创建从start_layer开始的指定数量的层
        for i in range(self.num_layers):
            layer_idx = start_layer + i
            out_channels = channels[layer_idx]
            stride = strides[layer_idx]

            layer = self._make_layer(block, out_channels, layers[layer_idx], stride=stride)
            self.layers.append(layer)

        # 平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, dropout_rate=self.dropout_rate))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, dropout_rate=self.dropout_rate))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 如果从头开始，先通过初始层
        if self.start_layer == 0:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.maxpool(x)

        # 依次通过所有配置的ResNet层
        for layer in self.layers:
            x = layer(x)

        # 最后通过平均池化层获取特征
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


def resnet18_backbone_partial(input_channels=1, input_height=1, input_width=1024,
                              dropout_rate=0, start_layer=0, num_layers=None, is_partial=False):
    """
    创建一个部分ResNet-18骨干网络，可以选择使用前几层或后几层
    
    Args:
        input_channels: 输入通道数
        input_height: 输入高度
        input_width: 输入宽度
        dropout_rate: dropout比率
        start_layer: 从哪一层开始，用于构建特定部分
        num_layers: 要使用的层数，如果为None则使用从start_layer开始的所有层
        is_partial: 是否是部分网络(用于深层模式)
        
    Returns:
        PartialResNetBackbone: 可部分使用的ResNet-18骨干网络
    """
    return PartialResNetBackbone(
        block=ResNetBlock,
        layers=[2, 2, 2, 2],  # ResNet18配置
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        dropout_rate=dropout_rate,
        start_layer=start_layer,
        num_layers=num_layers,
        is_partial=is_partial
    )


def resnet18_shared_backbone(input_channels=1, input_height=1, input_width=1024, dropout_rate=0):
    """
    创建一个共享的ResNet-18浅层(前2个残差块)骨干网络
    
    Returns:
        PartialResNetBackbone: ResNet-18前两层的骨干网络
    """
    return resnet18_backbone_partial(
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        dropout_rate=dropout_rate,
        start_layer=0,
        num_layers=2,  # 只使用前2个残差块
        is_partial=False
    )


def resnet18_expert_backbone(input_channels=1, input_height=1, input_width=1024, dropout_rate=0):
    """
    创建一个专家独享的ResNet-18深层(后2个残差块)骨干网络
    
    Returns:
        PartialResNetBackbone: ResNet-18后两层的骨干网络
    """
    return resnet18_backbone_partial(
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        dropout_rate=dropout_rate,
        start_layer=2,  # 从第3个残差块开始
        num_layers=2,  # 使用2个残差块
        is_partial=True  # 标记为部分网络
    )
