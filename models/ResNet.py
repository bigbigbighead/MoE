import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout_rate=0):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()  # 添加可选dropout
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()  # 添加可选dropout
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout1(out)  # 第一个激活函数后应用dropout
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)  # 第二个激活函数前应用dropout
        out += identity
        out = F.relu(out)
        return out


# 实现可自定义层数的ResNetBackbone
class ResNetBackbone(nn.Module):
    def __init__(self, block=ResNetBlock, layers=None, input_channels=1, input_height=1, input_width=1024,
                 dropout_rate=0):
        """
        可自定义每一层块数的ResNet骨干网络

        Args:
            block: 使用的残差块类型
            layers: 每一层的块数列表，例如[2,2,2,2]表示ResNet18, [3,4,6,3]表示ResNet34
            input_channels: 输入通道数
            input_height: 输入高度
            input_width: 输入宽度
            dropout_rate: dropout比率
        """
        super(ResNetBackbone, self).__init__()
        if layers is None:
            layers = [2, 2, 2, 2]
        self.in_channels = 64
        self.dropout_rate = dropout_rate

        # 根据输入尺寸调整初始卷积层
        if input_height == 1:
            # 使用1x7的卷积核，不改变高度维度
            self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(1, 7), stride=(1, 2), padding=(0, 3), bias=False)
        else:
            # 原始7x7卷积
            self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()  # 添加可选dropout层

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
        layers.append(block(self.in_channels, out_channels, stride, downsample, dropout_rate=self.dropout_rate))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, dropout_rate=self.dropout_rate))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)  # 添加dropout
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=100, input_channels=1, input_height=1, input_width=1024):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # 根据输入尺寸调整初始卷积层
        # 对于 [batch, 1024, 1, 20] 这样的输入，需要特别处理
        if input_height == 1:
            # 使用1x3的卷积核，不改变高度维度
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

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

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
        x = self.fc(x)
        return x


def resnet18(num_classes=100, input_channels=1, input_height=1, input_width=1024, dropout_rate=0):
    """
    创建一个ResNet-18模型

    Args:
        num_classes: 分类类别数
        input_channels: 输入通道数
        input_height: 输入高度
        input_width: 输入宽度
        dropout_rate: dropout比率
    """
    model = ResNet(ResNetBlock, [2, 2, 2, 2], num_classes, input_channels, input_height, input_width)
    return model


def resnet34(num_classes=100, input_channels=1, input_height=1, input_width=1024, dropout_rate=0):
    """
    创建一个ResNet-34模型

    Args:
        num_classes: 分类类别数
        input_channels: 输入通道数
        input_height: 输入高度
        input_width: 输入宽度
        dropout_rate: dropout比率
    """
    return ResNet(ResNetBlock, [3, 4, 6, 3], num_classes, input_channels, input_height, input_width)


def resnet18_backbone(input_channels=1, input_height=1, input_width=1024, dropout_rate=0):
    """
    创建一个ResNet-18骨干网络，不包含最终的分类层

    Args:
        input_channels: 输入通道数
        input_height: 输入高度
        input_width: 输入宽度
        dropout_rate: dropout比率
    
    Returns:
        ResNetBackbone: ResNet-18骨干网络
    """
    return ResNetBackbone(
        block=ResNetBlock,
        layers=[2, 2, 2, 2],  # ResNet18配置
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        dropout_rate=dropout_rate
    )


def resnet34_backbone(input_channels=1, input_height=1, input_width=1024, dropout_rate=0):
    """
    创建一个ResNet-34骨干网络，不包含最终的分类层

    Args:
        input_channels: 输入通道数
        input_height: 输入高度
        input_width: 输入宽度
        dropout_rate: dropout比率
    
    Returns:
        ResNetBackbone: ResNet-34骨干网络
    """
    return ResNetBackbone(
        block=ResNetBlock,
        layers=[3, 4, 6, 3],  # ResNet34配置
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        dropout_rate=dropout_rate
    )
