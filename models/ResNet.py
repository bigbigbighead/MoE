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


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, input_channels=1, input_height=1, input_width=1024):
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


def resnet18(num_classes=200, input_channels=1, input_height=1, input_width=1024):
    """
    创建一个ResNet-18模型

    Args:
        num_classes: 分类类别数
        input_channels: 输入通道数
        input_height: 输入高度
        input_width: 输入宽度
    """
    return ResNet(ResNetBlock, [2, 2, 2, 2], num_classes, input_channels, input_height, input_width)


def resnet34(num_classes=200, input_channels=1, input_height=1, input_width=1024):
    return ResNet(ResNetBlock, [3, 4, 6, 3], num_classes, input_channels, input_height, input_width)
