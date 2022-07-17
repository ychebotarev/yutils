import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    def _conv3x3(in_channels, out_channels, stride=1):
        """3x3 convolution with padding and without bias"""
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

    def __init__(self, in_channels, out_channels):
        super().__init__()
        stride = 1 if in_channels == out_channels else 2
        self.block = nn.Sequential(
            ResNetBlock._conv3x3(in_channels, out_channels, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ResNetBlock._conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
        )

        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)

        out = self.block(x)
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, num_classes, segments, dropout=0.5):
        super().__init__()
        in_channels = 3
        self.preambula = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        in_cannels = 64
        self.resnet_layers = []
        for segment in segments:
            out_cannels, blocks = segment
            layer = self._make_layer(in_cannels, out_cannels, blocks)
            self.resnet_layers.append(layer)
            in_cannels = out_cannels

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_cannels, out_cannels, blocks):
        layers = []
        layers.append(ResNetBlock(in_cannels, out_cannels))
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_cannels, out_cannels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.preambula(x)

        for resnet_layer in self.resnet_layers:
            x = resnet_layer(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


resnet_configs = {"resnet18": [(64, 2), (128, 2), (256, 2), (512, 2)]}


def resnet_18(num_classes=10):
    return ResNet(num_classes, resnet_configs["resnet18"])
