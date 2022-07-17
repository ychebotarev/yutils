import torch
import torch.nn as nn


def _make_simple_block(in_features, out_features, drop_rate=0.0):
    layers = [
        nn.BatchNorm2d(in_features),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_features, out_features, kernel_size=3, padding=1, bias=False),
    ]
    if drop_rate > 0.0:
        layers.append(nn.Dropout(p=drop_rate))
    return nn.Sequential(*layers)


def _make_bottle_neck_block(in_features, out_features, drop_rate=0.0):
    inner_features = 4 * out_features
    layers = [
        nn.BatchNorm2d(in_features),
        nn.ReLU(inplace=True),
        nn.Conv2d(
            in_features, inner_features, kernel_size=1, stride=1, padding=0, bias=False
        ),
    ]
    if drop_rate > 0.0:
        layers.append(nn.Dropout(p=drop_rate))
    layers.extend(
        [
            nn.BatchNorm2d(inner_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                inner_features, out_features, kernel_size=3, padding=1, bias=False
            ),
        ]
    )
    return nn.Sequential(*layers)


def _make_transition(in_features, out_features):
    return nn.Sequential(
        nn.BatchNorm2d(in_features),
        nn.ReLU(inplace=True),
        nn.Conv2d(
            in_features, out_features, kernel_size=1, stride=1, padding=0, bias=False
        ),
        nn.AvgPool2d(kernel_size=2, stride=2),
    )


class DenseNetBlock(nn.Module):
    def __init__(
        self, num_layers, in_features, growth_rate, block_function, drop_rate=0.0
    ):

        super().__init__()
        for i in range(num_layers):
            module = block_function(
                in_features + i * growth_rate, growth_rate, drop_rate
            )
            self.add_module(f"denselayer{i + 1}", module)

    def forward(self, x):
        x = [x]
        for name, layer in self.items():
            output = layer(x)
            x.append(output)
        return torch.cat(x, 1)


class DenseNet(nn.Module):
    def __init__(
        self,
        init_features,
        growth_rate,
        block_config,
        block_function,
        drop_rate=0.0,
        num_classes=1000,
    ):
        super().__init__()

        self.preambule = nn.Sequential(
            nn.Conv2d(3, init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.features = nn.Sequential()

        num_features = init_features
        for i, num_layers in enumerate(block_config):
            block = DenseNetBlock(
                num_layers, num_features, growth_rate, block_function, drop_rate
            )
            self.features.add_module("denseblock{i + 1}", block)

            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                transition_block = _make_transition(num_features, num_features // 2)
                self.features.add_module(f"transition{i+1}", transition_block)
                num_features = num_features // 2
            else:
                self.features.add_module(nn.BatchNorm2d(num_features))
                self.features.add_module(nn.ReLU(inplace=True))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.preambule(x)
        features = self.features(x)
        out = self.avgpool(features)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def densenet121():
    return DenseNet(
        init_features=64,
        growth_rate=32,
        block_config=[6, 12, 24, 16],
        block_function=_make_bottle_neck_block,
    )
