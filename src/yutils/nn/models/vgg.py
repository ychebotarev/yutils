import torch
import torch.nn as nn

vgg_configs = {
    "A": [ (1,64), (1, 128), (2, 256), (2, 512), (2,512)],
    "B": [ (2,64), (2, 128), (2, 256), (2, 512), (2,512)],
    "D": [ (2,64), (2, 128), (3, 256), (3, 512), (3,512)],
    "E": [ (2,64), (2, 128), (4, 256), (4, 512), (4,512)],

}

def _make_layers(in_channels, segments, batch_norm=False):
    layers=[]
    for segment in segments:
        (blocks, filters) = segment
        for _ in range(blocks):
            layer= nn.Conv2d(in_channels, filters, kernel_size=3, padding=1)
            layers.append(nn.Conv2d(in_channels, filters, kernel_size=3, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(filters))
            layers.append(nn.ReLU(inplace=True))    
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

def vgg_11():
    return VGG(vgg_configs['A'])

class VGG(nn.Module):
    def __init__(self, num_classes, channels, segments, dropout: float = 0.5):
        self.features = _make_layers(channels, segments)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x
