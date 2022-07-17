import torch
import torch.nn as nn


class BaseDeep(nn.Module):
    def __init__(
        self, num_classes, channels, width, height, hidden_size1=64, hidden_size2=32
    ):
        super().__init__()
        self.num_classes = num_classes

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size1),
            nn.BatchNorm1d(hidden_size1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size1, hidden_size2),
            nn.BatchNorm1d(hidden_size2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size2, self.num_classes),
        )
        for m in self.modules():
            if not isinstance(m, torch.nn.Linear):
                continue
            m.weight.detach().normal_(0, 0.001)
            if m.bias is not None:
                m.bias.detach().zero_()

    def forward(self, x):
        x = self.model(x)
        return x
