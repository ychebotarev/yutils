import torch
import torch.nn as nn
import torch.nn.functional as F

from yutils.pytorch_nn_summary import nn_summary


class SingleInputNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(320, 50), nn.ReLU(inplace=True), nn.Linear(50, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 320)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


class Lenet5(nn.Module):
    def __init__(self, num_classes, grayscale, padding=0):
        super().__init__()

        self.learning_rate = 0.01
        self.num_classes = num_classes

        if grayscale:
            in_channels = 1
        else:
            in_channels = 3

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6 * in_channels, kernel_size=5, padding=padding),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6 * in_channels, 16 * in_channels, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5 * in_channels, 120 * in_channels),
            nn.Tanh(),
            nn.Linear(120 * in_channels, 84 * in_channels),
            nn.Tanh(),
            nn.Linear(84 * in_channels, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class MultipleInputNet(nn.Module):
    def __init__(self):
        super(MultipleInputNet, self).__init__()
        self.fc1a = nn.Linear(300, 50)
        self.fc1b = nn.Linear(50, 10)

        self.fc2a = nn.Linear(300, 50)
        self.fc2b = nn.Linear(50, 10)

    def forward(self, x1, x2):
        x1 = F.relu(self.fc1a(x1))
        x1 = self.fc1b(x1)
        x2 = F.relu(self.fc2a(x2))
        x2 = self.fc2b(x2)
        x = torch.cat((x1, x2), 0)
        return F.log_softmax(x, dim=1)


def check_result_integrity(summary_str, summary):
    print(type(summary_str) == str)
    print(type(summary) == list)
    print("name" in summary[0])
    print("output_shape" in summary[0])
    print("input_shape" in summary[0])
    print("trainable" in summary[0])


def test_single_input():
    input = (1, 28, 28)
    summary_str, summary = nn_summary(SingleInputNet(), input)
    check_result_integrity(summary_str, summary)


def test_multi_input():
    input1 = (1, 300)
    input2 = (1, 300)
    summary_str, summary = nn_summary(MultipleInputNet(), [input1, input2])
    check_result_integrity(summary_str, summary)


def test_lenet5():
    model = Lenet5(10, True)
    input = (1, 32, 32)
    summary_str, summary = nn_summary(model, input)
    check_result_integrity(summary_str, summary)
