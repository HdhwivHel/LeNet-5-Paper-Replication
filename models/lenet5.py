import torch
import torch.nn as nn


class ScaledTanh(nn.Module):
    def forward(self, x):
        return 1.7159 * torch.tanh((2 / 3) * x)


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            ScaledTanh(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, kernel_size=5),
            ScaledTanh(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(16, 120, kernel_size=5),
            ScaledTanh(),
            nn.Flatten(),
            nn.Linear(120, 84),
            ScaledTanh(),
        )

        self.classifier = nn.Linear(84, 10)

    def forward(self, x):
        x = self.network(x)
        x = self.classifier(x)
        return x
