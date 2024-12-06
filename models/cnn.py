import torch
from torch import nn


class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=24, out_channels=1, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Upsample(64),
            nn.Flatten(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Unflatten(1, (64, 64)),
        )

    def forward(self, x):
        return self.cnn(x).unsqueeze(1)
