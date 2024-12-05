import torch
from torch import nn


class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=1, kernel_size=1, stride=1),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Unflatten(1, (64, 64)),
            # nn.LogSoftmax(),
            # torch.nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Flatten(),
            # nn.LogSoftmax()
            # nn.Flatten(),
            # nn.Linear(in_features=49152, out_features=4096),
        )

    def forward(self, x):
        return self.cnn(x).unsqueeze(1)


class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(in_features=49152, out_features=24576),
            # nn.ReLU(),
            # nn.Linear(in_features=24576, out_features=4096),
            nn.Linear(in_features=49152, out_features=4096),
            nn.Unflatten(1, (64, 64)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.linear(x).unsqueeze(1)


class LinearBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1),
            # nn.MaxPool2d(2, 2),
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=49152, out_features=4096),
            # nn.LogSoftmax(dim=1),
            nn.Unflatten(1, (64, 64)),
        )

    def forward(self, features):
        f = []
        for feature in features.transpose(0, 1):
            d = self.cnn(feature.unsqueeze(1))
            f.append(d)

        f = torch.cat(f, dim=1)
        return self.linear(f).unsqueeze(1)
