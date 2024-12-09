import torch
from torch import nn


class Reshape(torch.nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class ConvoAE(torch.nn.Module):
    def __init__(self):
        super(ConvoAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(12, 24, 3, 1, 1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(24, 24, 3, 1, 0),  # 32 x 32 -> 30 x 30
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),
            nn.Conv2d(24, 32, 3, 2, 0),  # 30 x 30 -> 14 x 14
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),
            nn.Conv2d(32, 32, 3, 2, 0),  # 14 x 14 -> 6 x 6
            nn.Flatten(),
            nn.Linear(1152, 2),  # 1152 = 32 * 6  * 6
        )

        self.decoder = nn.Sequential(
            nn.Linear(2, 1152),
            Reshape(-1, 32, 6, 6),
            nn.ConvTranspose2d(32, 32, 3, 1, 0),  # 6 x 6 -> 8 x 8
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32, 16, 3, 2, 1),  # 8 x 8 -> 15 x 15
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(16, 16, 3, 2, 0),  # 15 x 15 -> 31 x 31
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(16, 8, 3, 1, 0),  # 31 x 31 -> 33 x 33
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(8, 1, 2, 2, 1),  # 33 x 33 -> 64 x 64
            nn.Flatten(),
            nn.Linear(in_features=4096, out_features=4096 * 2),
            nn.Unflatten(1, (2, 4096)),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
