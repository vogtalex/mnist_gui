import torch.nn as nn


class MadryNet(nn.Module):
    def __init__(self):
        super(MadryNet, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=7*7*64, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=10)
        )

    def forward(self, x):
        s = self.layers(x)
        return s
