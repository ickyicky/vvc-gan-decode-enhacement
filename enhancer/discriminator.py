import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(
        self,
        nc: int = 3,
        size: int = 128,
        depth_features: int = 64,
    ):
        super().__init__()
        self.model = nn.Sequential(
            # input is (nc) x size x size
            nn.Conv2d(nc, depth_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (depth_features) x size/2 x size/2
            nn.Conv2d(ndf, depth_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(depth_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (depth_features/2) x size/4 x size/4
            nn.Conv2d(depth_features * 2, depth_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(depth_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (depth_features/2) x size/4 x size/4
            nn.Conv2d(depth_features * 4, depth_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(depth_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(depth_features * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.model(input)
