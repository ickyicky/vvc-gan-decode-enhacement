import torch.nn as nn
import torch


def DiscriminatorBlock(in_features, out_features, kernel_size=4, stride=4, padding=1):
    parts = [
        nn.Conv2d(
            in_features,
            out_features,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        ),
        nn.BatchNorm2d(out_features),
        nn.LeakyReLU(0.2, inplace=True),
    ]
    return nn.Sequential(*parts)


class Discriminator(nn.Module):
    def __init__(
        self,
        nc: int = 3,
        size: int = 132,
    ):
        super().__init__()

        blocks = [DiscriminatorBlock(nc, 128, 4, 2, 1)]

        cur_size = size // 2
        cur_features = 128

        while True:
            blocks.append(
                DiscriminatorBlock(
                    cur_features,
                    cur_features * 2,
                    kernel_size=4,
                    stride=2 if cur_size > 16 else 4,
                    padding=1 if cur_size > 16 else 0,
                ),
            )
            cur_features *= 2
            cur_size = cur_size // 2 if cur_size > 16 else cur_size // 4
            if cur_size == 1:
                break

        parts = [
            nn.Conv2d(
                cur_features, 1, kernel_size=1, stride=1, padding=0, bias=False
            ),  # practically linear layer
            nn.Flatten(),
        ]

        self.blocks = nn.Sequential(*blocks)
        self.output = nn.Sequential(*parts)

    def forward(self, x):
        features = self.blocks(x)
        result = self.output(features)
        return result


class WrapperInception(nn.Module):
    def __init__(self, discriminator: Discriminator):
        super().__init__()
        self.discriminator = Discriminator()
        self.flatten = nn.Flatten()

    @torch.no_grad()
    def forward(self, x):
        features = self.flatten(self.discriminator.blocks(x))
        return features


if __name__ == "__main__":
    from torchsummary import summary

    g = Discriminator()

    target = {}
    summary(g, (3, 132, 132), device="cpu")

    g = WrapperInception(g)
    summary(g, (3, 132, 132), device="cpu")
