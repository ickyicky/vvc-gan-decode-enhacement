import torch.nn as nn
import torch
from torchvision.models.densenet import DenseNet


def DiscriminatorBlock(in_features, out_features, padding=1):
    return nn.Sequential(
        nn.Conv2d(
            in_features,
            out_features,
            kernel_size=4,
            stride=2,
            padding=padding,
            bias=False,
        ),
        nn.BatchNorm2d(out_features),
        nn.LeakyReLU(0.2, inplace=True),
    )


class Discriminator(nn.Module):
    def __init__(
        self,
        nc: int = 3,
        size: int = 132,
        in_features: int = 128,
    ):
        super().__init__()

        blocks = [DiscriminatorBlock(nc, in_features)]

        cur_size = size // 2
        cur_features = in_features

        while True:
            blocks.append(DiscriminatorBlock(cur_features, cur_features * 2))
            cur_features *= 2
            cur_size = cur_size // 2
            if cur_size == 4:
                break

        parts = [
            *blocks,
            nn.Conv2d(cur_features, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Flatten(),
            nn.Sigmoid(),
        ]

        self.model = nn.Sequential(
            *parts,
        )

    def forward(self, x):
        return self.model(x)


# class Discriminator(DenseNet):
#     def __init__(self):
#         super().__init__(num_classes=1)


if __name__ == "__main__":
    from torchsummary import summary

    g = Discriminator()
    random_image = torch.rand((132, 3, 132, 132))
    print(g(random_image).shape)

    # summary(g, (3, 132, 132), device="cpu")
