import torch.nn as nn


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
    ):
        super().__init__()

        blocks = [DiscriminatorBlock(nc, 128)]

        cur_size = size // 2
        cur_features = 128

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
        ]

        self.model = nn.Sequential(
            *parts,
        )

    def forward(self, x):
        result = self.model(x)
        return result

    def register_hook(self, target):
        return self.model[-3][-1].register_forward_hook(self.save_features(target))

    def save_features(self, target):
        def hook(model, input, output):
            target["features"] = output.detach()

        return hook


if __name__ == "__main__":
    from torchsummary import summary

    g = Discriminator()

    target = {}
    g.register_hook(target)
    summary(g, (3, 132, 132), device="cpu")
    print(target["features"].shape)
