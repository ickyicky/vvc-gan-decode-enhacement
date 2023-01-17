from torchvision.models.densenet import DenseNet


class Discriminator(DenseNet):
    def __init__(
        self,
        nc: int = 3,
        size: int = 132,
    ):
        super().__init__(
            num_init_features=nc,
            num_classes=1,
            block_config=(2, 4, 8, 6),
            bn_size=2,
        )


if __name__ == "__main__":
    from torchsummary import summary

    g = Discriminator()

    summary(g, (3, 132, 132))
