from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from pydantic import validate_arguments


class DenseLayer(nn.Module):
    """DenseLayer."""

    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bn_size: Optional[int] = None,
        is_up: bool = False,
    ) -> None:
        """__init__.

        :param num_input_features:
        :type num_input_features: int
        :param growth_rate:
        :type growth_rate: int
        :param kernel_size:
        :type kernel_size: int
        :param stride:
        :type stride: int
        :param padding:
        :type padding: int
        :param bn_size:
        :type bn_size: Optional[int]
        :rtype: None
        """
        super().__init__()

        parts = []
        num_features = num_input_features

        conv = nn.ConvTranspose2d if is_up else nn.Conv2d

        if bn_size is not None:
            parts = [
                nn.BatchNorm2d(num_input_features),
                nn.PReLU(),
                conv(
                    num_features,
                    bn_size * growth_rate,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
            ]
            num_features = bn_size * growth_rate

        self.model = nn.Sequential(
            *parts,
            nn.BatchNorm2d(num_features),
            nn.PReLU(),
            conv(
                num_features,
                growth_rate,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
        )

    def forward(self, _input: Tensor) -> Tensor:
        """forward.

        :param _input:
        :type _input: Tensor
        :rtype: Tensor
        """
        output = self.model(_input)
        return torch.cat((_input, output), 1)


class DenseBlock(nn.Module):
    """DenseBlock."""

    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        num_layers: int = 1,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bn_size: Optional[int] = None,
        is_up: bool = False,
    ) -> None:
        """__init__.

        :param num_input_features:
        :type num_input_features: int
        :param growth_rate:
        :type growth_rate: int
        :param layers:
        :type layers: int
        :param kernel_size:
        :type kernel_size: int
        :param stride:
        :type stride: int
        :param padding:
        :type padding: int
        :param bn_size:
        :type bn_size: Optional[int]
        :rtype: None
        """
        super().__init__()

        layers = []
        num_features = num_input_features

        for _ in range(num_layers):
            layers.append(
                DenseLayer(
                    num_input_features=num_features,
                    growth_rate=growth_rate,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bn_size=bn_size,
                    is_up=is_up,
                )
            )
            num_features += growth_rate

        self.model = nn.Sequential(
            *layers,
        )

    def forward(self, _input: Tensor) -> Tensor:
        """forward.

        :param _input:
        :type _input: Tensor
        :rtype: Tensor
        """
        return self.model(_input)


class Transition(nn.Module):
    """Transition."""

    def __init__(
        self,
        num_input_features: int,
        num_output_features: int,
        _type: str = "same",
        stride: int = 2,
    ):
        """__init__.

        :param num_input_features:
        :type num_input_features: int
        :param num_output_features:
        :type num_output_features: int
        """
        super().__init__()
        self.bn = nn.BatchNorm2d(num_input_features)
        self.conv = nn.Conv2d(
            num_input_features,
            num_output_features,
            kernel_size=1,
            bias=False,
        )
        self.relu = nn.PReLU()
        self.rescale = None

        if _type == "down":
            self.rescale = nn.AvgPool2d(kernel_size=2, stride=2)

        elif _type == "up":
            self.rescale = nn.Sequential(
                nn.BatchNorm2d(num_output_features),
                nn.PReLU(),
                nn.ConvTranspose2d(
                    num_output_features,
                    num_output_features,
                    kernel_size=2,
                    stride=stride,
                ),
            )

    def forward(self, x: Tensor) -> Tensor:
        """forward.

        :param x:
        :type x: Tensor
        :rtype: Tensor
        """
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv(out)
        if self.rescale is not None:
            out = self.rescale(out)
        return out


class EncoderBlock(nn.Sequential):
    """EncoderBlock."""

    def __init__(
        self,
        num_input_features: int,
        num_output_features: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            num_input_features,
            num_output_features,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.BatchNorm2d(num_output_features)
        self.relu = nn.ReLU(inplace=True)


class MetadataEncoder(nn.Module):
    """
    Encoder of metadata
    """

    @validate_arguments
    def __init__(
        self,
        metadata_size: int = 6,
        metadata_features: int = 6,
        size: int = 132,
    ) -> None:
        super().__init__()

        self.size = size
        num_features = metadata_features

        model = []
        model.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    num_features,
                    metadata_features,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.Tanh(),
            )
        )
        self.model = nn.Sequential(*model)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.nn.functional.interpolate(x, size=self.size // 2)
        return self.model(x)


class Enhancer(nn.Module):
    """
    DenseNet-based generator that consists of DenseBlocks
    """

    @validate_arguments
    def __init__(
        self,
        nc: int = 3,
        size: int = 132,
        metadata_size: int = 6,
        metadata_features: int = 6,
        structure=(
            (9, 4, None, 1, 128, "down"),
            (7, 3, None, 2, 64, "down"),
            (5, 2, None, 2, 32, "same"),
            (3, 1, None, 2, 16, "same"),
            (3, 1, 2, 2, 8, "up"),
            (3, 1, 2, 2, 8, "up"),
        ),
    ) -> None:
        super().__init__()

        self.metadata_encoder = MetadataEncoder(
            metadata_size=metadata_size,
            metadata_features=metadata_features,
            size=size,
        )

        num_features = nc + metadata_features

        # dense blocks
        dense_blocks = []
        for (
            kernel_size,
            padding,
            stride,
            num_layers,
            growth_rate,
            transition,
        ) in structure:
            dense_blocks.append(
                DenseBlock(
                    num_input_features=num_features,
                    growth_rate=growth_rate,
                    kernel_size=kernel_size,
                    padding=padding,
                    num_layers=num_layers,
                    is_up=transition == "up",
                )
            )
            num_features += growth_rate * num_layers

            dense_blocks.append(
                Transition(
                    num_features,
                    num_features // 2,
                    _type=transition,
                    stride=stride,
                )
            )
            num_features = num_features // 2

        self.dense_blocks = nn.Sequential(*dense_blocks)

        # output part
        self.output_block = nn.Sequential(
            nn.Conv2d(num_features, nc, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
        )

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, input_: Tensor, metadata: Tensor) -> Tensor:
        encoded_metadata = self.metadata_encoder(metadata)
        data = torch.cat((input_, encoded_metadata), 1)
        data = self.dense_blocks(data)
        return self.output_block(data)


if __name__ == "__main__":
    from torchsummary import summary

    g = Enhancer()
    result = g(torch.rand((1, 3, 132, 132)), torch.rand((1, 6, 1, 1)))
    print(result.shape)

    summary(g, [(3, 132, 132), (6, 1, 1)], device="cpu")
