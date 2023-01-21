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

        if bn_size is not None:
            parts = [
                nn.BatchNorm2d(num_input_features),
                nn.PReLU(),
                nn.Conv2d(
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
            nn.Conv2d(
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

    def __init__(self, num_input_features: int, num_output_features: int):
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

    def forward(self, x: Tensor) -> Tensor:
        """forward.

        :param x:
        :type x: Tensor
        :rtype: Tensor
        """
        out = self.bn(x)
        out = self.conv(out)
        out = self.relu(out)
        return out


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

    def forward(self, x: Tensor) -> Tensor:
        x = torch.nn.functional.interpolate(x, size=self.size)
        return x


class Enhancer(nn.Module):
    """
    DenseNet-based generator that consists of DenseBlocks
    """

    @validate_arguments
    def __init__(
        self,
        nc: int = 3,
        size: int = 132,
        init_num_features: int = 128,
        growth_rate: int = 8,
        metadata_size: int = 6,
        metadata_features: int = 6,
    ) -> None:
        super().__init__()

        self.metadata_encoder = MetadataEncoder(
            metadata_size=metadata_size,
            metadata_features=metadata_features,
            size=size,
        )

        # input encoding
        self.input_encoder = nn.Conv2d(
            nc,
            init_num_features - metadata_features,
            kernel_size=9,
            stride=1,
            padding=4,
            bias=False,
        )
        num_features = init_num_features

        # dense blocks
        # block 1, 7x7
        dense_blocks = [
            DenseBlock(
                num_input_features=num_features,
                growth_rate=growth_rate * 8,
                kernel_size=7,
                padding=3,
                num_layers=1,
            ),
        ]
        num_features += growth_rate * 8

        dense_blocks.append(
            Transition(
                num_features,
                num_features // 2,
            )
        )
        num_features = num_features // 2

        # block 2, 5x5
        dense_blocks.append(
            DenseBlock(
                num_input_features=num_features,
                growth_rate=growth_rate * 8,
                kernel_size=5,
                padding=2,
                num_layers=1,
            )
        )
        num_features += growth_rate * 8

        dense_blocks.append(
            Transition(
                num_features,
                num_features // 2,
            )
        )
        num_features = num_features // 2

        # block 3, 3x3
        dense_blocks.append(
            DenseBlock(
                num_input_features=num_features,
                growth_rate=growth_rate * 4,
                num_layers=1,
            )
        )
        num_features += growth_rate * 4

        dense_blocks.append(
            Transition(
                num_features,
                num_features // 2,
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
        encoded_input = self.input_encoder(input_)
        data = torch.cat((encoded_input, encoded_metadata), 1)
        data = self.dense_blocks(data)
        return self.output_block(data)


if __name__ == "__main__":
    from torchsummary import summary

    g = Enhancer()
    result = g(torch.rand((1, 3, 132, 132)), torch.rand((1, 6, 1, 1)))
    print(result.shape)

    summary(g.metadata_encoder, (6, 1, 1))
    summary(g, [(3, 132, 132), (6, 1, 1)])
