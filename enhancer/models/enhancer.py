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

    def __init__(
        self,
        num_input_features: int,
        num_output_features: int,
        _type: str = "same",
        kernel_size: int = 4,
        padding: int = 1,
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
                    kernel_size=kernel_size,
                    padding=padding,
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
        is_last: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            num_input_features,
            num_output_features,
            kernel_size=kernel_size,
            stride=stride,
        )
        if is_last is False:
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
        metadata_features: int = 64,
        size: int = 132,
        num_of_blocks: int = 2,
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
        initial_features: int = 64,
        metadata_size: int = 6,
        metadata_features: int = 6,
        structure=(
            (7, 3, 0, 0, 0, 1, 48, "same"),
            (5, 2, 0, 0, 0, 1, 32, "same"),
            (3, 1, 0, 0, 0, 1, 32, "same"),
            (3, 1, 0, 0, 0, 2, 16, "same"),
            (3, 1, 0, 0, 0, 1, 16, "same"),
        ),
    ) -> None:
        super().__init__()

        self.metadata_encoder = MetadataEncoder(
            metadata_size=metadata_size,
            metadata_features=metadata_features,
            size=size,
        )

        self.input_encoder = nn.Conv2d(
            nc + metadata_features,
            initial_features,
            kernel_size=9,
            padding=4,
        )

        num_features = initial_features

        # dense blocks
        dense_blocks = []
        for (
            kernel_size,
            padding,
            tr_kernel_size,
            tr_stride,
            tr_padding,
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
                )
            )
            num_features += growth_rate * num_layers

            if transition is not None:
                dense_blocks.append(
                    Transition(
                        num_features,
                        num_features // 2,
                        _type=transition,
                        kernel_size=tr_kernel_size,
                        stride=tr_stride,
                        padding=tr_padding,
                    )
                )
                num_features = num_features // 2

        self.dense_blocks = nn.Sequential(*dense_blocks)

        # output part
        self.output_block = nn.Sequential(
            nn.Conv2d(num_features, nc, kernel_size=3, stride=1, padding=1),
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
        data = self.input_encoder(data)
        data = self.dense_blocks(data)
        return self.output_block(data)


if __name__ == "__main__":
    from torchsummary import summary

    g = Enhancer()
    result = g(torch.rand((1, 3, 132, 132)), torch.rand((1, 6, 1, 1)))
    print(result.shape)

    summary(g, [(3, 132, 132), (6, 1, 1)], device="cpu")
