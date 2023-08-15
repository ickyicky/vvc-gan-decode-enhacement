import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from pydantic import validate_arguments
from typing import Optional
from ..config import NetworkConfig


class ConvLayer(nn.Module):
    """ConvLayer."""

    @validate_arguments
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout: float = 0.0,
        reflect_padding: bool = True,
        prelu: bool = True,
    ) -> None:
        super().__init__()

        self.pad = None
        if reflect_padding and padding > 0:
            self.pad = nn.ReflectionPad2d(
                padding,
            )

        self.dropout = dropout

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding if not reflect_padding else 0,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(out_channels)

        if prelu:
            self.activation = nn.PReLU()
        else:
            self.activation = nn.ReLU()

    def forward(self, _input: Tensor) -> Tensor:
        """forward.

        :param _input:
        :type _input: Tensor
        :rtype: Tensor
        """
        if self.pad is not None:
            _input = self.pad(_input)

        data = self.conv(_input)
        data = self.bn(data)
        data = self.activation(data)

        if self.dropout > 0:
            data = F.dropout(data, p=self.dropout, training=self.training)

        return data


class ConvBlock(nn.Sequential):
    """ConvBlock."""

    @validate_arguments
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout: float = 0.0,
        reflect_padding: bool = True,
        prelu: bool = True,
    ) -> None:
        super().__init__()

        self.add_module(
            "conv0",
            ConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dropout=dropout,
                reflect_padding=reflect_padding,
                prelu=prelu,
            ),
        )

        for i in range(1, num_layers):
            layer = ConvLayer(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dropout=dropout,
                reflect_padding=reflect_padding,
                prelu=prelu,
            )
            self.add_module(f"conv{i}", layer)


class OutputBlock(nn.Sequential):
    """OutputBlock."""

    @validate_arguments
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout: float = 0.0,
        reflect_padding: bool = True,
        tanh: bool = False,
    ) -> None:
        super().__init__()

        if reflect_padding and padding > 0:
            self.add_module(
                "pad",
                nn.ReflectionPad2d(
                    padding,
                ),
            )

        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding if not reflect_padding else 0,
                bias=False,
            ),
        )

        if tanh:
            self.add_module(
                "tanh",
                nn.Tanh(),
            )


class Classifier(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        reflect_padding: bool = True,
        sigmoid: bool = False,
    ) -> None:
        super().__init__()

        if reflect_padding and padding > 0:
            self.add_module(
                "pad",
                nn.ReflectionPad2d(
                    padding,
                ),
            )

        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding if not reflect_padding else 0,
                bias=False,
            ),
        )

        self.add_module(
            "flatten",
            nn.Flatten(1),
        )

        if sigmoid:
            self.add_module(
                "sigmoid",
                nn.Sigmoid(),
            )


class ConvNet(nn.Sequential):
    """
    ConvNet-based network structure
    """

    @validate_arguments
    def __init__(
        self,
        config: NetworkConfig,
        initial_features: Optional[int] = None,
    ) -> None:
        super().__init__()

        num_features = initial_features or config.input_shape[2]

        for i, block_config in enumerate(config.structure.blocks):
            block = ConvBlock(
                num_layers=block_config.num_layers,
                in_channels=num_features,
                out_channels=block_config.features,
                kernel_size=block_config.kernel_size,
                stride=block_config.stride,
                padding=block_config.padding,
                dropout=block_config.dropout,
                reflect_padding=config.reflect_padding,
                prelu=config.prelu,
            )
            num_features = block_config.features
            self.add_module(f"block{i}", block)

        if config.classifier:
            self.add_module(
                "classifier",
                Classifier(
                    in_channels=num_features,
                    out_channels=config.classifier.features,
                    kernel_size=config.classifier.kernel_size,
                    stride=config.classifier.stride,
                    padding=config.classifier.padding,
                    reflect_padding=config.reflect_padding,
                    sigmoid=config.classifier.sigmoid,
                ),
            )

        if config.output_block:
            self.add_module(
                "output_block",
                OutputBlock(
                    in_channels=num_features,
                    out_channels=config.output_block.features,
                    kernel_size=config.output_block.kernel_size,
                    stride=config.output_block.stride,
                    padding=config.output_block.padding,
                    reflect_padding=config.reflect_padding,
                    tanh=config.output_block.tanh,
                ),
            )
