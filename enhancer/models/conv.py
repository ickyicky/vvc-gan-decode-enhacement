import torch
import torch.nn as nn
from torch import Tensor
from pydantic import validate_arguments
from typing import Optional
from ..config import NetworkConfig, TransitionConfig
from ..utils import weights_init


class ConvLayer(nn.Module):
    """ConvLayer."""

    def __init__(
        self,
        num_input_features: int,
        features: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        no_bn: bool = False,
    ) -> None:
        super().__init__()

        if no_bn:
            parts = [
                nn.BatchNorm2d(num_input_features),
                nn.PReLU(),
            ]
        else:
            parts = []

        num_features = num_input_features

        self.model = nn.Sequential(
            *parts,
            nn.ReflectionPad2d(
                padding,
            ),
            nn.Conv2d(
                num_features,
                features,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
                bias=False,
            ),
        )

    def forward(self, _input: Tensor) -> Tensor:
        """forward.

        :param _input:
        :type _input: Tensor
        :rtype: Tensor
        """
        return self.model(_input)


class ConvBlock(nn.Module):
    """ConvBlock."""

    def __init__(
        self,
        transition: TransitionConfig,
        num_input_features: int,
        features: int,
        num_layers: int = 1,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        no_bn: bool = False,
    ) -> None:
        """__init__.

        :param num_input_features:
        :type num_input_features: int
        :param features:
        :type features: int
        :param layers:
        :type layers: int
        :param kernel_size:
        :type kernel_size: int
        :param stride:
        :type stride: int
        :param padding:
        :type padding: int
        :rtype: None
        """
        super().__init__()

        layers = [
            ConvLayer(
                num_input_features=num_input_features,
                features=features,
                kernel_size=kernel_size,
                stride=transition.stride,
                padding=padding,
                no_bn=no_bn,
            )
        ]

        for i in range(1, num_layers):
            layers.append(
                ConvLayer(
                    num_input_features=features,
                    features=features,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    no_bn=False,
                )
            )

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


class ConvNet(nn.Module):
    """
    ConvNet-based network structure
    """

    @validate_arguments
    def __init__(
        self,
        config: NetworkConfig,
        initial_features: int,
    ) -> None:
        super().__init__()

        num_features = initial_features

        # res blocks
        blocks = []
        for i, block_conf in enumerate(config.structure.blocks):
            blocks.append(
                ConvBlock(
                    transition=block_conf.transition,
                    num_input_features=num_features,
                    features=block_conf.features,
                    kernel_size=block_conf.kernel_size,
                    padding=block_conf.padding,
                    num_layers=block_conf.num_layers,
                    no_bn=i == 0,
                )
            )
            num_features = block_conf.features

        self.blocks = nn.Sequential(*blocks)

        # output part
        self.output_block = None
        if config.no_output_block is False:
            self.output_block = nn.Sequential(
                nn.Conv2d(
                    num_features,
                    config.output_shape[2],
                    kernel_size=config.output_kernel_size,
                    stride=config.output_stride,
                    padding=config.output_padding,
                ),
            )

    def forward(self, _input: Tensor) -> Tensor:
        data = self.blocks(_input)

        if self.output_block is not None:
            return self.output_block(data)

        return data
