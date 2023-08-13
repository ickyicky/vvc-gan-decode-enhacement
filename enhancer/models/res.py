import torch.nn as nn
from torch import Tensor
from pydantic import validate_arguments
from typing import Optional
from ..config import NetworkConfig, TransitionConfig


class ResLayer(nn.Module):
    """ResLayer."""

    def __init__(
        self,
        num_input_features: int,
        features: int,
        kernel_size: int = 3,
        stride_0: int = 1,
        stride: int = 1,
        padding: int = 1,
        no_bn: bool = False,
        resample: Optional[nn.Module] = None,
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
                stride=stride_0,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(features),
            nn.ReflectionPad2d(
                padding,
            ),
            nn.Conv2d(
                features,
                features,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
                bias=False,
            ),
        )

        self.resample = resample

    def forward(self, _input: Tensor) -> Tensor:
        """forward.

        :param _input:
        :type _input: Tensor
        :rtype: Tensor
        """
        output = self.model(_input)

        if self.resample is not None:
            return output + self.resample(_input)

        return output + _input


class ResBlock(nn.Module):
    """ResBlock."""

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
            ResLayer(
                num_input_features=num_input_features,
                features=features,
                kernel_size=kernel_size,
                stride_0=transition.stride,
                stride=stride,
                padding=padding,
                no_bn=no_bn,
                resample=nn.Sequential(
                    nn.Conv2d(
                        num_input_features,
                        features,
                        kernel_size=1,
                        stride=transition.stride,
                    ),
                    nn.BatchNorm2d(features),
                ),
            )
        ]

        for i in range(1, num_layers):
            layers.append(
                ResLayer(
                    num_input_features=features,
                    features=features,
                    kernel_size=kernel_size,
                    stride_0=stride,
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


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        kernel_size,
        stride,
        padding,
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.ReflectionPad2d(
                padding,
            ),
            nn.Conv2d(
                in_features,
                out_features,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(out_features),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.model(x)


class ResNet(nn.Module):
    """
    ResNet-based network structure
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
        res_blocks = []
        for i, block_conf in enumerate(config.structure.blocks):
            if block_conf.flags == "nores":
                res_blocks.append(
                    ConvLayer(
                        in_features=num_features,
                        out_features=block_conf.features,
                        kernel_size=block_conf.kernel_size,
                        padding=block_conf.padding,
                        stride=block_conf.stride,
                    )
                )
                num_features = block_conf.features
                continue

            res_blocks.append(
                ResBlock(
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

        self.res_blocks = nn.Sequential(*res_blocks)

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

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, _input: Tensor) -> Tensor:
        data = self.res_blocks(_input)

        if self.output_block is not None:
            return self.output_block(data)

        return data
