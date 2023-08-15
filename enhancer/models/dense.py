import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from pydantic import validate_arguments
from ..config import NetworkConfig, TransitionConfig, TransitionMode, BlockConfig


class DenseLayer(nn.Module):
    """DenseLayer."""

    def __init__(
        self,
        num_input_features: int,
        config: BlockConfig,
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
        :rtype: None
        """
        super().__init__()
        self.dropout = config.dropout

        self.bottleneck = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.PReLU(),
            nn.Conv2d(
                num_input_features,
                config.growth_rate * config.bn_size,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(config.growth_rate * config.bn_size),
            nn.PReLU(),
        )

        self.conv = nn.Sequential(
            nn.ReflectionPad2d(
                config.padding,
            ),
            nn.Conv2d(
                config.growth_rate * config.bn_size,
                config.growth_rate,
                kernel_size=config.kernel_size,
                stride=config.stride,
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
        bottleneck = self.bottneleck(_input)
        output = self.conv(bottleneck)

        if self.dropout > 0:
            output = F.dropout(output, p=self.dropout, training=self.training)

        return torch.cat((_input, output), 1)


class DenseBlock(nn.Module):
    """DenseBlock."""

    def __init__(
        self,
        num_input_features: int,
        config: BlockConfig,
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
        :rtype: None
        """
        super().__init__()

        layers = []
        num_features = num_input_features

        for i in range(config.num_layers):
            layers.append(
                DenseLayer(
                    num_input_features=num_features,
                    config=config,
                )
            )
            num_features += config.growth_rate

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
        conf: TransitionConfig,
        num_input_features: int,
        num_output_features: int,
    ):
        """__init__.

        :param num_input_features:
        :type num_input_features: int
        :param num_output_features:
        :type num_output_features: int
        """
        super().__init__()
        self.bn = nn.BatchNorm2d(num_input_features)
        self.relu = nn.PReLU()
        self.conv = nn.Conv2d(
            num_input_features,
            num_output_features,
            kernel_size=1,
            bias=False,
        )
        self.rescale = None

        if conf.mode == TransitionMode.down:
            self.rescale = nn.AvgPool2d(kernel_size=2, stride=2)

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


class DenseNet(nn.Module):
    """
    DenseNet-based network structure
    """

    @validate_arguments
    def __init__(
        self,
        config: NetworkConfig,
        initial_features: int,
    ) -> None:
        super().__init__()

        num_features = initial_features
        self.features = nn.Sequential(
            nn.Conv2d(initial_features, num_features, kernel_size=7, stride=1, padding=3),
        )

        # dense blocks
        dense_blocks = []
        for i, block_conf in enumerate(config.structure.blocks):
            if block_conf.flags == "nodense":
                dense_blocks.append(
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

            dense_blocks.append(
                DenseBlock(
                    num_input_features=num_features,
                    growth_rate=block_conf.features,
                    kernel_size=block_conf.kernel_size,
                    padding=block_conf.padding,
                    num_layers=block_conf.num_layers,
                    no_bn=i == 0,
                )
            )
            num_features += block_conf.features * block_conf.num_layers

            if block_conf.transition is not None:
                dense_blocks.append(
                    Transition(
                        block_conf.transition,
                        num_features,
                        num_features // 2,
                    )
                )
                num_features = num_features // 2

        self.dense_blocks = nn.Sequential(*dense_blocks)

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
        data = self.dense_blocks(_input)

        if self.output_block is not None:
            return self.output_block(data)

        return data
