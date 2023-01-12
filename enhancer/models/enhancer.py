from typing import List

import torch
import torch.nn as nn
from math import sqrt
from torch import Tensor
from torchvision.models.densenet import _DenseBlock, _Transition
from pydantic import validate_arguments


class _TransitionUp(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.ConvTranspose2d(
            num_input_features, num_output_features, kernel_size=2, stride=2
        )


class EncoderBlock(nn.Sequential):
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

        num_features = metadata_features

        model = [
            EncoderBlock(
                metadata_size,
                metadata_features,
                kernel_size=size // 2,
                stride=1,
                padding=0,
            )
        ]

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
        init_num_features: int = 3,
        growth_rate: int = 8,
        bn_size: int = 2,
        drop_rate: float = 0,
        metadata_size: int = 6,
        metadata_features: int = 6,
        up_blocks_config: List[int] = [2, 2, 2],
        down_blocks_config: List[int] = [2, 2, 2],
    ) -> None:
        """Construct a DenseNet-based generator

        Parameters:
            nc (int)    -- number of channels in input and output image
            size (int)  -- size of input and output image
        """

        super().__init__()

        self.encoder = MetadataEncoder(
            metadata_size=metadata_size,
            metadata_features=metadata_features,
            size=size,
        )

        # blocks
        parts = []
        blocks_down = [[n, _Transition] for n in down_blocks_config]
        blocks_down[-1][1] = None  # no transition at end
        blocks_up = [[n, _TransitionUp] for n in up_blocks_config]
        blocks_up[-1][1] = None  # no transition at end

        num_features = init_num_features + metadata_features

        for num_layers, transition in blocks_down + blocks_up:
            parts.append(
                _DenseBlock(
                    num_layers=num_layers,
                    num_input_features=num_features,
                    bn_size=bn_size,
                    growth_rate=growth_rate,
                    drop_rate=drop_rate,
                )
            )
            num_features += growth_rate * num_layers
            if transition:
                parts.append(
                    transition(
                        num_features,
                        num_features // 2,
                    )
                )
                num_features = num_features // 2

        # output part
        self.output_block = nn.Sequential(
            nn.Conv2d(num_features, nc, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
        )

        self.model = nn.Sequential(*parts)

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
        encoded = self.encoder(metadata)
        data = torch.cat((input_, encoded), 1)
        data = self.model(data)
        # data = torch.cat((input_, data), 1)
        return self.output_block(data)


if __name__ == "__main__":
    from torchsummary import summary

    g = Enhancer()
    result = g(torch.rand((1, 3, 132, 132)), torch.rand((1, 6, 1, 1)))
    print(result.shape)

    summary(g.encoder, (6, 1, 1))
    summary(g, [(3, 132, 132), (6, 1, 1)])
