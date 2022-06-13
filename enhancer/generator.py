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
        metadata_size: int = 5,
        metadata_features: int = 1,
        init_num_features: int = 32,
        size: int = 128,
    ) -> None:
        super().__init__()

        num_blocks = int(sqrt(size / 8))
        num_features = init_num_features * (2**num_blocks)

        model = [
            EncoderBlock(
                metadata_size, num_features, kernel_size=4, stride=1, padding=0
            )
        ]
        for i in range(num_blocks):
            model.append(
                EncoderBlock(
                    num_features, num_features // 2, kernel_size=4, stride=2, padding=1
                )
            )
            num_features = num_features // 2

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


class DenseGenerator(nn.Module):
    """
    DenseNet-based generator that consists of DenseBlocks
    """

    @validate_arguments
    def __init__(
        self,
        nc: int = 3,
        size: int = 128,
        init_num_features: int = 32,
        growth_rate: int = 8,
        bn_size: int = 2,
        drop_rate: float = 0,
        memory_efficient: bool = False,
        metadata_size: int = 5,
        metadata_features: int = 1,
        up_blocks_config: List[int] = (4, 4),
        down_blocks_config: List[int] = (4, 4),
    ) -> None:
        """Construct a DenseNet-based generator

        Parameters:
            nc (int)    -- number of channels in input and output image
            size (int)  -- size of input and output image
        """

        super().__init__()
        num_features = init_num_features
        # input block, doesnt resize
        input_block = nn.Sequential(
            nn.Conv2d(
                nc + metadata_features,
                num_features,
                kernel_size=2,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
        )

        # blocks
        parts = []
        blocks_down = [[n, _Transition] for n in down_blocks_config]
        blocks_down[-1][1] = None  # no transition at end
        blocks_up = [[n, _TransitionUp] for n in up_blocks_config]
        blocks_up[-1][1] = None  # no transition at end
        blocks = blocks_down + blocks_up

        for num_layers, transition in blocks:
            parts.append(
                _DenseBlock(
                    num_layers=num_layers,
                    num_input_features=num_features,
                    bn_size=bn_size,
                    growth_rate=growth_rate,
                    drop_rate=drop_rate,
                    memory_efficient=memory_efficient,
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
        output_block = nn.Sequential(
            nn.ConvTranspose2d(num_features, nc, kernel_size=2, stride=2),
            nn.Tanh(),
        )

        self.model = nn.Sequential(
            input_block,
            nn.Sequential(*parts),
            output_block,
        )

        self.encoder = MetadataEncoder(
            metadata_size=metadata_size,
            metadata_features=metadata_features,
            init_num_features=init_num_features,
            size=size,
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
        encoded = self.encoder(metadata)
        data = torch.cat((input_, encoded), 1)
        return self.model(data)
