from typing import List

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.densenet import _DenseBlock, _Transition
from pydantic import validate_arguments


class ContextEncoder(nn.Module):
    def __init__(
        self, init_num_features: int = 5, num_features: int = 1, size: int = 128
    ):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(
                init_num_features,
                size * 2,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(size * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                size * 2,
                size,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(size * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                size,
                num_features,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.main(input)


class DenseGenerator(nn.Module):
    """
    DenseNet-based generator that consists of DenseBlocks
    """

    @validate_arguments
    def __init__(
        self,
        growth_rate: int = 32,
        init_num_features: int = 64,
        nc: int = 3,
        size: int = 128,
        blocks_config: List[int] = (4, 4, 4, 4),
        bn_size: int = 4,
        drop_rate: float = 0,
        memory_efficient: bool = False,
        metadata_size: int = 5,
        metadata_features: int = 1,
    ) -> None:
        """Construct a DenseNet-based generator

        Parameters:
            nc (int)    -- number of channels in input and output image
            size (int)  -- size of input and output image
        """

        super().__init__()
        # define model
        num_features = init_num_features
        model = [
            nn.Conv2d(nc, num_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ]

        for i, num_layers in enumerate(blocks_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            model.append(block)
            num_features = num_features + num_layers * growth_rate
            if i != len(blocks_config) - 1:
                model.append(
                    _Transition(
                        num_input_features=num_features,
                        num_output_features=num_features // 2,
                    )
                )
                num_features = num_features // 2

        model += [
            nn.Conv2d(num_features // 2, nc, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        # define encoder for metadata
        encoder = [
            ContextEncoder(metadata_size, metadata_features, size),
        ]

        # init sequential models
        self.model = nn.Sequential(*model)
        self.encoder = nn.Sequential(*encoder)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        encoded_metadata = self.encoder(y)
        data = torch.cat(x, encoded_metadata)
        result = self.model(data)
        return result
