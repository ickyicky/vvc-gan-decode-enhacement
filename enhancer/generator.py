import torch.nn as nn
from pydantic import validate_arguments


class DenseLayer(nn.Module):
    @validate_arguments
    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        bn_size: int,
        drop_rate: float,
        memory_efficient: bool,
    ):
        super().__init__()


class DenseBlock(nn.Module):
    pass


class DenseNetGenerator(nn.Module):
    pass
