import os

from dataclasses import field
from argparse_dataclass import dataclass
from typing import Optional


def dir_path(path: str) -> str:
    if os.path.isdir(path):
        return path
    raise NotADirectoryError(path)


@dataclass
class DataConfiguration:
    data_path: str = field(
        default_factory=lambda: dir_path("data"),
        metadata={"type": dir_path, "help": "folder with prepared data"},
    )
    encoded_path: str = field(
        default_factory=lambda: dir_path("encoded"),
        metadata={"type": dir_path, "help": "folder with encoded and decoded data"},
    )
    batch_size: int = field(
        default=1000,
        metadata={"type": int, "help": "Size of each batch of data"},
    )
    num_workers: int = field(
        default_factory=lambda: os.cpu_count(),
        metadata={"type": int, "help": "Number of workers for data loading"},
    )


"""
Both inheritance and variable of given subconfig
is required for flat representation in command line yet
elegant config interpretation in code
"""


@dataclass
class TrainingConfiguration(DataConfiguration):
    data: DataConfiguration = field(
        default_factory=lambda: DataConfiguration.parse_args(),
        metadata={"type": DataConfiguration, "help": "Data configuration"},
    )
    learning_rate: float = field(
        default=0.0001,
        metadata={"type": float, "help": "Learning rate"},
    )
    beta1: float = field(
        default=0.5,
        metadata={"type": float, "help": "Beta1"},
    )
    epochs: int = field(
        default=1,
        metadata={"type": int, "help": "Number of epochs for training"},
    )
