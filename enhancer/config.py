import os

from dataclasses import field
from argparse_dataclass import dataclass


def dir_path(path: str) -> str:
    if os.path.isdir(path):
        return path
    raise NotADirectoryError(path)


@dataclass
class Configuration:
    data: str = field(
        default_factory=lambda: dir_path("data"),
        metadata={"type": dir_path, "help": "folder with prepared data"},
    )
    encoded: str = field(
        default_factory=lambda: dir_path("encoded"),
        metadata={"type": dir_path, "help": "folder with encoded and decoded data"},
    )
