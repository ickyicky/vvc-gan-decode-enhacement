import torch
import os
import re
import math
import numpy as np
import cv2
from tqdm import tqdm
from typing import List, Tuple, Any, Dict, Optional
from dataclasses import dataclass, asdict
from pydantic import validate_arguments
from random import choice


@validate_arguments
@dataclass
class Metadata:
    file: str
    width: int
    height: int
    frames: int
    profile: str
    qp: int
    alf: bool
    sao: bool
    db: bool


@validate_arguments
@dataclass
class Chunk:
    position: Tuple[int, int]
    metadata: Any
    frame: int


class VVCDataset(torch.utils.data.Dataset):
    """
    Custom DataSet loader

    It handles extracting features from each frame
    decoding YUV files etc, pretty time consuming tasks
    """

    def __init__(
        self,
        chunk_folder: str,
        orig_chunk_folder: str,
        metadata_folder: str,
    ) -> None:
        super().__init__()

        self.chunk_folder = chunk_folder
        self.orig_chunk_folder = orig_chunk_folder
        self.metadata_folder = metadata_folder

        self.chunks = self.load_chunks()

    def _metadata_to_np(self, metadata: Metadata) -> Any:
        """
        Numpy representation of metadata
        """
        return np.array(
            (
                0 if metadata.profile == "RA" else 1,
                metadata.qp,
                metadata.alf,
                metadata.sao,
                metadata.db,
            )
        )

    def _to_torch(
        self, _input: Any, original: Any, metadata: Metadata
    ) -> Dict[Any, Any]:
        """
        Representation for torch
        """
        return (
            torch.as_tensor(_input.copy()).float().contiguous(),
            torch.as_tensor(original.copy()).float().contiguous(),
            torch.as_tensor(self._metadata_to_np(metadata))
            .float()
            .unsqueeze(-1)
            .unsqueeze(-1)
            .contiguous(),
        )

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        chunk = self.chunks[idx]
        return chunk


if __name__ == "__main__":
    import sys
    from pprint import pprint

    d = VVCDataset(*sys.argv[1:])
    pprint(d)
    len_d = len(d)
    pprint(len_d)
    import random

    idx = random.randint(0, len_d)
    pprint(d.chunks[idx])
    pprint(d[idx])
