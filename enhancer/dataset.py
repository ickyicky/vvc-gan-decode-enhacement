import torch
import os
import cv2
import numpy as np
from typing import List, Tuple, Any, Dict, Optional
from dataclasses import dataclass, asdict
from pydantic import validate_arguments
from glob import glob


@validate_arguments
@dataclass
class Metadata:
    file: str
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

    CHUNK_GLOB = "{folder}/*/*/*.png"
    CHUNK_NAME = "{file}/{profile}_QP{qp:d}_ALF{alf:d}_DB{db:d}_SAO{sao:d}/{frame}_{position[0]}_{position[1]}.png"
    ORIG_CHUNK_NAME = "{file}/{frame}_{position[0]}_{position[1]}.png"

    def __init__(
        self,
        chunk_folder: str,
        orig_chunk_folder: str,
        chunk_transform: Any,
        metadata_transform: Any,
        chunk_height: int = 128,
        chunk_width: int = 128,
    ) -> None:
        super().__init__()

        self.chunk_folder = chunk_folder
        self.orig_chunk_folder = orig_chunk_folder

        self.chunk_height = chunk_height
        self.chunk_width = chunk_width

        self.chunk_transform = chunk_transform
        self.metadata_transform = metadata_transform

        self.chunk_files = glob(self.CHUNK_GLOB.format(folder=self.chunk_folder))

    def get_chunk(self, fname: str) -> List[Chunk]:
        """load_chunks.
        Loads list of available chunks

        :rtype: List[Chunk]
        """
        _, fname, profiles, position = fname.split("/")
        profile, qp, alf, db, sao = profiles.split("_")
        frame, pos0, pos1 = position.split(".")[0].split("_")

        metadata = Metadata(
            file=fname,
            profile=profile,
            qp=int(qp[2:]),
            alf=bool(int(alf[3:])),
            sao=bool(int(sao[3:])),
            db=bool(int(db[2:])),
        )
        chunk = Chunk(
            position=(int(pos0), int(pos1)),
            metadata=metadata,
            frame=int(frame),
        )
        return chunk

    def load_chunk(self, chunk: Chunk) -> Tuple[Any, Any, Any]:
        chunk_path = self.CHUNK_NAME.format_map(
            dict(**asdict(chunk), **asdict(chunk.metadata))
        )
        chunk_path = os.path.join(self.chunk_folder, chunk_path)
        orig_chunk_path = self.ORIG_CHUNK_NAME.format_map(
            dict(**asdict(chunk), **asdict(chunk.metadata))
        )
        orig_chunk_path = os.path.join(self.orig_chunk_folder, orig_chunk_path)

        with open(chunk_path, "rb") as f:
            _chunk = np.frombuffer(f.read(), dtype=np.uint8)
            _chunk = np.resize(_chunk, (3, self.chunk_height, self.chunk_width))

        with open(orig_chunk_path, "rb") as f:
            orig_chunk = np.frombuffer(f.read(), dtype=np.uint8)
            orig_chunk = np.resize(orig_chunk, (3, self.chunk_height, self.chunk_width))

        return (_chunk, orig_chunk, self._metadata_to_np(chunk.metadata))

    def _metadata_to_np(self, metadata: Metadata) -> Any:
        """
        Numpy representation of metadata
        """
        return np.array(
            (
                0 if metadata.profile == "RA" else 1,
                metadata.qp / 64,
                metadata.alf,
                metadata.sao,
                metadata.db,
            )
        )

    def __len__(self) -> int:
        return len(self.chunk_files)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any]:
        chunk = self.get_chunk(self.chunk_files[idx])
        chunk, orig_chunk, metadata = self.load_chunk(chunk)
        return (
            self.chunk_transform(chunk),
            self.chunk_transform(orig_chunk),
            self.metadata_transform(metadata),
        )


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
