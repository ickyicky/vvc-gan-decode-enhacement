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
    frame: int
    is_intra: bool


@validate_arguments
@dataclass
class Chunk:
    position: Tuple[int, int]
    corner: str
    metadata: Any


class VVCDataset(torch.utils.data.Dataset):
    """
    Custom DataSet loader

    It handles extracting features from each frame
    decoding YUV files etc, pretty time consuming tasks
    """

    CHUNK_GLOB = "{folder}/*/*/*/*.png"
    CHUNK_NAME = "{file}/{profile}_QP{qp:d}_ALF{alf:d}_DB{db:d}_SAO{sao:d}/{frame}_{is_intra}/{position[0]}_{position[1]}_{corner}.png"
    ORIG_CHUNK_NAME = "{file}/{frame}_{position[0]}_{position[1]}.png"

    def __init__(
        self,
        chunk_folder: str,
        orig_chunk_folder: str,
        chunk_transform: Any,
        metadata_transform: Any,
        chunk_height: int = 132,
        chunk_width: int = 132,
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
        _, fname, profiles, frame, position = fname.split("/")
        profile, qp, alf, db, sao = profiles.split("_")
        frame, is_intra = frame.split("_")
        pos0, pos1, corner = position.split(".")[0].split("_")

        metadata = Metadata(
            file=fname,
            profile=profile,
            qp=int(qp[2:]),
            alf=bool(int(alf[3:])),
            sao=bool(int(sao[3:])),
            db=bool(int(db[2:])),
            frame=int(frame),
            is_intra=is_intra == "True",
        )
        chunk = Chunk(
            position=(int(pos0), int(pos1)),
            metadata=metadata,
            corner=corner,
        )
        return chunk

    def mask_for(self, chunk: Chunk) -> Any:
        mask = np.ones((self.chunk_height, self.chunk_width))

        if "u" not in chunk.corner:
            mask[0] = 0.2
            mask[1] = 0.4
            mask[2] = 0.6
            mask[3] = 0.8

        if "b" not in chunk.corner:
            mask[-1] = 0.2
            mask[-2] = 0.4
            mask[-3] = 0.6
            mask[-4] = 0.8

        if "l" not in chunk.corner:
            mask[:, 0] = 0.2
            mask[:, 1] = np.minimum(mask[:, 1], 0.4)
            mask[:, 2] = np.minimum(mask[:, 2], 0.6)
            mask[:, 3] = np.minimum(mask[:, 3], 0.8)

        if "r" not in chunk.corner:
            mask[:, -1] = 0.2
            mask[:, -2] = np.minimum(mask[:, -2], 0.4)
            mask[:, -3] = np.minimum(mask[:, -3], 0.6)
            mask[:, -4] = np.minimum(mask[:, -4], 0.8)

        return mask

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
            _chunk = np.resize(_chunk, (self.chunk_height, self.chunk_width, 3))

        with open(orig_chunk_path, "rb") as f:
            orig_chunk = np.frombuffer(f.read(), dtype=np.uint8)
            orig_chunk = np.resize(orig_chunk, (self.chunk_height, self.chunk_width, 3))
            mask = self.mask_for(chunk)
            orig_chunk[:, :, 0] = orig_chunk[:, :, 0] * mask
            orig_chunk[:, :, 1] = orig_chunk[:, :, 1] * mask
            orig_chunk[:, :, 2] = orig_chunk[:, :, 2] * mask

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
                metadata.is_intra,
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


class OnlyOrigVVCDataset(torch.utils.data.Dataset):
    CHUNK_GLOB = "{folder}/*/*.png"
    ORIG_CHUNK_NAME = "{file}/{frame}_{position[0]}_{position[1]}.png"

    def __init__(
        self,
        chunk_folder: str,
        transform: Any,
        chunk_height: int = 132,
        chunk_width: int = 132,
    ) -> None:
        self.chunk_folder = chunk_folder
        self.chunk_files = glob(self.CHUNK_GLOB.format(folder=self.chunk_folder))

        self.transform = transform

        self.chunk_height = chunk_height
        self.chunk_width = chunk_width

    def get_chunk(self, fname: str) -> List[Chunk]:
        """load_chunks.
        Loads list of available chunks

        :rtype: List[Chunk]
        """
        _, fname, data = fname.split("/")
        frame, pos0, pos1, corner = data.split(".")[0].split("_")

        metadata = Metadata(
            file=fname,
            profile="SOME",
            qp=0,
            alf=False,
            sao=False,
            db=False,
            frame=int(frame),
            is_intra=False,
        )
        chunk = Chunk(
            position=(int(pos0), int(pos1)),
            metadata=metadata,
            corner=corner,
        )
        return chunk

    def load_chunk(self, chunk: Chunk) -> Tuple[Any, Any, Any]:
        orig_chunk_path = self.ORIG_CHUNK_NAME.format_map(
            dict(**asdict(chunk), **asdict(chunk.metadata))
        )
        orig_chunk_path = os.path.join(self.chunk_folder, orig_chunk_path)

        with open(orig_chunk_path, "rb") as f:
            orig_chunk = np.frombuffer(f.read(), dtype=np.uint8)
            orig_chunk = np.resize(orig_chunk, (self.chunk_height, self.chunk_width, 3))

        return orig_chunk

    def __len__(self) -> int:
        return len(self.chunk_files)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any]:
        chunk = self.get_chunk(self.chunk_files[idx])
        return self.transform(self.load_chunk(chunk))


if __name__ == "__main__":
    import sys
    from pprint import pprint

    transform = lambda x: x

    d = VVCDataset(*sys.argv[1:], *[transform, transform])
    pprint(d)
    len_d = len(d)
    pprint(len_d)
    import random

    idx = random.randint(0, len_d)
    pprint(d[idx])
