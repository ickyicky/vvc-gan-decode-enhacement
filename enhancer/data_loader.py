import torch
import os
import re
import math
import numpy as np
import cv2
from typing import List, Tuple, Any
from dataclasses import dataclass, asdict
from pydantic import validate_arguments


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
        data_path: str,
        encoded_path: str,
        file_format: str = "yuv",
        chunk_width: int = 128,
        chunk_height: int = 128,
        chunk_border: int = 8,
        encoded_regex: str = "^(?P<name>\d+P_\w+)_(?P<profile>AI|RA)_QP(?P<qp>\d{2})_ALF(?P<alf>\d{1})_DB(?P<db>\d{1})_SAO(?P<sao>\d{1})_rec.yuv",
        metadata_format: str = "{name}.mkv.info",
        info_height_regex: str = "^\s*Height\s*:\s*(\d+)\s*$",
        info_width_regex: str = "^\s*Width\s*:\s*(\d+)\s*$",
        info_frames_regex: str = "^\s*Frame count\s*:\s*(\d+)\s*$",
        decoded_format: str = "{file}_{profile}_QP{qp:d}_ALF{alf:d}_DB{db:d}_SAO{sao:d}_rec.yuv",
        original_format: str = "{file}.yuv",
    ) -> None:
        super().__init__()

        self.file_format = file_format
        self.chunk_width = chunk_width
        self.chunk_height = chunk_height
        self.chunk_border = chunk_border

        self.encoded_regex = re.compile(encoded_regex)
        self.metadata_format = metadata_format
        self.info_height_regex = re.compile(info_height_regex)
        self.info_width_regex = re.compile(info_width_regex)
        self.info_frames_regex = re.compile(info_frames_regex)
        self.decoded_format = decoded_format
        self.original_format = original_format

        self.data_path = data_path
        self.encoded_path = encoded_path
        self.chunks = self.load_chunks()

    def load_chunks(self) -> List[Chunk]:
        """
        Load list of avalible chunks.

        Chunk describes chunk of frame of desired size
        of each avalible video
        """
        chunks = []

        for file in os.listdir(self.encoded_path):
            if not file.endswith(self.file_format):
                continue

            metadata = self.load_metadata_for(file)

            horizontal_chunks = math.ceil(metadata.width / self.chunk_width)
            vertical_chunks = math.ceil(metadata.height / self.chunk_height)

            for h_part in range(horizontal_chunks):
                h_pos = min(
                    (h_part * self.chunk_width, metadata.width - self.chunk_width)
                )
                for v_part in range(vertical_chunks):
                    v_pos = min(
                        (
                            v_part * self.chunk_height,
                            metadata.height - self.chunk_height,
                        )
                    )
                    for frame in range(metadata.frames):
                        chunks.append(
                            Chunk(
                                metadata=metadata, frame=frame, position=(v_pos, h_pos)
                            )
                        )

        return chunks

    def load_metadata_for(self, file: str) -> Metadata:
        """
        Loads metadata for given file
        """
        m = re.match(self.encoded_regex, file)
        assert m, f"Invalid file name: {file} not matching regex: {self.encoded_regex}"
        match_group = m.groupdict()

        height = width = frames = None

        with open(
            os.path.join(self.data_path, self.metadata_format.format_map(match_group))
        ) as f:
            for line in f.readlines():
                h = re.match(self.info_height_regex, line)
                height = h.groups()[0] if h else height
                w = re.match(self.info_width_regex, line)
                width = w.groups()[0] if w else width
                f = re.match(self.info_frames_regex, line)
                frames = f.groups()[0] if f else frames

        return Metadata(
            file=match_group["name"],
            width=width,
            height=height,
            frames=frames,
            profile=match_group["profile"],
            qp=match_group["qp"],
            alf=match_group["alf"],
            db=match_group["db"],
            sao=match_group["sao"],
        )

    def load_frame_chunk(self, file_path: str, chunk: Chunk) -> Any:
        """
        Loads chunk of frame as 2d np array
        """
        nh = chunk.metadata.height * 3 // 2
        frame_size = chunk.metadata.width * nh

        with open(file_path, "rb") as f:
            f.seek(chunk.frame * frame_size)
            frame = np.frombuffer(f.read(frame_size), dtype=np.uint8)
            frame.resize((nh, chunk.metadata.width))

        if (
            chunk.position[0] == 0
            or chunk.position[1] == 0
            or chunk.position[0] + self.chunk_height >= chunk.metadata.height
            or chunk.position[1] + self.chunk_width >= chunk.metadata.width
        ):
            temp = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB_I420)
            temp = cv2.copyMakeBorder(
                temp,
                top=self.chunk_border,
                bottom=self.chunk_border,
                left=self.chunk_border,
                right=self.chunk_border,
                borderType=cv2.BORDER_REFLECT,
            )
            frame = cv2.cvtColor(temp, cv2.COLOR_RGB2YUV_I420)
            start_w = chunk.position[0]
            start_h = chunk.position[0] * 3 // 2
        else:
            start_w = chunk.position[0] - self.chunk_border
            start_h = (chunk.position[0] - self.chunk_border) * 3 // 2

        chunk_nh = (self.chunk_height + 2 * self.chunk_border) * 3 // 2
        chunk_w = self.chunk_width + 2 * self.chunk_border

        frame_chunk = frame[start_h:, start_w:][:chunk_nh, :chunk_w]
        return frame_chunk

    def load_decoded_chunk(self, chunk: Chunk) -> Tuple[Any, Any]:
        """
        Loads decoded chunk
        """
        # TODO because right now it is in 10 bit color...
        file_path = os.path.join(
            self.encoded_path, self.decoded_format.format_map(asdict(chunk.metadata))
        )
        frame_part = self.load_frame_chunk(file_path, chunk)
        return (frame_part, chunk.metadata)

    def load_original_chunk(self, chunk: Chunk) -> Tuple[Any, Any]:
        """
        Loads original chunk
        """
        file_path = os.path.join(
            self.data_path, self.original_format.format_map(asdict(chunk.metadata))
        )
        frame_part = self.load_frame_chunk(file_path, chunk)
        return (frame_part, chunk.metadata)

    def __len__(self) -> int:
        return len(self.chunks)


class VVCDecodedDataset(VVCDataset):
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        chunk = self.chunks[idx]
        return self.load_decoded_chunk(chunk)


class VVCOriginalDataset(VVCDataset):
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        chunk = self.chunks[idx]
        return self.load_original_chunk(chunk)


if __name__ == "__main__":
    import sys
    from pprint import pprint

    d = VVCOriginalDataset(*sys.argv[1:])
    pprint(d)
    len_d = len(d)
    pprint(len_d)
    import random

    idx = random.randint(0, len_d)
    pprint(d.chunks[idx])
    pprint(d[idx])
