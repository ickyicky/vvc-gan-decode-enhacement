import os
import re
import math
import numpy as np
import cv2
import json
from tqdm import tqdm
from typing import List, Tuple, Any, Dict, Optional
from dataclasses import dataclass, asdict
from pydantic import validate_arguments
from random import choice
from pathlib import Path


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


class Splitter:
    """
    Splits videos into chunks for easy processing
    """

    INFO_HEIGHT_REGEX: str = re.compile(r"^\s*Height\s*:\s*(\d+)\s*$")
    INFO_WIDTH_REGEX: str = re.compile(r"^\s*Width\s*:\s*(\d+)\s*$")
    INFO_FRAMES_REGEX: str = re.compile(r"^\s*Frame count\s*:\s*(\d+)\s*$")
    ENCODED_REGEX: str = re.compile(
        r"^(?P<name>\d+P_\w+)_(?P<profile>AI|RA)_QP(?P<qp>\d{2})_ALF(?P<alf>\d{1})_DB(?P<db>\d{1})_SAO(?P<sao>\d{1}).yuv"
    )

    METADATA_FORMAT: str = "{name}.mkv.info"
    DECODED_FORMAT: str = "{file}_{profile}_QP{qp:d}_ALF{alf:d}_DB{db:d}_SAO{sao:d}.yuv"
    ORIGINAL_FORMAT: str = "{file}.yuv"
    FILE_FORMAT: str = "yuv"

    CHUNK_NAME = "{file}/{profile}_QP{qp:d}_ALF{alf:d}_DB{db:d}_SAO{sao:d}/{frame}_{position[0]}_{position[1]}.yuv"
    ORIG_CHUNK_NAME = "{file}/{frame}_{position[0]}_{position[1]}.yuv"

    def __init__(
        self,
        data_path: str,
        encoded_path: str,
        chunk_folder: str,
        orig_chunk_folder: str,
        chunk_width: int = 128,
        chunk_height: int = 128,
        chunk_border: int = 8,
    ) -> None:
        super().__init__()

        self.chunk_width = chunk_width
        self.chunk_height = chunk_height
        self.chunk_border = chunk_border

        self.data_path = data_path
        self.encoded_path = encoded_path

        self.chunk_folder = chunk_folder
        self.orig_chunk_folder = orig_chunk_folder

    def split_chunks(self) -> None:
        """
        splits chunks :)
        """
        files = os.listdir(self.encoded_path)

        for file in tqdm(files):
            if not file.endswith(self.FILE_FORMAT):
                continue

            metadata = self.load_metadata_for(file)

            horizontal_chunks = math.ceil(metadata.width / self.chunk_width)
            vertical_chunks = math.ceil(metadata.height / self.chunk_height)

            video_chunks = []

            for frame in range(metadata.frames):
                for h_part in range(horizontal_chunks):
                    h_pos = min(
                        (
                            h_part * (self.chunk_width - self.chunk_border),
                            metadata.width - self.chunk_width,
                        )
                    )
                    for v_part in range(vertical_chunks):
                        v_pos = min(
                            (
                                v_part * (self.chunk_height - self.chunk_border),
                                metadata.height - self.chunk_height,
                            )
                        )
                        chunk = Chunk(
                            metadata=metadata, frame=frame, position=(v_pos, h_pos)
                        )
                        video_chunks.append(chunk)

            self.save_chunks(video_chunks)

    def load_metadata_for(self, file: str) -> Metadata:
        """
        Loads metadata for given file
        """
        m = re.match(self.ENCODED_REGEX, file)
        assert m, f"Invalid file name: {file} not matching regex: {self.ENCODED_REGEX}"
        match_group = m.groupdict()

        height = width = frames = None

        with open(
            os.path.join(self.data_path, self.METADATA_FORMAT.format_map(match_group))
        ) as f:
            for line in f.readlines():
                h = re.match(self.INFO_HEIGHT_REGEX, line)
                height = h.groups()[0] if h else height
                w = re.match(self.INFO_WIDTH_REGEX, line)
                width = w.groups()[0] if w else width
                f = re.match(self.INFO_FRAMES_REGEX, line)
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

    def save_chunks(self, chunks: List[Chunk]) -> None:
        """
        Splits chunks
        """
        metadata = chunks[0].metadata
        nh = metadata.height * 3 // 2
        frame_size = metadata.width * nh

        orig_file_path = os.path.join(
            self.data_path, self.ORIGINAL_FORMAT.format_map(asdict(metadata))
        )
        file_path = os.path.join(
            self.encoded_path, self.DECODED_FORMAT.format_map(asdict(metadata))
        )

        with open(file_path, "rb") as f:
            buff = f.read()

        with open(orig_file_path, "rb") as orig_f:
            orig_buff = orig_f.read()

        for frame_num in tqdm(range(metadata.frames)):
            orig_frame = np.frombuffer(
                orig_buff[frame_num * frame_size * 2 : frame_size * 2], dtype=np.uint16
            )
            orig_frame = np.round(orig_frame / 4).astype(np.uint8)

            frame = np.frombuffer(
                buff[frame_num * frame_size : frame_size], dtype=np.uint8
            )

            frame = frame.copy()
            frame.resize((nh, metadata.width))
            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB_I420)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)

            orig_frame = orig_frame.copy()
            orig_frame.resize((nh, metadata.width))
            orig_frame = cv2.cvtColor(orig_frame, cv2.COLOR_YUV2RGB_I420)
            orig_frame = cv2.cvtColor(orig_frame, cv2.COLOR_RGB2YUV)

            for chunk in (c for c in chunks if c.frame == frame_num):
                start_h = chunk.position[0]
                start_w = chunk.position[1]

                chunk_h = self.chunk_height
                chunk_w = self.chunk_width

                chunk_name = self.CHUNK_NAME.format_map(
                    dict(**asdict(chunk.metadata), **asdict(chunk))
                )
                orig_chunk_name = self.ORIG_CHUNK_NAME.format_map(
                    dict(**asdict(chunk.metadata), **asdict(chunk))
                )

                fname = os.path.join(self.chunk_folder, chunk_name)
                folder = os.path.dirname(fname)
                Path(folder).mkdir(parents=True, exist_ok=True)

                if not os.path.exists(fname):
                    frame_chunk = frame[start_h:, start_w:, :][:chunk_h, :chunk_w, :]
                    frame_chunk = frame_chunk.transpose((2, 0, 1))

                    with open(fname, "wb") as f:
                        f.write(frame_chunk.tobytes())

                fname = os.path.join(self.orig_chunk_folder, orig_chunk_name)
                folder = os.path.dirname(fname)
                Path(folder).mkdir(parents=True, exist_ok=True)

                if not os.path.exists(fname):
                    orig_frame_chunk = orig_frame[start_h:, start_w:, :][
                        :chunk_h, :chunk_w, :
                    ]
                    orig_frame_chunk = orig_frame_chunk.transpose((2, 0, 1))

                    with open(fname, "wb") as f:
                        f.write(orig_frame_chunk.tobytes())


if __name__ == "__main__":
    import sys

    s = Splitter(*sys.argv[1:])
    s.split_chunks()
