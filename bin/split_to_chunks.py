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
    is_intra: bool
    corner: str


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
    DECODED_LOG_FORMAT: str = (
        "{file}_{profile}_QP{qp:d}_ALF{alf:d}_DB{db:d}_SAO{sao:d}.yuv.log"
    )
    ORIGINAL_FORMAT: str = "{file}.yuv"
    FILE_FORMAT: str = "yuv"

    CHUNK_NAME = "{file}/{profile}_QP{qp:d}_ALF{alf:d}_DB{db:d}_SAO{sao:d}/{frame}_{is_intra}/{position[0]}_{position[1]}_{corner}.png"
    ORIG_CHUNK_NAME = "{file}/{frame}_{position[0]}_{position[1]}.png"

    def __init__(
        self,
        data_path: str,
        encoded_path: str,
        chunk_folder: str,
        orig_chunk_folder: str,
        done_cache: str,
        chunk_width: int = 132,
        chunk_height: int = 132,
        chunk_border: int = 2,
    ) -> None:
        super().__init__()

        self.chunk_width = chunk_width
        self.chunk_height = chunk_height
        self.chunk_border = chunk_border

        self.data_path = data_path
        self.encoded_path = encoded_path

        self.chunk_folder = chunk_folder
        self.orig_chunk_folder = orig_chunk_folder
        self.done_cache = done_cache

    def load_intra_frames(self, metadata: Metadata) -> List[int]:
        file_path = os.path.join(
            self.encoded_path, self.DECODED_LOG_FORMAT.format_map(asdict(metadata))
        )

        with open(file_path) as f:
            lines = f.read().splitlines()

        lines = [l for l in lines if l.startswith("POC")]
        return {i for i, l in enumerate(lines) if "I-SLICE" in l}

    def split_chunks(self) -> None:
        """
        splits chunks :)
        """
        files = sorted(os.listdir(self.encoded_path))

        with open(self.done_cache) as f:
            done = f.read().splitlines()

        for file in tqdm(files):
            if not file.endswith(self.FILE_FORMAT):
                continue

            if file in done:
                continue

            metadata = self.load_metadata_for(file)
            intra_frames = self.load_intra_frames(metadata)

            horizontal_chunks = math.ceil(
                metadata.width / (self.chunk_width - 2 * self.chunk_border)
            )
            vertical_chunks = math.ceil(
                metadata.height / (self.chunk_height - 2 * self.chunk_border)
            )

            video_chunks = []

            for frame in range(metadata.frames):
                for h, h_part in enumerate(range(horizontal_chunks)):
                    h_pos = h_part * (self.chunk_width - self.chunk_border * 2)

                    for v, v_part in enumerate(range(vertical_chunks)):
                        v_pos = v_part * (self.chunk_height - self.chunk_border * 2)

                        corner = []

                        if h == 0:
                            corner.append("l")
                        if h == horizontal_chunks - 1:
                            corner.append("r")
                        if v == 0:
                            corner.append("u")
                        if v == vertical_chunks - 1:
                            corner.append("b")

                        chunk = Chunk(
                            metadata=metadata,
                            frame=frame,
                            position=(v_pos, h_pos),
                            is_intra=frame in intra_frames or metadata.profile == "AI",
                            corner="".join(corner),
                        )
                        video_chunks.append(chunk)

            self.save_chunks(video_chunks)
            with open(self.done_cache, "a") as f:
                f.write(f"\n{file}")
            print(f"DONE {file}")

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

        orig_file_path = os.path.join(
            self.data_path, self.ORIGINAL_FORMAT.format_map(asdict(metadata))
        )
        file_path = os.path.join(
            self.encoded_path, self.DECODED_FORMAT.format_map(asdict(metadata))
        )

        with open(file_path, "rb") as f:
            buff = np.frombuffer(f.read(), dtype=np.uint16)

        with open(orig_file_path, "rb") as f:
            orig_buff = np.frombuffer(f.read(), dtype=np.uint8)

        buff = np.round(buff / 4).astype(np.uint8)
        buff = np.resize(buff, (metadata.frames, nh * metadata.width))

        orig_buff = np.resize(orig_buff, (metadata.frames, nh * metadata.width))

        for frame_num in tqdm(range(metadata.frames)):
            frame = buff[frame_num]
            frame = self.upsample_uv(frame, metadata.width, metadata.height)
            frame = cv2.copyMakeBorder(
                frame,
                self.chunk_border,
                2 * self.chunk_border
                + ((-metadata.height) % (self.chunk_height - 2 * self.chunk_border)),
                self.chunk_border,
                2 * self.chunk_border
                + ((-metadata.width) % (self.chunk_height - 2 * self.chunk_border)),
                cv2.BORDER_CONSTANT,
                value=0.0,
            )

            orig_frame = orig_buff[frame_num]
            orig_frame = self.upsample_uv(orig_frame, metadata.width, metadata.height)
            orig_frame = cv2.copyMakeBorder(
                orig_frame,
                self.chunk_border,
                2 * self.chunk_border
                + ((-metadata.height) % (self.chunk_height - 2 * self.chunk_border)),
                self.chunk_border,
                2 * self.chunk_border
                + ((-metadata.width) % (self.chunk_height - 2 * self.chunk_border)),
                cv2.BORDER_CONSTANT,
                value=0.0,
            )

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

                    with open(fname, "wb") as f:
                        f.write(frame_chunk.tobytes())

                fname = os.path.join(self.orig_chunk_folder, orig_chunk_name)
                folder = os.path.dirname(fname)
                Path(folder).mkdir(parents=True, exist_ok=True)

                if not os.path.exists(fname):
                    orig_frame_chunk = orig_frame[start_h:, start_w:, :][
                        :chunk_h, :chunk_w, :
                    ]

                    with open(fname, "wb") as f:
                        f.write(orig_frame_chunk.tobytes())

    def upsample_uv(self, frame_buffer, width, height):
        i = width * height
        Y = frame_buffer[0:i]
        Y = np.reshape(Y, (height, width))

        uv_size = width * height // 4
        U = frame_buffer[i : i + uv_size]
        U = np.reshape(U, (height // 2, width // 2))
        U = cv2.resize(U, (width, height))

        i += uv_size
        V = frame_buffer[i:]
        V = np.reshape(V, (height // 2, width // 2))
        V = cv2.resize(V, (width, height))

        return np.dstack([Y, U, V])


if __name__ == "__main__":
    import sys

    s = Splitter(*sys.argv[1:])
    s.split_chunks()
