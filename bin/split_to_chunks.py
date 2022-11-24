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


class VVCDataset:
    """
    Custom DataSet loader

    It handles extracting features from each frame
    decoding YUV files etc, pretty time consuming tasks
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

    CHUNK_NAME = "{file}_{frame}_{position[0]}_{position[1]}.npy"

    def __init__(
        self,
        data_path: str,
        encoded_path: str,
        chunk_folder: str,
        orig_chunk_folder: str,
        metadata_folder: str,
        chunk_width: int = 128,
        chunk_height: int = 128,
        chunk_border: int = 8,
    ) -> None:
        super().__init__()

        self.chunk_width = chunk_width
        self.chunk_height = chunk_height
        self.chunk_border = chunk_border
        self.chunk_actual_width = chunk_width + chunk_border
        self.chunk_actual_height = chunk_height + chunk_border

        self.data_path = data_path
        self.encoded_path = encoded_path
        self.chunks = self.load_chunks()

        self.chunk_folder = chunk_folder
        self.orig_chunk_folder = orig_chunk_folder
        self.metadata_folder = metadata_folder

    def load_chunks(self) -> List[Chunk]:
        """
        Load list of avalible chunks.

        Chunk describes chunk of frame of desired size
        of each avalible video
        """
        chunks = []
        files = os.listdir(self.encoded_path)

        for file in tqdm(files):
            if not file.endswith(self.FILE_FORMAT):
                continue

            metadata = self.load_metadata_for(file)

            horizontal_chunks = math.ceil(metadata.width / self.chunk_actual_width)
            vertical_chunks = math.ceil(metadata.height / self.chunk_actual_height)

            for h_part in range(horizontal_chunks):
                h_pos = min(
                    (
                        h_part * self.chunk_actual_width,
                        metadata.width - self.chunk_width,
                    )
                )
                for v_part in range(vertical_chunks):
                    v_pos = min(
                        (
                            v_part * self.chunk_actual_height,
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

    def load_frame_chunk(
        self, file_path: str, chunk: Chunk, bit10: bool = False
    ) -> Any:
        """
        Loads chunk of frame as 2d np array
        """
        nh = chunk.metadata.height * 3 // 2
        frame_size = chunk.metadata.width * nh

        with open(file_path, "rb") as f:
            if bit10:
                f.seek(chunk.frame * frame_size * 2)
                frame = np.frombuffer(f.read(frame_size * 2), dtype=np.uint16)
                frame = np.round(frame / 4).astype(np.uint8)
            else:
                f.seek(chunk.frame * frame_size)
                frame = np.frombuffer(f.read(frame_size), dtype=np.uint8)

        frame = frame.copy()
        frame.resize((nh, chunk.metadata.width))
        frame = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB_I420)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)

        start_h = chunk.position[0]
        start_w = chunk.position[1]

        chunk_h = self.chunk_height
        chunk_w = self.chunk_width

        frame_chunk = frame[start_h:, start_w:, :][:chunk_h, :chunk_w, :]
        frame_chunk = frame_chunk.transpose((2, 0, 1))
        return frame_chunk

    def load_chunk(self, chunk: Chunk) -> Tuple[Any, Any, Any]:
        """
        Loads original chunk
        """
        orig_frame_part = self.load_orig_part(chunk)
        frame_part = self.load_decoded_part(chunk)
        return (frame_part, orig_frame_part, chunk.metadata)

    def _load_part(
        self, chunk: Chunk, path_: str, format_template: str, bit10=False
    ) -> Any:
        path = os.path.join(path_, format_template.format_map(asdict(chunk.metadata)))
        part = self.load_frame_chunk(path, chunk, bit10)
        return part

    def load_orig_part(self, chunk: Chunk) -> Any:
        return self._load_part(chunk, self.data_path, self.ORIGINAL_FORMAT, False)

    def load_decoded_part(self, chunk: Chunk) -> Any:
        return self._load_part(chunk, self.encoded_path, self.DECODED_FORMAT, True)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        chunk = self.chunks[idx]
        return self.load_chunk(chunk)

    def save_chunk(self, chunk: Chunk) -> None:
        name = self.CHUNK_NAME.format_map(
            dict(**asdict(chunk.metadata), **asdict(chunk))
        )

        if not os.path.exists(os.path.join(self.chunk_folder, name)):
            frame_part = self.load_decoded_part(chunk)
            with open(os.path.join(self.chunk_folder, name), "wb") as f:
                np.save(f, frame_part)

        if not os.path.exists(os.path.join(self.orig_chunk_folder, name)):
            orig_frame_part = self.load_orig_part(chunk)
            with open(os.path.join(self.orig_chunk_folder, name), "wb") as f:
                np.save(f, orig_frame_part)

        if not os.path.exists(
            os.path.join(self.metadata_folder, name.replace("npy", "json"))
        ):
            with open(
                os.path.join(self.metadata_folder, name.replace("npy", "json")), "w"
            ) as f:
                json.dump(asdict(chunk), f)


if __name__ == "__main__":
    import sys
    from pprint import pprint

    d = VVCDataset(*sys.argv[1:])
    pprint(d)
    len_d = len(d)
    pprint(len_d)

    for chunk in tqdm(d.chunks):
        d.save_chunk(chunk)
