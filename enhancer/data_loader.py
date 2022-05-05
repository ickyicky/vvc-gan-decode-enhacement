import torch
import os
import re
import math
from typing import List, Tuple, Any
from dataclasses import dataclass
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


class VVCDecodedDataset(torch.utils.data.IterableDataset):
    """
    Custom DataSet loader
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

        self.data_path = data_path
        self.encoded_path = encoded_path
        self.chunks = self.load_chunks()

    def load_chunks(self) -> List[Chunk]:
        """
        Load list of avalible chunks
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
                for w_part in range(vertical_chunks):
                    w_pos = min(
                        (
                            w_part * self.chunk_height,
                            metadata.height - self.chunk_height,
                        )
                    )
                    chunks.append(Chunk(metadata=metadata, position=(w_pos, h_pos)))

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
            file=file,
            width=width,
            height=height,
            frames=frames,
            profile=match_group["profile"],
            qp=match_group["qp"],
            alf=match_group["alf"],
            db=match_group["db"],
            sao=match_group["sao"],
        )


if __name__ == "__main__":
    import sys
    from pprint import pprint

    for chunk in VVCDecodedDataset(*sys.argv[1:]).chunks:
        pprint(chunk)
