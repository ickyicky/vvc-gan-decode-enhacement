import os
import re
import sys


INFO_HEIGHT_REGEX = re.compile(r"^\s*Height\s*:\s*(\d+)\s*$")
INFO_WIDTH_REGEX = re.compile(r"^\s*Width\s*:\s*(\d+)\s*$")
INFO_FRAME_RATE_REGEX = re.compile(r"^\s*Frame rate\s*:\s*(\d+)\.(\d+)\s*$")


def normalize_name(movie: str) -> str:
    name = os.path.basename(movie)
    name = name.split(".")[0]

    if "_RA_" in name:
        name = name.split("_RA_")[0]
    elif "_AI_" in name:
        name = name.split("_AI_")[0]

    return name


def read_movie_metadata(movie: str) -> tuple[str, int, int, float]:
    height, width, frame_rate = None, None, None

    name = normalize_name(movie)
    info_file = f"test_data/{name}.y4m.info"

    with open(info_file) as f:
        for line in f.readlines():
            h = re.match(INFO_HEIGHT_REGEX, line)
            height = h.groups()[0] if h else height
            w = re.match(INFO_WIDTH_REGEX, line)
            width = w.groups()[0] if w else width
            fr = re.match(INFO_FRAME_RATE_REGEX, line)
            frame_rate = fr.groups()[0] if fr else frame_rate

    assert height is not None, "Height not found"
    assert width is not None, "Width not found"
    assert frame_rate is not None, "Frame rate not found"

    return name, int(height), int(width), float(frame_rate)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        movie = sys.argv[1]
    else:
        movie = sys.stdin.readline().strip()

    name, height, width, frame_rate = read_movie_metadata(movie)
    print(f"{name} {height} {width} {frame_rate}")
