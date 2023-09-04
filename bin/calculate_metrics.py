import glob
import os
import re
import sys

if len(sys.argv) == 2:
    ROOT = sys.argv[1]
else:
    ROOT = "enhanced"


INFO_HEIGHT_REGEX: str = re.compile(r"^\s*Height\s*:\s*(\d+)\s*$")
INFO_WIDTH_REGEX: str = re.compile(r"^\s*Width\s*:\s*(\d+)\s*$")
INFO_FRAME_RATE_REGEX: str = re.compile(r"^\s*Frame rate\s*:\s*(\d+)\.(\d+)\s*$")


def read_movie_metadata(movie):
    height, width, frame_rate = None, None, None
    info_file = f"test_data/{movie}.y4m.info"

    with open(info_file) as f:
        for line in f.readlines():
            h = re.match(INFO_HEIGHT_REGEX, line)
            height = h.groups()[0] if h else height
            w = re.match(INFO_WIDTH_REGEX, line)
            width = w.groups()[0] if w else width
            fr = re.match(INFO_FRAME_RATE_REGEX, line)
            frame_rate = fr.groups()[0] if fr else frame_rate

    return int(height), int(width), float(frame_rate)


TASKS = []


for movie in glob.glob(f"{ROOT}/*.yuv"):
    if "420" not in movie:
        continue

    height, width, frame_rate = read_movie_metadata(movie)
    for params in os.listdir(os.path.join(ROOT, movie)):
        source = f"{ROOT}/{movie}/{params}.yuv"
        target = f"{ROOT}/{movie}/{params}_420.yuv"
        if ROOT == "test_data":
            cmd = f"ffmpeg -pix_fmt yuv420p10le -s {width}x{height} -i {source} -pix_fmt yuv420p {target}"
        else:
            cmd = f"ffmpeg -pix_fmt yuv444p -s {width}x{height} -i {source} -pix_fmt yuv420p {target}"
        TASKS.append(cmd)


with open(f"{ROOT}_convert_tasks", "w") as f:
    f.write("\n".join(TASKS))
