import glob
import sys
from utils.read_metadata import read_movie_metadata

if len(sys.argv) == 2:
    ROOT = sys.argv[1]
else:
    ROOT = "enhanced"


CMD_TEMPLATE = """ffmpeg
-pix_fmt yuv420p
-s {res}
-i {reffile}
-pix_fmt yuv420p
-s {resfile}
-i {infile}
-filter_complex "{filter}"
-f null /dev/null
> {logfile}
2>&1"""


TASKS = []


for movie in glob.glob(f"{ROOT}/*.yuv"):
    if "420" not in movie:
        continue

    # do magic here :)


with open(f"{ROOT}_calculate_metrics", "w") as f:
    f.write("\n".join(TASKS))
