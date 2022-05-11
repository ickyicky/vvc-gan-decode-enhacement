import os
import sys

encoded = [x for x in os.listdir("encoded") if x.endswith("vvc")]

template = "vvdec/bin/release-static/vvdecapp -b encoded/{encoded} -o decoded/{decoded} > decoded/{decoded}.log"

tasks = []

for e in encoded:
    d = e.split(".")[0] + ".yuv"
    tasks.append(template.format(encoded=e, decoded=d))

with open(sys.argv[1], "w") as f:
    f.write("\n".join(tasks))
