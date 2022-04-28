import os
import sys


mkvs = [x for x in os.listdir("data") if x.endswith("yuv")]
profiles = ["RA", "AI"]
QP = 32, 37, 42, 47


tasks = []


for m in mkvs:
    for p in profiles:
        for q in QP:
            for p1 in (0, 1):
                for p2 in (0, 1):
                    for p3 in (0, 1):
                        tasks.append(f"bin/encode_data.sh {p} {q} {p1} {p2} {p3} {m}")

with open(sys.argv[1], "w") as f:
    f.write("\n".join(tasks))
