import os
import sys
import re


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

if len(sys.argv) == 3:
    r = "^(?P<name>\d+P_\w+)_(?P<profile>AI|RA)_QP(?P<qp>\d{2})_ALF(?P<alf>\d{1})_DB(?P<db>\d{1})_SAO(?P<sao>\d{1}).log"
    t = "bin/encode_data.sh {profile} {qp} {alf} {db} {sao} {name}.yuv"

    done = []
    for f in os.listdir(sys.argv[2]):
        m = re.match(r, f)

        if not m:
            continue

        with open(os.path.join(sys.argv[2], f)) as ff:
            if "finished" not in ff.read():
                continue

        d = m.groupdict()
        task = t.format_map(d)
        done.append(task)

    tasks = [t for t in tasks if t not in done]


with open(sys.argv[1], "w") as f:
    f.write("\n".join(tasks))
