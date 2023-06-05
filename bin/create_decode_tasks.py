import os
import sys

if len(sys.argv) == 4:
    enc_dir = sys.argv[2]
    dec_dir = sys.argv[3]
else:
    enc_dir = "encoded"
    dec_dir = "decoded"

encoded = [x for x in os.listdir(enc_dir) if x.endswith("vvc")]

template = "vvdec/bin/release-static/vvdecapp -b {enc_dir}/{encoded} -o {dec_dir}/{decoded} > {dec_dir}/{decoded}.log"

tasks = []

for e in encoded:
    d = e.split(".")[0] + ".yuv"
    tasks.append(
        template.format(encoded=e, decoded=d, enc_dir=enc_dir, dec_dir=dec_dir)
    )

with open(sys.argv[1], "w") as f:
    f.write("\n".join(tasks))
