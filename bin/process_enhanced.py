import os
from tqdm import tqdm
import cv2
import numpy as np
import sys
from bin.utils.read_medatada import read_movie_metadata

if len(sys.argv) == 2:
    ROOT = sys.argv[1]
else:
    ROOT = "enhanced"


def mask(t: str):
    mask = np.ones((132, 132, 3), dtype=np.float32)
    a = 0.2

    if "u" not in t:
        for i in range(1, 5):
            mask[i - 1, :, :] *= a * i

    if "b" not in t:
        for i in range(1, 5):
            mask[-i, :, :] *= a * i

    if "r" not in t:
        for i in range(1, 5):
            mask[:, -i, :] *= a * i

    if "l" not in t:
        for i in range(1, 5):
            mask[:, i - 1, :] *= a * i

    return mask


for movie in tqdm(os.listdir(ROOT)):
    name, height, width, frame_rate = read_movie_metadata(movie)
    frames = [None] * 64
    for params in os.listdir(os.path.join(ROOT, movie)):
        if not os.path.isdir(os.path.join(ROOT, movie, params)):
            continue

        for frame in os.listdir(os.path.join(ROOT, movie, params)):
            if not os.path.isdir(os.path.join(ROOT, movie, params, frame)):
                continue

            image = np.zeros((height + 136, width + 136, 3), dtype=np.float32)

            for part in os.listdir(os.path.join(ROOT, movie, params, frame)):
                with open(os.path.join(ROOT, movie, params, frame, part), "rb") as f:
                    buff = np.frombuffer(f.read(), dtype=np.float32)
                    buff = np.resize(buff, (3, 132, 132))
                    buff = buff.transpose(1, 2, 0)

                    m = mask(part.split("_")[2])
                    h, w = part.split("_")[:2]
                    w, h = int(w), int(h)
                    image[h:, w:, :][:132, :132, :] += buff * m

            image = image[2:, 2:, :][:height, :width, :]
            frames[int(frame.split("_")[0])] = (image * 255).astype(np.ubyte)
            image = cv2.cvtColor(image, cv2.COLOR_YUV2RGB)
            cv2.imwrite(f"{ROOT}/{movie}/{params}/{frame}.png", image * 255)

        target = f"{ROOT}/{movie}/{params}.yuv"
        with open(target, "wb") as f:
            for frame in frames:
                Y, U, V = cv2.split(frame)
                f.write(Y.tobytes())
                f.write(V.tobytes())
                f.write(U.tobytes())

"""
 ffplay -v info -f rawvideo -pixel_format yuv444p -video_size 1920x1080 enhanced/tractor_1080p25/AI_QP32_ALF0_DB0_SAO0.yuv
"""
