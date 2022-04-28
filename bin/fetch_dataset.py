#! /usr/bin/python3

# This script fetches selected data from YT-UGC dataset and saves it
# to data folder.

from tqdm import tqdm
import os
import requests
import argparse


CHUNK_SIZE = 128 * 1024  # 128 KB
DEFAULT_FOLDER = "data"
BASE_URL = "https://storage.googleapis.com/ugc-dataset/original_videos"

SELECTED_VIDEOS = [
    # "Animation/360P/Animation_360P-188f.mkv",
    # "Animation/720P/Animation_720P-620f.mkv",
    # "CoverSong/480P/CoverSong_480P-2142.mkv",
    # "CoverSong/360P/CoverSong_360P-59d1.mkv",
    # "Gaming/480P/Gaming_480P-14fc.mkv",
    # "Gaming/360P/Gaming_360P-3794.mkv",
    # "HowTo/480P/HowTo_480P-60b3.mkv",
    # "HowTo/360P/HowTo_360P-3aa6.mkv",
    # "Lecture/720P/Lecture_720P-3b7f.mkv",
    # "Lecture/360P/Lecture_360P-2d1f.mkv",
    # "LiveMusic/720P/LiveMusic_720P-0cd5.mkv",
    # "LiveMusic/480P/LiveMusic_480P-0638.mkv",
    # "LyricVideo/360P/LyricVideo_360P-0ac7.mkv",
    # "LyricVideo/360P/LyricVideo_360P-17ce.mkv",
    # "Lyrics/480P/LyricVideo_480P-5c17.mkv",
    "MusicVideo/480P/MusicVideo_480P-3aa2.mkv",
    "MusicVideo/720P/MusicVideo_720P-7bcf.mkv",
    "NewsClip/360P/NewsClip_360P-1093.mkv",
    "NewsClip/480P/NewsClip_480P-36d0.mkv",
    "NewsClip/480P/NewsClip_480P-0269.mkv",
    "Vlog/360P/Vlog_360P-3121.mkv",
    "Vlog/480P/Vlog_480P-1b39.mkv",
    "VerticalVideo/480P/VerticalVideo_480P-3a6a.mkv",
    "VerticalVideo/360P/VerticalVideo_360P-3b4e.mkv",
    "VR/720P/VR_720P-380e.mkv",
    "VR/720P/VR_720P-1dd7.mkv",
    "TelevisionClip/360P/TelevisionClip_360P-29f1.mkv",
    "TelevisionClip/720P/TelevisionClip_720P-7c06.mkv",
    "Sports/360P/Sports_360P-0dda.mkv",
    "Sports/360P/Sports_360P-32d3.mkv",
    "Sports/480P/Sports_480P-77e3.mkv",
    "Sports/480P/Sports_480P-1fe9.mkv",
]


def download_videos(target: str):
    for video in tqdm(SELECTED_VIDEOS):
        url = os.path.join(BASE_URL, video)
        category, resolution, _ = video.split("/")

        target_filename = os.path.join(target, f"{resolution}_{category}.mkv")

        with requests.get(url, stream=True) as data:
            data.raise_for_status()

            with open(target_filename, "wb") as f:
                for chunk in tqdm(
                    data.iter_content(chunk_size=CHUNK_SIZE), leave=False
                ):
                    f.write(chunk)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", action="store", default=DEFAULT_FOLDER)

    args = parser.parse_args()

    if not os.path.exists(args.dir):
        os.mkdir(args.dir)

    download_videos(args.dir)
