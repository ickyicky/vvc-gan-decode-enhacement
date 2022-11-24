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
    "NewsClip/360P/NewsClip_360P-0ff8.mkv",
    "NewsClip/720P/NewsClip_720P-2182.mkv",
    "TelevisionClip/720P/TelevisionClip_720P-4edb.mkv",
    "TelevisionClip/480P/TelevisionClip_480P-415c.mkv",
    "HowTo/360P/HowTo_360P-127f.mkv",
    "HowTo/720P/HowTo_720P-7c38.mkv",
    "HowTo/1080P/HowTo_1080P-13aa.mkv",
    "Sports/1080P/Sports_1080P-3a3b.mkv",
    "Sports/720P/Sports_720P-4b69.mkv",
    "MusicVideo/720P/MusicVideo_720P-4895.mkv",
    "MusicVideo/480P/MusicVideo_480P-7955.mkv",
    "Lecture/480P/Lecture_480P-5cd7.mkv",
    "Lecture/1080P/Lecture_1080P-1b5f.mkv",
    "LiveMusic/1080P/LiveMusic_1080P-2930.mkv",
    "LiveMusic/480P/LiveMusic_480P-4f88.mkv",
]


def download_videos(target: str):
    for video in tqdm(SELECTED_VIDEOS):
        url = os.path.join(BASE_URL, video)
        category, resolution, _ = video.split("/")

        target_filename = os.path.join(target, f"{resolution}_{category}.mkv")

        if os.path.exists(target_filename):
            should_proceed = (
                input(f"file {target_filename} already exists, overwrite it? (y/n)")
                .strip()
                .upper()
            )
            if should_proceed != "Y":
                continue

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
