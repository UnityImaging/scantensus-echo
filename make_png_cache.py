import logging

import json
import os
import requests
import time

import multiprocessing
from functools import partial

from Scantensus.datasets.unity import UnityO

SOURCE = "b"
NUM_PROCESSES = 6

# Use the direct ip so you dont get rate limited
MAGIQUANT_ADDRESS = "89.39.141.131"
# MAGIQUANT_ADDRESS = "files.magiquant.com"

if SOURCE == "a":
    SOURCE_TYPE = "json"
    LABELS_PATH = "/Volumes/Matt-Data/Projects-Clone/scantensus-data/labels/unity/labels.json"
    OUTPUT_ROOT_DIR = "/Volumes/Matt-Data/Projects-Clone/scantensus-data/png-cache/"
elif SOURCE == "b":
    SOURCE_TYPE = "txt"
    LABELS_PATH = "/Volumes/Matt-Data/Projects-Clone/scantensus-data/validation/imp-echo-validation100-a4c.txt"
    OUTPUT_ROOT_DIR = "/Volumes/Matt-Data/Projects-Clone/scantensus-data/png-cache/"
elif SOURCE == "c":
    SOURCE_TYPE = "txt"
    LABELS_PATH = "/Volumes/Matt-Data/Projects-Clone/scantensus-data/validation/A4CH_LV.csv"
    OUTPUT_ROOT_DIR = "/Volumes/Matt-Data/Projects-Clone/scantensus-data/png-cache-a4c/"
elif SOURCE == "d":
    SOURCE_TYPE = "txt"
    LABELS_PATH = "/Volumes/Matt-Data/Projects-Clone/scantensus-data/validation/imp-echo-stowell-a4c-lv-systole.txt"
    OUTPUT_ROOT_DIR = "/Volumes/Matt-Data/Projects-Clone/scantensus-data/png-cache/"
elif SOURCE == "e":
    SOURCE_TYPE = "txt"
    LABELS_PATH = "/Volumes/Matt-Data/Projects-Clone/scantensus-data/validation/unity-plax-1.txt"
    OUTPUT_ROOT_DIR = "/Volumes/Matt-Data/Projects-Clone/scantensus-data/png-cache/"
elif SOURCE == "f":
    SOURCE_TYPE = "txt"
    LABELS_PATH = "/Volumes/Matt-Data/Projects-Clone/scantensus-data/validation/imp-echo-shunshin-plax-measure-100.txt"
    OUTPUT_ROOT_DIR = "/Volumes/Matt-Data/Projects-Clone/scantensus-data/png-cache/"
elif SOURCE == "g":
    SOURCE_TYPE = "txt"
    LABELS_PATH = "/Volumes/Matt-Data/Projects-Clone/scantensus-data/validation/unity-plax-2.txt"
    OUTPUT_ROOT_DIR = "/Volumes/Matt-Data/Projects-Clone/scantensus-data/png-cache/"
elif SOURCE == "h":
    SOURCE_TYPE = "txt"
    LABELS_PATH = "/Volumes/Matt-Data/Projects-Clone/scantensus-data/validation/imp-echo-stowell-plax-measure-train-b.txt"
    OUTPUT_ROOT_DIR = "/Volumes/Matt-Data/Projects-Clone/scantensus-data/png-cache/"
elif SOURCE == "i":
    SOURCE_TYPE = "txt"
    LABELS_PATH = "/Volumes/Matt-Data/Projects-Clone/scantensus-data/validation/a4c_zoom_all.txt"
    OUTPUT_ROOT_DIR = "/Volumes/Matt-Data/Projects-Clone/scantensus-data/png-cache/"
elif SOURCE == "j":
    SOURCE_TYPE = "txt"
    LABELS_PATH = "/Volumes/Matt-Data/Projects-Clone/scantensus-data/validation/unity-a4c-1.txt"
    LABELS_PATH = "/Volumes/Matt-Data/Projects-Clone/scantensus-data/validation/test_agg1.txt"
    OUTPUT_ROOT_DIR = "/Volumes/Matt-Data/Projects-Clone/scantensus-data/png-cache-3d/"
elif SOURCE == "k":
    SOURCE_TYPE = "txt"
    LABELS_PATH = "/Volumes/Matt-Data/Projects-Clone/scantensus-data/validation/02-validation100_a4c_filehash.txt"
    OUTPUT_ROOT_DIR = "/Volumes/Matt-Data/Projects-Clone/scantensus-data/png-cache-02-a4c/"
else:
    print(f"No Source Set")

def download_hash(file, output_root_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.info(f"{file} Starting")

    if len(file) == 67:
        hash = file
        ALL_FRAMES = True
        print(f"{file} All Frames")
    elif len(file) == 76:
        hash = file[:-9]
        ALL_FRAMES = False
        frame_num = int(file[-8:-4])
        frame_nums = [frame_num-5, frame_num, frame_num+5]
    elif len(file) == 72:
        hash = file[:-5]
        ALL_FRAMES = False
        frame_num = int(file[-4:])
        frame_nums = [frame_num-5, frame_num, frame_num+5]
    else:
        return

    if ALL_FRAMES:
        frame_nums = range(1000)

    failed_count = 0

    for frame_num in frame_nums:
        if failed_count > 5:
            break

        sub_a = hash[:2]
        sub_b = hash[3:5]
        sub_c = hash[5:7]

        file_name = f"{hash}-{frame_num:04}.png"
        location = f"http://{MAGIQUANT_ADDRESS}/scantensus-database-png-flat/{sub_a}/{sub_b}/{sub_c}/{file_name}"
        output_dir = os.path.join(output_root_dir, sub_a, sub_b, sub_c)
        output_path = os.path.join(output_dir, file_name)

        os.makedirs(output_dir, exist_ok=True)

        logger.warning(f"{location} Downloading")
        response = requests.get(location)

        if response.status_code == 200:
            with open(output_path, 'wb') as outfile:
                outfile.write(response.content)
        else:
            failed_count = failed_count + 1
            logger.warning(f"{location} Fail {response.status_code}")

        time.sleep(0.5)


def main():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if SOURCE_TYPE == "json":
        with open(LABELS_PATH, 'r') as json_f:
            db_raw = json.load(json_f)

        hashes = list(db_raw.keys())

    if SOURCE_TYPE == "txt":
        hashes = []
        with open(LABELS_PATH, 'r') as labels_f:
            for line in labels_f:
                hashes.append(line[:-1])

    pool = multiprocessing.Pool(NUM_PROCESSES)
    pool.map(partial(download_hash, output_root_dir=OUTPUT_ROOT_DIR), hashes)
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
