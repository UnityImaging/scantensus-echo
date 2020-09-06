import logging

import json
import os
import requests
import time
import shutil
from pathlib import Path

import multiprocessing
from functools import partial




SOURCE = "d"
NUM_PROCESSES = 6

if SOURCE == "a":
    pass
elif SOURCE == "d":
    SOURCE_TYPE = "txt"
    INPUT_DIR = Path("/mnt/Storage/scantensus-database-dicom/02/")
    LABELS_PATH = Path("/mnt/Storage/scantensus-data/validation/agg1.txt")
    OUTPUT_ROOT_DIR = Path("/mnt/Storage/scantensus-data/dicom-cache")
else:
    print(f"No Source Set")


def main():
    os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)
    if SOURCE_TYPE == "txt":
        hashes = []
        with open(LABELS_PATH, 'r') as labels_f:
            for line in labels_f:
                hashes.append(line[:-1])

    db = {}
    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:

            if not file.startswith("02-"):
                continue

            if not file.endswith(".dcm"):
                continue

            ref_file = file[:-4]
            db[ref_file] = os.path.join(root, file)

    for hash in hashes:
        out_path = OUTPUT_ROOT_DIR / f"{hash}.dcm"

        try:
            in_path = db[hash]
            shutil.copy(in_path, out_path)
            print(f"{hash} Success")
        except Exception as e:
            print(f"{hash} Failed")




if __name__ == "__main__":
    main()

