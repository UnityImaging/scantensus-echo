import json
import glob
from pathlib import Path

SOURCE = "d"

if SOURCE == "a":
    ROOT = Path("/") / "Volumes" / "Matt-Data" / "Projects-Clone" / "scantensus-data"
    OUT_JSON_PATH = ROOT / "predictions" / "firebase.json"
    file_paths =[
        ROOT / "predictions" / "imp-echo-shunshin-plax-measure-firebase.json",
        ROOT / "predictions" / "imp-echo-stowell-plax-measure-train-b-firebase.json",
    ]
elif SOURCE == "b":
    ROOT = Path("/") / "Volumes" / "Matt-Data" / "Projects-Clone" / "scantensus-data"
    OUT_JSON_PATH = ROOT / "json" / "database.json"
    file_paths = [
        ROOT / "json" / "e_database.json",
        ROOT / "json" / "g_database.json"
    ]
elif SOURCE == "c":
    ROOT = Path("/") / "Volumes" / "Matt-Data" / "Projects-Clone" / "scantensus-data"
    OUT_JSON_PATH = ROOT / "json" / "views.json"
    file_paths = glob.glob(f"{ROOT / 'json' / 'views' / '*.json'}")

elif SOURCE == "d":
    ROOT = Path("/") / "Volumes" / "Matt-Temp" / "scantensus-database-json" / "02"
    OUT_JSON_PATH = Path("/") / "Volumes" / "Matt-Temp" / "scantensus-database-json" / "02_database.json"
    file_paths = glob.glob(f"{ROOT / '**/*.json'}", recursive=True)

else:
    raise Exception

out = {}

if False:
    for file_path in file_paths:
        print(f"Opening: {file_path}")
        with open(file_path, "r") as read_file:
            file_data = json.load(read_file)['videos']
        print(f"Merging: {file_path}")
        out = {**out, **file_data}


elif True:
    for file_path in file_paths:
        print(f"Opening: {file_path}")
        with open(file_path, "r") as read_file:
            file_data = json.load(read_file)
        print(f"Merging: {file_path}")

        FileHash = file_data['FileHash']
        del file_data['FileHash']
        del file_data['anon_code']
        del file_data['InstanceNumber']
        out[FileHash] = file_data

with open(OUT_JSON_PATH, "w") as out_file:
    json.dump(out, out_file, indent=4)

print('done')

