import json
from pathlib import Path

import pandas as pd


ROOT = Path("/") / "Volumes" / "Matt-Data" / "Projects-Clone" / "scantensus-data"
IN_JSON_PATH = ROOT / "json" / "views.json"
OUT_CSV_PATH = ROOT / "json" / "a4c.csv"

with open(IN_JSON_PATH, "r") as read_file:
    file_data = json.load(read_file)

out_db = []

for key, data in file_data.items():
    if data['classes']['value'] == "A4CH_LV":
        out = {}
        out['FileHash'] = key# + "-0000"
        print(out)
        out_db.append(out)

out_csv = pd.DataFrame(out_db)
out_csv.to_csv(OUT_CSV_PATH, index=False)