import json
from pathlib import Path

import pandas as pd

ROOT = Path("/") / "Volumes" / "Matt-Data" / "Projects-Clone" / "scantensus-data"
JSON_IN_PATH = ROOT / "json" / "02_database.json"
CSV_OUT_PATH = ROOT / "json" / "02_database.csv"

with open(JSON_IN_PATH, "r") as read_file:
    data = json.load(read_file)

out = []

for key, value in data.items():
    try:
        temp = {}
        temp['FileHash'] = key
        temp['NumberOfFrames'] = value.get('NumberOfFrames')
        temp['PatientSex'] = value.get('PatientSex')
        temp['Manufacturer'] = value.get('Manufacturer')
        temp['ManufacturerModelName'] = value.get('ManufacturerModelName')
        temp['PatientBirthYear'] = value.get('PatientBirthYear')
        temp['SeriesYear'] = value.get('SeriesYear')
        regions = value.get('SequenceOfUltrasoundRegions')
        if regions is not None:
            for region in regions:
                if int(region['PhysicalUnitsXDirection']) == 3 and int(region['PhysicalUnitsYDirection']) == 3:
                    temp['PhysicalDeltaX'] = region['PhysicalDeltaX']
                    temp['PhysicalDeltaY'] = region['PhysicalDeltaY']
                    break
                else:
                    continue
        else:
            temp['PhysicalDeltaX'] = None
            temp['PhysicalDeltaY'] = None
        value['file'] = key
        out.append(temp)
    except Exception as e:
        print(e)

out_db = pd.DataFrame(out)
out_db.to_csv(CSV_OUT_PATH)