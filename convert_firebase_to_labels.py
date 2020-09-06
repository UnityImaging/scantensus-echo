import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import imageio

from Scantensus.datasets.unity import UnityO
from Scantensus.utils.json import get_cols_from_firebase_project_config, get_labels_from_firebase_project_data

HOST = "matt-laptop"

###############
if HOST == "server":
    DATA_DIR = Path("/") / "mnt" / "Storage" / "scantensus-data"
    OUTPUT_DIR = Path("/") / "mnt" / "Storage" / "matt-output"
elif HOST == "matt-laptop":
    DATA_DIR = Path("/") / "Volumes" / "Matt-Data" / "Projects-Clone" / "scantensus-data"
    OUTPUT_DIR = Path("/") / "Volumes" / "Matt-Data" / "matt-output"
else:
    raise Exception

################

CSV_KEYS_IN_PATH = DATA_DIR / "resources" / "keys.csv"
MAPPING_DB_PATH = DATA_DIR / "resources" / "mapping.csv"
FIREBASE_IN_PATH = DATA_DIR / "firebase" / "scantensus-export.json"
PNG_CACHE_DIR = DATA_DIR / "png-cache"

####################

SOURCE = "d"

if SOURCE == "a":
    GROUP = 'Unity'

    PROJECTS = [
        'luxy-chambers',
        'unity-a2c-1',
        'unity-a4c-1',
        'unity-plax-1',
        'imp-echo-ao-plax',
        'unity-plax-1-b',
        'imp-echo-stowell-a4c-lv-systole',
        'imp-echo-stowell-a3c',
        'imp-echo-shunshin-a5c-lv-segment',
        'imp-echo-shunshin-plax-measure',
        'imp-echo-shunshin-a4c-lv',
        'imp-echo-stowell-plax-measure-train-b',
        'imp-echo-shunshin-plax-measure-fix-1',
        'imp-echo-shunshin-a4c-zoom-3point',
        'imp-echo-shunshin-a4c-3point-difficult',
    ]

elif SOURCE == "b":
    GROUP = 'james'

    PROJECTS = [
        'unity-a2c-1',
        'unity-a4c-1',
        'imp-echo-stowell-a4c-lv-systole',
        'imp-echo-stowell-a3c',
        'imp-echo-shunshin-a5c-lv-segment',
        'imp-echo-shunshin-a4c-lv',
        'imp-echo-shunshin-a4c-zoom-3point',
        'imp-echo-shunshin-a4c-3point-difficult',
    ]

elif SOURCE == 'c':
    GROUP = 'plax'

    PROJECTS = [
        'imp-echo-shunshin-plax-measure',
        'imp-echo-stowell-plax-measure-train-b',
    ]

elif SOURCE == 'd':
    GROUP = 'validation100-a4c-flick'

    PROJECTS = [
        "imp-echo-shunshin-validation100-a4c-flick"
    ]
else:
    raise Exception

############

LABELS_OUT_DIR = DATA_DIR / "labels" / GROUP
os.makedirs(name=LABELS_OUT_DIR, exist_ok=True)

JSON_KEYS_OUT_PATH = LABELS_OUT_DIR / "keys.json"
JSON_ALL_LABELS_OUT_PATH = LABELS_OUT_DIR / "labels-all.json"
JSON_TRAIN_LABELS_OUT_PATH = LABELS_OUT_DIR / "labels-train.json"
JSON_VAL_LABELS_OUT_PATH = LABELS_OUT_DIR / "labels-val.json"
TXT_ALL_KEYS_OUT_PATH = LABELS_OUT_DIR / "labels-all.txt"
TXT_TRAIN_KEYS_OUT_PATH = LABELS_OUT_DIR / "labels-train.txt"
TXT_VAL_KEYS_OUT_PATH = LABELS_OUT_DIR / "labels-val.txt"
CSV_ALL_LABELS_OUT_PATH = LABELS_OUT_DIR / "labels-all.csv"
CSV_ALL_USER_LABELS_OUT_PATH = LABELS_OUT_DIR / "user-labels-all.csv"

#############

if __name__ == "__main__":

    logging.info("Starting")

    keys_db_pd = pd.read_csv(CSV_KEYS_IN_PATH, index_col='name')
    keys_db_pd = keys_db_pd.loc[keys_db_pd['active'] == 'yes']
    keys_db = keys_db_pd.to_dict("index")

    with open(JSON_KEYS_OUT_PATH, 'w') as outfile:
        json.dump(keys_db, outfile, indent=4)

    mapping_db = pd.read_csv(MAPPING_DB_PATH)

    with open(FIREBASE_IN_PATH, "r") as f:
        firebase_data = json.load(f)

    project_dbs = []

    #PROJECTS = list(pd.unique(mapping_db['project']))
    print(f'Projects to be included: {PROJECTS}')

    for project in PROJECTS:

        mapping_data = {}
        for _, row in mapping_db[mapping_db['project'] == project].iterrows():
            mapping_data[row['name_old']] = row['name_new']

        project_config_data = firebase_data['fiducial'][project]['config']
        col_data = get_cols_from_firebase_project_config(config_data=project_config_data)

        print(col_data)

        project_data = firebase_data['fiducial'][project]['labels']

        db = get_labels_from_firebase_project_data(project_data=project_data,
                                                   project=project,
                                                   mapping_data=mapping_data)

        project_dbs.extend(db)

    db_all = pd.DataFrame(project_dbs)
    db_all.to_csv(CSV_ALL_USER_LABELS_OUT_PATH)

    db_final = db_all.sort_values(by=['file', 'label', 'time']).groupby(['file', 'label']).last().reset_index()

    files = list(set(db_final['file']))

    json_out = {}

    for i, file in enumerate(files):
        print(i)

        file_out = {}
        file_out['labels'] = {}

        file_db = db_final[db_final['file'] == file]

        file_projects = list(set(file_db['project']))

        for label_name, label_data in keys_db.items():

            file_label_out = {}

            main_vis = [keys_db[label_name][x] == 'yes' for x in file_projects]
            main_vis = any(main_vis)

            file_label_row = file_db[file_db['label'] == label_name]

            if len(file_label_row) == 0:
                if main_vis:
                    true_vis = 'blurred'
                else:
                    true_vis = 'off'

                file_label_out['type'] = true_vis
                file_label_out["y"] = ""
                file_label_out["x"] = ""
                file_out['labels'][label_name] = file_label_out
                continue

            else:
                label_vis = file_label_row['vis'].item()
                if not main_vis:
                    true_vis = 'off'
                else:
                    true_vis = label_vis

            if true_vis == 'off':
                file_label_out['type'] = 'off'
                file_label_out["y"] = ""
                file_label_out["x"] = ""
                file_out['labels'][label_name] = file_label_out
                continue

            if true_vis == 'blurred':
                file_label_out['type'] = 'blurred'
                file_label_out["y"] = ""
                file_label_out["x"] = ""
                file_out['labels'][label_name] = file_label_out
                continue

            file_label_out['type'] = label_data['type']
            file_label_out["y"] = file_label_row['value_y'].item()
            file_label_out["x"] = file_label_row['value_x'].item()
            #file_label_out['time'] = file_label_row['time'].item()
            #file_label_out['project'] = file_label_row['project'].item()
            file_out['labels'][label_name] = file_label_out

        json_out[file] = file_out

    for file, node in json_out.items():
        try:
            # RV Apex
            if node['labels']['curve-rv-endo']['type'] == 'curve':
                x = node['labels']['curve-rv-endo']['x']
                y = node['labels']['curve-rv-endo']['y']

                x = [float(value) for value in x.split()]
                y = [float(value) for value in y.split()]

                min_y = min(y)
                min_y_x = x[y.index(min_y)]

                node['labels']['rv-apex-endo'] = {}
                node['labels']['rv-apex-endo']['type'] = 'point'
                node['labels']['rv-apex-endo']['x'] = str(min_y_x)
                node['labels']['rv-apex-endo']['y'] = str(min_y)
            else:
                node['labels']['rv-apex-endo'] = {}
                node['labels']['rv-apex-endo']['type'] = node['labels']['curve-rv-endo']['type']
                node['labels']['rv-apex-endo']['x'] = ""
                node['labels']['rv-apex-endo']['y'] = ""

            # LV antsep-endo apex
            if node['labels']['curve-lv-antsep-endo']['type'] == 'curve':
                x = node['labels']['curve-lv-antsep-endo']['x']
                y = node['labels']['curve-lv-antsep-endo']['y']

                x = [float(value) for value in x.split()]
                y = [float(value) for value in y.split()]

                find_x = min(x)
                find_x_y = y[x.index(find_x)]

                node['labels']['lv-antsep-endo-apex'] = {}
                node['labels']['lv-antsep-endo-apex']['type'] = 'point'
                node['labels']['lv-antsep-endo-apex']['x'] = str(find_x)
                node['labels']['lv-antsep-endo-apex']['y'] = str(find_x_y)
            else:
                node['labels']['lv-antsep-endo-apex'] = {}
                node['labels']['lv-antsep-endo-apex']['type'] = node['labels']['curve-lv-antsep-endo']['type']
                node['labels']['lv-antsep-endo-apex']['x'] = ""
                node['labels']['lv-antsep-endo-apex']['y'] = ""

            # LV antsep-endo base
            if node['labels']['curve-lv-antsep-endo']['type'] == 'curve':
                x = node['labels']['curve-lv-antsep-endo']['x']
                y = node['labels']['curve-lv-antsep-endo']['y']

                x = [float(value) for value in x.split()]
                y = [float(value) for value in y.split()]

                find_x = max(x)
                find_x_y = y[x.index(find_x)]

                node['labels']['lv-antsep-endo-base'] = {}
                node['labels']['lv-antsep-endo-base']['type'] = 'point'
                node['labels']['lv-antsep-endo-base']['x'] = str(find_x)
                node['labels']['lv-antsep-endo-base']['y'] = str(find_x_y)
            else:
                node['labels']['lv-antsep-endo-base'] = {}
                node['labels']['lv-antsep-endo-base']['type'] = node['labels']['curve-lv-antsep-endo']['type']
                node['labels']['lv-antsep-endo-base']['x'] = ""
                node['labels']['lv-antsep-endo-base']['y'] = ""

            # LV antsep-rv apex
            if node['labels']['curve-lv-antsep-rv']['type'] == 'curve':
                x = node['labels']['curve-lv-antsep-rv']['x']
                y = node['labels']['curve-lv-antsep-rv']['y']

                x = [float(value) for value in x.split()]
                y = [float(value) for value in y.split()]

                find_x = min(x)
                find_x_y = y[x.index(find_x)]

                node['labels']['lv-antsep-rv-apex'] = {}
                node['labels']['lv-antsep-rv-apex']['type'] = 'point'
                node['labels']['lv-antsep-rv-apex']['x'] = str(find_x)
                node['labels']['lv-antsep-rv-apex']['y'] = str(find_x_y)
            else:
                node['labels']['lv-antsep-rv-apex'] = {}
                node['labels']['lv-antsep-rv-apex']['type'] = node['labels']['curve-lv-antsep-rv']['type']
                node['labels']['lv-antsep-rv-apex']['y'] = ""
                node['labels']['lv-antsep-rv-apex']['x'] = ""

            # LV antsep-rv base
            if node['labels']['curve-lv-antsep-rv']['type'] == 'curve':
                x = node['labels']['curve-lv-antsep-rv']['x']
                y = node['labels']['curve-lv-antsep-rv']['y']

                x = [float(value) for value in x.split()]
                y = [float(value) for value in y.split()]

                find_x = max(x)
                find_x_y = y[x.index(find_x)]

                node['labels']['lv-antsep-rv-base'] = {}
                node['labels']['lv-antsep-rv-base']['type'] = 'point'
                node['labels']['lv-antsep-rv-base']['x'] = str(find_x)
                node['labels']['lv-antsep-rv-base']['y'] = str(find_x_y)
            else:
                node['labels']['lv-antsep-rv-base'] = {}
                node['labels']['lv-antsep-rv-base']['type'] = node['labels']['curve-lv-antsep-rv']['type']
                node['labels']['lv-antsep-rv-base']['y'] = ""
                node['labels']['lv-antsep-rv-base']['x'] = ""

            # LV post-endo apex
            if node['labels']['curve-lv-post-endo']['type'] == 'curve':
                x = node['labels']['curve-lv-post-endo']['x']
                y = node['labels']['curve-lv-post-endo']['y']

                x = [float(value) for value in x.split()]
                y = [float(value) for value in y.split()]

                find_x = min(x)
                find_x_y = y[x.index(find_x)]

                node['labels']['lv-post-endo-apex'] = {}
                node['labels']['lv-post-endo-apex']['type'] = 'point'
                node['labels']['lv-post-endo-apex']['x'] = str(find_x)
                node['labels']['lv-post-endo-apex']['y'] = str(find_x_y)
            else:
                node['labels']['lv-post-endo-apex'] = {}
                node['labels']['lv-post-endo-apex']['type'] = node['labels']['curve-lv-post-endo']['type']
                node['labels']['lv-post-endo-apex']['y'] = ""
                node['labels']['lv-post-endo-apex']['x'] = ""

            # LV post-endo base
            if node['labels']['curve-lv-post-endo']['type'] == 'curve':
                x = node['labels']['curve-lv-post-endo']['x']
                y = node['labels']['curve-lv-post-endo']['y']

                x = [float(value) for value in x.split()]
                y = [float(value) for value in y.split()]

                find_x = max(x)
                find_x_y = y[x.index(find_x)]

                node['labels']['lv-post-endo-base'] = {}
                node['labels']['lv-post-endo-base']['type'] = 'point'
                node['labels']['lv-post-endo-base']['x'] = str(find_x)
                node['labels']['lv-post-endo-base']['y'] = str(find_x_y)
            else:
                node['labels']['lv-post-endo-base'] = {}
                node['labels']['lv-post-endo-base']['type'] = node['labels']['curve-lv-post-endo']['type']
                node['labels']['lv-post-endo-base']['y'] = ""
                node['labels']['lv-post-endo-base']['x'] = ""

            # LV post-epi apex
            if node['labels']['curve-lv-post-epi']['type'] == 'curve':
                x = node['labels']['curve-lv-post-epi']['x']
                y = node['labels']['curve-lv-post-epi']['y']

                x = [float(value) for value in x.split()]
                y = [float(value) for value in y.split()]

                find_x = min(x)
                find_x_y = y[x.index(find_x)]

                node['labels']['lv-post-epi-apex'] = {}
                node['labels']['lv-post-epi-apex']['type'] = 'point'
                node['labels']['lv-post-epi-apex']['x'] = str(find_x)
                node['labels']['lv-post-epi-apex']['y'] = str(find_x_y)
            else:
                node['labels']['lv-post-epi-apex'] = {}
                node['labels']['lv-post-epi-apex']['type'] = node['labels']['curve-lv-post-epi']['type']
                node['labels']['lv-post-epi-apex']['y'] = ""
                node['labels']['lv-post-epi-apex']['x'] = ""

            # LV post-epi base
            if node['labels']['curve-lv-post-epi']['type'] == 'curve':
                x = node['labels']['curve-lv-post-epi']['x']
                y = node['labels']['curve-lv-post-epi']['y']

                x = [float(value) for value in x.split()]
                y = [float(value) for value in y.split()]

                find_x = max(x)
                find_x_y = y[x.index(find_x)]

                node['labels']['lv-post-epi-base'] = {}
                node['labels']['lv-post-epi-base']['type'] = 'point'
                node['labels']['lv-post-epi-base']['x'] = str(find_x)
                node['labels']['lv-post-epi-base']['y'] = str(find_x_y)
            else:
                node['labels']['lv-post-epi-base'] = {}
                node['labels']['lv-post-epi-base']['type'] = node['labels']['curve-lv-post-epi']['type']
                node['labels']['lv-post-epi-base']['y'] = ""
                node['labels']['lv-post-epi-base']['x'] = ""

            top = node['labels']['lv-ivs-top']
            bottom = node['labels']['lv-pw-bottom']

            if top['type'] == bottom['type'] == "point":
                x_top = float(top['x'])
                y_top = float(top['y'])
                x_bot = float(bottom['x'])
                y_bot = float(bottom['y'])

                x_mid = (x_top + x_bot) / 2
                y_mid = (y_top + y_bot) / 2

                x = [x_top, x_mid, x_bot]
                y = [y_top, y_mid, y_bot]

                node['labels']['curve-lv-ed-connect'] = {}
                node['labels']['curve-lv-ed-connect']['type'] = 'curve'
                node['labels']['curve-lv-ed-connect']['x'] = " ".join([str(round(value, 1)) for value in x])
                node['labels']['curve-lv-ed-connect']['y'] = " ".join([str(round(value, 1)) for value in y])

                unity_o = UnityO(unity_code=file, png_cache_dir=Path(PNG_CACHE_DIR))
                img = imageio.imread(unity_o.get_frame_path())

                x_dir = np.sign(x_bot - x_top)
                y_dir = np.sign(y_bot - y_top)

                x_step = (x_bot - x_top) / np.abs(y_bot - y_top)
                y_step = (y_bot - y_top) / np.abs(x_bot - x_top)

                if x_step > y_step:
                    test_y = (np.arange(0, 7) * y_step) + y_bot
                    test_x = (np.arange(0, 7) * x_dir) + x_bot
                else:
                    test_y = (np.arange(0, 7) * y_dir) + y_bot
                    test_x = (np.arange(0, 7) * x_step) + x_bot

                print(test_y, test_x)

                values = img[np.round(test_y).astype(np.int), np.round(test_x).astype(np.int)]

                values = values.astype(np.float)
                if values.ndim == 2:
                    values = np.sum(values, axis=-1)

                if np.argmax(values) == 0:
                    shift = 0
                else:
                    shift = np.argmax(np.diff(values)) + 1

                print(shift)

                new_y_bot = test_y[shift]
                new_x_bot = test_x[shift]

                bottom['y'] = str(round(new_y_bot, 1))
                bottom['x'] = str(round(new_x_bot, 1))

            elif top == bottom == 'off':
                node['labels']['curve-lv-ed-connect'] = {}
                node['labels']['curve-lv-ed-connect']['type'] = 'off'
                node['labels']['curve-lv-ed-connect']['y'] = ""
                node['labels']['curve-lv-ed-connect']['x'] = ""

            else:
                node['labels']['curve-lv-ed-connect'] = {}
                node['labels']['curve-lv-ed-connect']['type'] = 'blurred'
                node['labels']['curve-lv-ed-connect']['y'] = ""
                node['labels']['curve-lv-ed-connect']['x'] = ""

            ant = node['labels']['mv-ant-hinge']
            post = node['labels']['mv-post-hinge']
            apex = node['labels']['lv-apex-endo']

            curve_name = "curve-mv-hinge-connect"
            if ant['type'] == post['type'] == 'point':
                start_x = float(ant['x'])
                start_y = float(ant['y'])
                end_x = float(post['x'])
                end_y = float(post['y'])

                mid_x = (start_x + end_x) / 2
                mid_y = (start_y + end_y) / 2

                x = [start_x, mid_x, end_x]
                y = [start_y, mid_y, end_y]

                node['labels'][curve_name] = {}
                node['labels'][curve_name]['type'] = 'curve'
                node['labels'][curve_name]['x'] = " ".join([str(round(value, 1)) for value in x])
                node['labels'][curve_name]['y'] = " ".join([str(round(value, 1)) for value in y])

            elif ant['type'] == post['type'] == 'off':
                node['labels'][curve_name] = {}
                node['labels'][curve_name]['type'] = 'off'
                node['labels'][curve_name]['y'] = ""
                node['labels'][curve_name]['x'] = ""

            else:
                node['labels'][curve_name] = {}
                node['labels'][curve_name]['type'] = 'blurred'
                node['labels'][curve_name]['y'] = ""
                node['labels'][curve_name]['x'] = ""

            curve_name = "curve-mv-ant-apex-connect"
            if ant['type'] == apex['type'] == 'point':
                start_x = float(ant['x'])
                start_y = float(ant['y'])
                end_x = float(apex['x'])
                end_y = float(apex['y'])

                mid_x = (start_x + end_x) / 2
                mid_y = (start_y + end_y) / 2

                x = [start_x, mid_x, end_x]
                y = [start_y, mid_y, end_y]

                node['labels'][curve_name] = {}
                node['labels'][curve_name]['type'] = 'curve'
                node['labels'][curve_name]['x'] = " ".join([str(round(value, 1)) for value in x])
                node['labels'][curve_name]['y'] = " ".join([str(round(value, 1)) for value in y])

            elif ant['type'] == apex['type'] == 'off':
                node['labels'][curve_name] = {}
                node['labels'][curve_name]['type'] = 'off'
                node['labels'][curve_name]['y'] = ""
                node['labels'][curve_name]['x'] = ""

            else:
                node['labels'][curve_name] = {}
                node['labels'][curve_name]['type'] = 'blurred'
                node['labels'][curve_name]['y'] = ""
                node['labels'][curve_name]['x'] = ""

            curve_name = "curve-mv-post-apex-connect"
            if post['type'] == apex['type'] == 'point':
                start_x = float(post['x'])
                start_y = float(post['y'])
                end_x = float(apex['x'])
                end_y = float(apex['y'])

                mid_x = (start_x + end_x) / 2
                mid_y = (start_y + end_y) / 2

                x = [start_x, mid_x, end_x]
                y = [start_y, mid_y, end_y]

                node['labels'][curve_name] = {}
                node['labels'][curve_name]['type'] = 'curve'
                node['labels'][curve_name]['y'] = " ".join([str(round(value, 1)) for value in y])
                node['labels'][curve_name]['x'] = " ".join([str(round(value, 1)) for value in x])

            elif post['type'] == apex['type'] == 'off':
                node['labels'][curve_name] = {}
                node['labels'][curve_name]['type'] = 'off'
                node['labels'][curve_name]['y'] = ""
                node['labels'][curve_name]['x'] = ""
            else:
                node['labels'][curve_name] = {}
                node['labels'][curve_name]['type'] = 'blurred'
                node['labels'][curve_name]['y'] = ""
                node['labels'][curve_name]['x'] = ""

        except Exception as e:
            print(e)
            pass

    json_train_out = {}
    json_val_out = {}

    if False:

        new_json_out = {}

        for file, node in json_out.items():
            #if node['labels']["mv-ant-hinge"]['type'] == node['labels']["mv-post-hinge"]['type'] == node['labels']["lv-apex-endo"]['type'] == 'point' or node['labels']['curve-lv-endo']['type'] == 'curve':
            if node['labels']["lv-ivs-top"]['type'] == node['labels']["lv-ivs-bottom"]['type'] == node['labels']["lv-pw-top"]['type'] == node['labels']["lv-pw-bottom"]['type'] == 'point':
                new_json_out[file] = node

        json_out = new_json_out

    for image_fn, image_data in json_out.items():
        num_split = int(image_fn.split("-")[1], 16) % 10
        if num_split < 8:
            json_train_out[image_fn] = image_data
        else:
            json_val_out[image_fn] = image_data

    with open(JSON_ALL_LABELS_OUT_PATH, 'w') as outfile:
        json.dump(json_out, outfile, indent=4)

    with open(JSON_TRAIN_LABELS_OUT_PATH, 'w') as outfile:
        json.dump(json_train_out, outfile, indent=4)

    with open(JSON_VAL_LABELS_OUT_PATH, 'w') as outfile:
        json.dump(json_val_out, outfile, indent=4)

    all_keys = json_out.keys()
    all_keys = map(lambda x: x + '\n', all_keys)

    train_keys = json_train_out.keys()
    train_keys = map(lambda x: x + '\n', train_keys)

    val_keys = json_val_out.keys()
    val_keys = map(lambda x: x + '\n', val_keys)

    with open(TXT_ALL_KEYS_OUT_PATH, 'w') as outfile:
        outfile.writelines(all_keys)

    with open(TXT_TRAIN_KEYS_OUT_PATH, 'w') as outfile:
        outfile.writelines(train_keys)

    with open(TXT_VAL_KEYS_OUT_PATH, 'w') as outfile:
        outfile.writelines(val_keys)

    out_csv = []
    points = ["lv-ivs-top", "lv-ivs-bottom", "lv-pw-top", "lv-pw-bottom"]
    for file, node in json_out.items():
        for point in points:
            if node['labels'][point]['type'] == 'point':
                out = {}
                out['file'] = file
                out['user'] = 'expert-last'
                out['time'] = '2100-01-02T01:01:01.001Z'
                out['label'] = point
                out['type'] = node['labels'][point]['type']
                out['y'] = node['labels'][point]['y']
                out['x'] = node['labels'][point]['x']
                out_csv.append(out)

    out_csv = pd.DataFrame(out_csv)
    out_csv.to_csv(CSV_ALL_LABELS_OUT_PATH)


    print("finished")
