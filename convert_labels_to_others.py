import json
import random
import pandas as pd

from pathlib import Path

from Scantensus.utils.json import convert_labels_to_csv, convert_labels_to_firebase

IN_LABELS_PATH = Path("/") / 'Volumes' / 'Matt-Temp' / "labels.json"
OUT_FIREBASE_PATH = Path("/") / 'Volumes' / 'Matt-Temp' / "firebase.json"
OUT_CSV_PATH = Path("/") / 'Volumes' / 'Matt-Temp' / "firebase.csv"


if True:
    USER_NAME = "thready-prediction"
    IMAGE_LIST = "hello"

    labels_to_process = [
        'lv-apex-endo',
        'mv-ant-hinge',
        'mv-post-hinge',
        'curve-lv-endo',
    ]

    firebase_reverse_dict = {
        'ao-valve-top-inner': 'ao-valve-top-inner',
        'lv-apex-endo': 'lv-apex-endo',
        'lv-apex-epi': 'lv-apex-epi',
        'mv-ant-hinge': 'mv-ant-hinge',
        'mv-post-hinge': 'mv-post-hinge',
        'curve-lv-endo': 'curve-lv-endo',
        "lv-pw-top": "lv-pw-top",
        "lv-pw-bottom": "lv-pw-bottom",
        "lv-ivs-top": "lv-ivs-top",
        "lv-ivs-bottom": "lv-ivs-bottom",
        "curve-lv-antsep-endo": "curve-lv-antsep-endo",
        "curve-lv-antsep-rv": "curve-lv-antsep-rv",
        "curve-lv-post-epi": "curve-lv-post-epi",
        "curve-lv-post-endo": "curve-lv-post-endo"
    }

###########

with open(IN_LABELS_PATH, 'r') as json_f:
    out_labels_dict = json.load(json_f)

############
## This randomloy moves the predicions + moves them out.

if True:
    for unity_code, unity_data in out_labels_dict.items():
        unity_labels = unity_data['labels']

        mean_x = (float(unity_labels['mv-post-hinge']['x']) + float(unity_labels['mv-ant-hinge']['x']) + 2 * float(unity_labels['lv-apex-endo']['x'])) / 4
        mean_y = (float(unity_labels['mv-post-hinge']['y']) + float(unity_labels['mv-ant-hinge']['y']) + 2 * float(unity_labels['lv-apex-endo']['y'])) / 4

        in_fix_x = unity_labels['curve-lv-endo']['x']
        in_fix_y = unity_labels['curve-lv-endo']['y']

        in_fix_x = [float(value) for value in in_fix_x.split(" ")]
        in_fix_y = [float(value) for value in in_fix_y.split(" ")]

        num_nodes = len(in_fix_x)

        out_fix_x = []
        out_fix_y = []

        for x, y in zip(in_fix_x, in_fix_y):
            out_x = 0.01 * random.randrange(15,30,1) * (x - mean_x) + x
            out_y = 0.01 * random.randrange(15,30,1) * (y - mean_y) + y

            out_fix_x.append(out_x)
            out_fix_y.append(out_y)

        unity_labels['mv-ant-hinge']['x'] = out_fix_x[0]
        unity_labels['mv-ant-hinge']['y'] = out_fix_y[0]

        unity_labels['lv-apex-endo']['x'] = out_fix_x[num_nodes // 2]
        unity_labels['lv-apex-endo']['y'] = out_fix_y[num_nodes // 2]

        unity_labels['mv-post-hinge']['x'] = out_fix_x[-1]
        unity_labels['mv-post-hinge']['y'] = out_fix_y[-1]

        unity_labels['curve-lv-endo']['x'] = " ".join([str(value) for value in out_fix_x])
        unity_labels['curve-lv-endo']['y'] = " ".join([str(value) for value in out_fix_y])


###########

out_csv_list = convert_labels_to_csv(out_labels_dict=out_labels_dict,
                                     labels_to_process=labels_to_process,
                                     project=IMAGE_LIST)

out_csv_pd = pd.DataFrame(out_csv_list)
out_csv_pd.to_csv(OUT_CSV_PATH)

#########

out_firebase_dict = convert_labels_to_firebase(out_labels_dict=out_labels_dict,
                                               labels_to_process=labels_to_process,
                                               firebase_reverse_dict=firebase_reverse_dict,
                                               user_name=USER_NAME)

with open(OUT_FIREBASE_PATH, "w+") as json_f:
    json.dump(out_firebase_dict, json_f, indent=2)

##########
