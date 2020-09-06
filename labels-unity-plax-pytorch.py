import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import imageio

import torch

from Scantensus.datasets.unity import UnityO
from Scantensus.utils.json import get_keypoint_names_and_colors_from_json

from ScantensusPT.utils import load_and_fix_state_dict

from Scantensus.transforms import center_crop_or_pad

from ScantensusPT.utils.image import image_logit_overlay_alpha
from ScantensusPT.utils.path import get_path_len
from ScantensusPT.utils.trace import get_label_path
from ScantensusPT.utils.keypoint import get_label_keypoint
from Scantensus.utils.heatmaps import make_curve_labels, make_dot_labels
from Scantensus.utils.json import add_snake_to_curve_dict

from Scantensus.nets.HRNet_CFG_I_Sigmoid import get_net_cfg
from ScantensusPT.nets.HRNetV2M7 import get_seg_model

HOST = 'thready3'

PROJECT = "unity"
EXPERIMENT = 'unity-84'
EPOCH = '90'

#IMAGE_LIST = "unity-a5c-1"
#IMAGE_LIST = "unity-plax-2"
IMAGE_LIST = "imp-echo-shunshin-plax-measure-100"
LIST_TYPE = "images"

###############
if HOST == "thready1":
    DATA_DIR = Path("/") / "mnt" / "Storage" / "scantensus-data"
    OUTPUT_DIR = Path("/") / "mnt" / "Storage" / "matt-output"
    DEVICE = torch.device("cuda:0")
elif HOST == "thready3":
    DATA_DIR = Path("/") / "home" / "matthew" / "scantensus-data"
    OUTPUT_DIR = Path("/") / "home" / "matthew" / "matt-output"
    DEVICE = torch.device("cuda:0")
elif HOST == "matt-laptop":
    DATA_DIR = Path("/") / "Users" / "matthew" / "Box" / "scantensus-data"
    OUTPUT_DIR = Path("/") / "Volumes" / "Matt-Data" / "matt-output"
    DEVICE = "cpu"
else:
    raise Exception
################

########

VALIDATION_IMAGE_FILE = DATA_DIR / "validation" / f"{IMAGE_LIST}.txt"
PNG_CACHE_DIR = DATA_DIR / "png-cache"

CSV_LABELS_PATH_HUMAN = DATA_DIR / "labels" / PROJECT / "labels-plax.csv"
CSV_LABELS_PATH_AI = OUTPUT_DIR / "validation" / PROJECT / EXPERIMENT / str(EPOCH) / IMAGE_LIST / "predictions.csv"

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints" / PROJECT / EXPERIMENT
JSON_KEYS_PATH = CHECKPOINT_DIR / "keys.json"

MOVIE_DIR = OUTPUT_DIR / "validation" / PROJECT / EXPERIMENT / str(EPOCH) / IMAGE_LIST / "movies"
IMAGE_DIR = OUTPUT_DIR / "validation" / PROJECT / EXPERIMENT / str(EPOCH) / IMAGE_LIST / "images-human"
os.makedirs(MOVIE_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

#########

valid_users = ["1brNX3XIVYhCq93KeMMY0AEauVo1",
               "2VhZAO48HXc6Hut8lgaOzkfUg6B2",
               "CSZONN4uJ4gMh0RAY2llpjIdZVg2",
               "GblV0kIx4sQF1I94qBnm716Ij3Y2",
               "PU9duvAh4lhxxIoMHarmO6RZIDg1",
               "PosPfN1VDhgHfbRQxEKuXFTvtxj1",
               "PwBpS83NxccFFfHxNLSIkKYPy3o2",
               "onzoCZlCxaTS2HNeP75OSgozs2y2",
               "teAZL0rNDyWevenTcbHQTdzCOGi2",
               "scantensus-echo"]

user_cols = [[0.9882352941176471, 0.3058823529411765, 0.164705882],
             [0.254901960784313, 0.6705882352941176, 0.364705882],
             [0.866666666666666, 0.20392156862745098, 0.592156863],
             [0.9921568627450981, 0.5529411764705883, 0.23921568627450981],
             [0.47058823529411764, 0.7764705882352941, 0.474509804],
             [0.968627450980392, 0.40784313725490196, 0.631372549],
             [0.137254901960784, 0.5176470588235295, 0.262745098],
             [0.6823529411764706, 0.00392156862745098, 0.494117647],
             [0.3, 0.4, 0.5],
             [1.0, 1.0, 0]]
##########

if __name__ == '__main__':

    validation_image_list = []
    with open(VALIDATION_IMAGE_FILE, 'r') as labels_f:
        for line in labels_f:
            validation_image_list.append(line[:-1])

    keypoint_names, keypoint_cols = get_keypoint_names_and_colors_from_json(JSON_KEYS_PATH)

    labels_human_pd = pd.read_csv(CSV_LABELS_PATH_HUMAN)
    labels_ai_pd = pd.read_csv(CSV_LABELS_PATH_AI)

    labels_pd = labels_human_pd.append(labels_ai_pd)

    for row_num, unity_code in enumerate(validation_image_list):

        try:
            unity_o = UnityO(unity_code, png_cache_dir=PNG_CACHE_DIR)
        except Exception as e:
            print(f"{unity_code} is not valid")
            continue

        if getattr(unity_o, 'code_type', None) != 'frame':
            print(f"{unity_code} is not a frame")
            continue

        out_img_fn = f"{row_num:04}-" + unity_o.unity_f_code + ".png"
        out_img_path = IMAGE_DIR / out_img_fn

        image_t = torch.zeros((1, 3, 608, 608), device=DEVICE, dtype=torch.uint8)

        try:
            image_path = unity_o.get_frame_path()
            logging.info(f"{image_path} Loading")
            image = imageio.imread(image_path)

            if image.ndim == 2:
                image = image[..., None]

            source_height = image.shape[0]
            source_width = image.shape[1]

            image = center_crop_or_pad(image)
            image_t[0, :, :, :] = torch.as_tensor(image, dtype=torch.uint8).permute(2, 0, 1)
        except Exception as e:
            print(f"{image_path} Failed to load image")
            continue

        image_t = (image_t.float() / 255.0) - 0.5

        y_pred_out = torch.zeros((1, len(valid_users), 608, 608), device=DEVICE)

        try:

            points_to_process = ["lv-pw-top", "lv-pw-bottom", "lv-ivs-top", "lv-ivs-bottom"]


            shift_height = int((source_height - 608) / 2)
            shift_width = int((source_width - 608) / 2)
            snake_shift = torch.tensor([shift_height, shift_width, 0], device=DEVICE)


            for label_name in points_to_process:
                idx = labels_pd.file.eq(unity_code + ".png") & labels_pd.label.eq(label_name)

                for _, row in labels_pd[idx].iterrows():

                    try:
                        user = row.user
                        x = row.value_x - shift_width
                        y = row.value_y - shift_height

                        if user == "scantensus-echo":
                            kernel_sd = 4
                            kernel_size = 25
                        else:
                            kernel_sd = 2
                            kernel_size = 9

                        img_labels = make_dot_labels(y=y,
                                                     x=x,
                                                     image_size=(608, 608),
                                                     kernel_sd=kernel_sd,
                                                     kernel_size=kernel_size)

                        in_labels = y_pred_out[0, valid_users.index(user), :, :]
                        out_labels = torch.max(torch.stack((in_labels, torch.tensor(img_labels, device=DEVICE))), axis=0)[0]
                        y_pred_out[0, valid_users.index(user), :, :] = out_labels

                    except Exception as e:
                        print(e)
                        pass

            y_pred_mix = image_logit_overlay_alpha(logits=y_pred_out, images=image_t, cols=user_cols)

            y_pred_mix = y_pred_mix.mul_(255).permute(0, 2, 3, 1).type(torch.uint8).cpu().numpy()

            print(f"writing {out_img_path}")
            imageio.imwrite(out_img_path, y_pred_mix[0, ...])

        except Exception as e:
            pass

        print(f"{unity_code} finished processing")
