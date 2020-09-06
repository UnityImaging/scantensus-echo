import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import os
import math
import json
from pathlib import Path

import pandas as pd

import imageio

import torch
from torch.utils.data import DataLoader

from Scantensus.datasets.unity import UnityO
from Scantensus.utils.json import get_keypoint_names_and_colors_from_json, convert_labels_to_csv, convert_labels_to_firebase, curve_list_to_str, curve_str_to_list
from Scantensus.nets.HRNet_CFG_K_Sigmoid import get_net_cfg

from ScantensusPT.nets.HRNetV2M7 import get_seg_model
#from ScantensusPT.nets.HRNetV2M8 import get_seg_model

from ScantensusPT.datasets.unity import UnityInferSet, make_labels_and_masks
from ScantensusPT.utils import load_and_fix_state_dict
from ScantensusPT.utils.heatmap_to_label import heatmap_to_label
from ScantensusPT.utils.heatmaps import gaussian_blur2d_norm
from ScantensusPT.utils.image import image_logit_overlay_alpha



#############

HOST = 'thready3'

PROJECT = "unity"
EXPERIMENT = 'unity-147' #HRNET V8
EPOCH = '300'

SOURCE = 'b'

######

DOT_SD = 4
CURVE_SD = 2

DOT_WEIGHT_SD = DOT_SD * 5
CURVE_WEIGHT_SD = CURVE_SD * 5

DOT_WEIGHT = 80
CURVE_WEIGHT = 20

IMAGE_CROP_SIZE = (640, 640)

PRE_POST = False

########

USE_MULTI_GPU = False

SINGLE_INFER_BATCH_SIZE = 16
SINGLE_INFER_WORKERS = 0

##############

if HOST == "thready1":
    DATA_DIR = Path("/") / "home" / "matthew" / "scantensus-data"
    OUTPUT_DIR = Path("/") / "home" / "matthew" / "matt-output"
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

PNG_CACHE_DIR = DATA_DIR / "png-cache"

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints" / PROJECT / EXPERIMENT
CHECKPOINT_PATH = CHECKPOINT_DIR / f'weights-{EPOCH}.pt'
JSON_KEYS_PATH = CHECKPOINT_DIR / "keys.json"

#######

if SOURCE == 'a':

    IMAGE_LIST = "unity-a4c-1"
    #IMAGE_LIST = "agg2"
    #IMAGE_LIST = "a4c_zoom"
    #IMAGE_LIST = "test_a4c_zoom_all"

    labels_to_process = [
        "curve-lv-endo",
        "curve-rv-endo",
        "curve-la-endo",
        "curve-ra-endo",
        "curve-lv-endo-jump"
        'lv-apex-endo',
        'lv-apex-epi',
        'mv-post-hinge',
        'mv-ant-hinge',
        'rv-apex-endo',
        'tv-ant-hinge',
        'tv-sep-hinge'
    ]

    firebase_reverse_dict = {
        'curve-lv-endo': 'lv-endo',
        'lv-apex-endo': 'apex-endo',
        'lv-apex-epi': 'apex-epi',
        'mv-post-hinge': 'lat-mv',
        'mv-ant-hinge': 'sep-mv-l',
    }

    firebase_reverse_dict = {
        'curve-lv-endo': 'curve-lv-endo',
        'lv-apex-endo': 'lv-apex-endo',
        'lv-apex-epi': 'lv-apex-epi',
        'mv-post-hinge': 'mv-post-hinge',
        'mv-ant-hinge': 'mv-ant-hinge',
    }

elif SOURCE == 'b':

    #IMAGE_LIST = "imp-echo-shunshin-plax-measure-100"
    IMAGE_LIST = "imp-echo-shunshin-validation100-a4c"
    #IMAGE_LIST = "imp-echo-shunshin-plax-measure"
    #IMAGE_LIST = "imp-echo-stowell-plax-measure-train-b"
    #IMAGE_LIST = "labels-train-plax"
    #IMAGE_LIST = "labels-train-plax"

    labels_to_process2 = [
        'curve-lv-antsep-endo',
        "curve-lv-antsep-rv",
        "curve-lv-post-epi",
        "curve-lv-post-endo",
        "lv-pw-top",
        "lv-pw-bottom",
        "lv-ivs-top",
        "lv-ivs-bottom"]

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
else:
    raise Exception

#######

VALIDATION_IMAGE_FILE = DATA_DIR / "validation" / f"{IMAGE_LIST}.txt"

OUTPUT_RUN_DIR = OUTPUT_DIR / "validation" / PROJECT / EXPERIMENT / str(EPOCH) / IMAGE_LIST
os.makedirs(OUTPUT_RUN_DIR, exist_ok=True)

MOVIE_DIR = OUTPUT_RUN_DIR / "movies"
IMAGE_DIR = OUTPUT_RUN_DIR / "images"
os.makedirs(MOVIE_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

OUT_LABELS_PATH = OUTPUT_RUN_DIR / 'labels.json'
OUT_FIREBASE_PATH = OUTPUT_RUN_DIR / 'firebase.json'
OUT_CSV_PATH = OUTPUT_RUN_DIR / 'predictions.csv'

#########

if __name__ == '__main__':
    BLANK_SPACE = " "

    out_csv_list = []
    out_labels_dict = {}
    firebase_prediction_dict = {}

    validation_list = []
    with open(VALIDATION_IMAGE_FILE, 'r') as labels_f:
        for line in labels_f:
            validation_list.append(line[:-1])

    validation_path_list = []
    image_source_list = []

    for row_num, unity_code in enumerate(validation_list):

        try:
            unity_o = UnityO(unity_code, png_cache_dir=PNG_CACHE_DIR)
        except Exception as e:
            print(f"{unity_code} is not valid")
            continue

        if unity_o.code_type == 'frame':
            validation_path_list.append(unity_o.get_frame_path())
            image_source_list.append("frame")

        if unity_o.code_type == 'video':
            paths = unity_o.get_all_frames_path()
            validation_path_list.extend(paths)
            image_source_list.extend(["video"] * len(paths))

    keypoint_names, keypoint_cols = get_keypoint_names_and_colors_from_json(JSON_KEYS_PATH)

    keypoint_sd = [CURVE_SD if 'curve' in keypoint_name else DOT_SD for keypoint_name in keypoint_names]
    keypoint_sd = torch.tensor(keypoint_sd, dtype=torch.float, device=DEVICE)
    keypoint_sd = keypoint_sd.unsqueeze(1).expand(-1, 2)

    net_cfg = get_net_cfg()

    net_cfg['DATASET'] = {}
    net_cfg['MODEL']['PRETRAINED'] = False
    net_cfg['DATASET']['NUM_CLASSES'] = len(keypoint_names)
    if PRE_POST:
        net_cfg['DATASET']['NUM_INPUT_CHANNELS'] = 3 * 3
    else:
        net_cfg['DATASET']['NUM_INPUT_CHANNELS'] = 1 * 3


    single_model = get_seg_model(cfg=net_cfg)

    single_model.init_weights()
    state_dict = load_and_fix_state_dict(CHECKPOINT_PATH, device=DEVICE)
    single_model.load_state_dict(state_dict)

    print(f"Model Loading onto: {DEVICE}")

    model = single_model.to(DEVICE)

    model.eval()

    infer_dataset = UnityInferSet(image_fn_list=validation_path_list,
                                  image_source_list=image_source_list,
                                  png_cache_dir=PNG_CACHE_DIR,
                                  image_crop_size=IMAGE_CROP_SIZE,
                                  pre_post=PRE_POST,
                                  device="cpu",
                                  name=None)

    infer_dataloader = DataLoader(infer_dataset,
                                  batch_size=SINGLE_INFER_BATCH_SIZE,
                                  shuffle=False,
                                  num_workers=SINGLE_INFER_WORKERS,
                                  pin_memory=False,
                                  sampler=None)

    for batch_num, batch in enumerate(infer_dataloader):
        batch_size = batch.image.shape[0]

        image_t = batch.image.to(device=DEVICE, dtype=torch.float32, non_blocking=True).div(255.0).add(-0.5)
        label_height_shifts = batch.label_height_shift.to(device=DEVICE)
        label_width_shifts = batch.label_width_shift.to(device=DEVICE)

        unity_f_codes = batch.unity_f_code
        image_source = batch.image_source

        label_zero_shifts = torch.zeros_like(label_height_shifts)

        path_shift = torch.stack([label_height_shifts, label_width_shifts, label_zero_shifts], dim=-1).unsqueeze(1).unsqueeze(1)

        with torch.no_grad():
            y_pred_25, y_pred_50 = model(image_t)

            y_pred_25 = gaussian_blur2d_norm(y_pred=y_pred_25, kernel_size=(25, 25), sigma=keypoint_sd)
            y_pred_50 = gaussian_blur2d_norm(y_pred=y_pred_50, kernel_size=(25, 25), sigma=keypoint_sd)

            y_pred_25 = torch.nn.functional.interpolate(y_pred_25, scale_factor=4, mode='bilinear', align_corners=True)
            y_pred_50 = torch.nn.functional.interpolate(y_pred_50, scale_factor=2, mode='bilinear', align_corners=True)

            y_pred = (y_pred_25 + y_pred_50) / 2.0
            #y_pred = y_pred_50

            del y_pred_25, y_pred_50

        if PRE_POST:
            image_t = image_t[:, 3:6, :, :]

        y_pred = torch.clamp(y_pred, 0, 1)

        ###

        y_pred_raw = image_logit_overlay_alpha(logits=y_pred, images=None, cols=keypoint_cols)
        y_pred_raw = y_pred_raw.mul_(255).permute(0, 2, 3, 1).type(torch.uint8).cpu().numpy()

        for i, (source, unity_f_code) in enumerate(zip(image_source, unity_f_codes)):
            if source == "frame":
                out_path = IMAGE_DIR / f"{unity_f_code}-raw.png"
            elif source == "video":
                out_path = MOVIE_DIR / unity_f_code[:-5] / "raw" / f"{unity_f_code}.png"
            print(f"Saving raw: {out_path}")
            os.makedirs(Path(out_path).parent, exist_ok=True)
            imageio.imwrite(out_path, y_pred_raw[i, ...])

        del y_pred_raw

        ###

        y_pred_mix = image_logit_overlay_alpha(logits=y_pred, images=image_t.add(0.5), cols=keypoint_cols)
        y_pred_mix = y_pred_mix.mul_(255).permute(0, 2, 3, 1).type(torch.uint8).cpu().numpy()

        for i, (source, unity_f_code) in enumerate(zip(image_source, unity_f_codes)):
            if source == "frame":
                out_path = IMAGE_DIR / f"{unity_f_code}-mix.png"
            elif source == "video":
                out_path = MOVIE_DIR / unity_f_code[:-5] / "mix" / f"{unity_f_code}.png"
            print(f"Saving mix: {out_path}")
            os.makedirs(Path(out_path).parent, exist_ok=True)
            imageio.imwrite(out_path, y_pred_mix[i, ...])

        del y_pred_mix

        ###

        for unity_f_code in unity_f_codes:
            out_labels_dict[unity_f_code] = {}
            out_labels_dict[unity_f_code]['labels'] = {}

        for label in labels_to_process:
            path_crop = heatmap_to_label(y_pred=y_pred, keypoint_names=keypoint_names, label=label, period=50, temporal_smooth=False)
            path_crop[..., 0:2] = path_crop[..., 0:2]

            path = path_crop - path_shift
            for i, unity_f_code in enumerate(unity_f_codes):

                ys = path[i, 0, :, 0].tolist()
                xs = path[i, 0, :, 1].tolist()
                confs = path[i, 0, :, 2].tolist()

                if any([not math.isfinite(y) for y in ys]) or any([not math.isfinite(x) for x in xs]):
                    type = "blurred"
                elif label.startswith('curve'):
                    type = "curve"
                else:
                    type = "point"

                ys = " ".join([str(round(y, 1)) for y in ys])
                xs = " ".join([str(round(x, 1)) for x in xs])
                confs = " ".join([str(round(conf, 3)) for conf in confs])

                out_labels_dict[unity_f_code]['labels'][label] = {}
                out_labels_dict[unity_f_code]['labels'][label]['type'] = type
                if type == "blurred":
                    out_labels_dict[unity_f_code]['labels'][label]['y'] = BLANK_SPACE
                    out_labels_dict[unity_f_code]['labels'][label]['x'] = BLANK_SPACE
                    out_labels_dict[unity_f_code]['labels'][label]['conf'] = BLANK_SPACE
                else:
                    out_labels_dict[unity_f_code]['labels'][label]['y'] = ys
                    out_labels_dict[unity_f_code]['labels'][label]['x'] = xs
                    out_labels_dict[unity_f_code]['labels'][label]['conf'] = confs

        for i, (source, unity_f_code) in enumerate(zip(image_source, unity_f_codes)):

            label_data = out_labels_dict[unity_f_code]['labels']

            heatmaps, weights = make_labels_and_masks(image_in_size=IMAGE_CROP_SIZE,
                                                      image_out_size=IMAGE_CROP_SIZE,
                                                      keypoint_names=keypoint_names,
                                                      label_data=label_data,
                                                      label_data_idx=None,
                                                      label_height_shift=label_height_shifts[i],
                                                      label_width_shift=label_width_shifts[i],
                                                      transform_matrix=None,
                                                      heatmap_scale_factor=1,
                                                      dot_sd=DOT_SD,
                                                      curve_sd=CURVE_SD,
                                                      dot_weight_sd=DOT_WEIGHT_SD,
                                                      curve_weight_sd=CURVE_WEIGHT_SD,
                                                      dot_weight=DOT_WEIGHT,
                                                      curve_weight=CURVE_WEIGHT,
                                                      sub_pixel=True,
                                                      device=DEVICE)

            heatmaps = heatmaps.float().div(255.0).unsqueeze(0)
            heatmaps_mix = image_logit_overlay_alpha(logits=heatmaps, images=image_t[[i], ...].add(0.5), cols=keypoint_cols)
            heatmaps_mix = heatmaps_mix.mul(255).permute(0, 2, 3, 1).type(torch.uint8).cpu().numpy()

            if source == "frame":
                out_path = IMAGE_DIR / f"{unity_f_code}-labels-mix.png"
            elif source == "video":
                out_path = MOVIE_DIR / unity_f_code[:-5] / "labels-mix" / f"{unity_f_code}.png"
            print(f"Saving labels mix: {out_path}")
            os.makedirs(Path(out_path).parent, exist_ok=True)
            imageio.imwrite(out_path, heatmaps_mix[0, ...])

    ########

    with open(OUT_LABELS_PATH, 'w+') as json_f:
        json.dump(out_labels_dict, fp=json_f, indent=2)

    ########

    out_csv_list = convert_labels_to_csv(out_labels_dict=out_labels_dict,
                                         labels_to_process=labels_to_process,
                                         project=IMAGE_LIST)

    out_csv_pd = pd.DataFrame(out_csv_list)
    out_csv_pd.to_csv(OUT_CSV_PATH)

    #########

    out_firebase_dict = convert_labels_to_firebase(out_labels_dict=out_labels_dict,
                                                   labels_to_process=labels_to_process,
                                                   firebase_reverse_dict=firebase_reverse_dict,
                                                   user_name="thready_prediction")

    with open(OUT_FIREBASE_PATH, "w+") as json_f:
        json.dump(out_firebase_dict, json_f, indent=2)
