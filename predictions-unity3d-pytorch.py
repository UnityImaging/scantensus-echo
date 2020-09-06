import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import os
import math
import json
import yaml
from pathlib import Path

import pandas as pd

import imageio

import torch
from torch.utils.data import DataLoader

from Scantensus.datasets.unity import UnityO
from Scantensus.utils.json import get_keypoint_names_and_colors_from_json
from Scantensus.nets.HRNet_CFG_J_Sigmoid import get_net_cfg

from ScantensusPT.nets.HRNetV2M7 import get_seg_model
from ScantensusPT.nets import unet
from ScantensusPT.datasets.unity3d import UnityInferSet3d, make_labels_and_masks
from ScantensusPT.utils import load_and_fix_state_dict
from ScantensusPT.utils.heatmap_to_label import heatmap_to_label
from ScantensusPT.utils.heatmaps import gaussian_blur2d_norm
from ScantensusPT.utils.image import image_logit_overlay_alpha
from ScantensusPT.utils.path import get_path_len

from Scantensus.utils.json import add_snake_to_curve_dict

#############

HOST = 'thready1'
#HOST = 'matt-laptop'

PROJECT = "unity"
#EXPERIMENT = 'unity-144'
EXPERIMENT = '004'
EPOCH = '300'
JAMES_WEIGHTS_NAME = 'FINAL_160_0.00072.pt.model'

SOURCE = 'c'

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

SINGLE_INFER_BATCH_SIZE = 1
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
    DATA_DIR = Path("/") / "Volumes" / "Matt-Data" / "Projects-Clone" / "scantensus-data"
    OUTPUT_DIR = Path("/") / "Volumes" / "Matt-Temp" / "matt-output"
    DEVICE = "cpu"
else:
    raise Exception
################

PNG_CACHE_DIR = DATA_DIR / "png-cache"
PNG_CACHE_3D_DIR = DATA_DIR / "png-cache-a4c"

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
        'lv-apex-endo',
        'lv-apex-epi',
        'mv-post-hinge',
        'mv-ant-hinge',
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
    #IMAGE_LIST = "imp-echo-shunshin-plax-measure"
    #IMAGE_LIST = "imp-echo-stowell-plax-measure-train-b"
    IMAGE_LIST = "labels-train-plax"
    #IMAGE_LIST = "labels-train-plax"

    labels_to_process = [
        'curve-lv-antsep-endo',
        "curve-lv-antsep-rv",
        "curve-lv-post-epi",
        "curve-lv-post-endo",
        "lv-pw-top",
        "lv-pw-bottom",
        "lv-ivs-top",
        "lv-ivs-bottom"]

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
elif SOURCE == "c":
    IMAGE_LIST = "02-validation100_a4c_filehash"
    #IMAGE_LIST = "broken"

    labels_to_process = [
        "curve-lv-endo",
        "curve-rv-endo",
        "curve-la-endo",
        "curve-ra-endo",
        #'lv-apex-endo',
        #'lv-apex-epi',
        #'mv-post-hinge',
        #'mv-ant-hinge',
    ]


else:
    raise Exception

#######

VALIDATION_IMAGE_FILE = DATA_DIR / "validation" / f"{IMAGE_LIST}.txt"
MOVIE_DIR = OUTPUT_DIR / "validation" / PROJECT / EXPERIMENT / str(EPOCH) / IMAGE_LIST / "movies"
IMAGE_DIR = OUTPUT_DIR / "validation" / PROJECT / EXPERIMENT / str(EPOCH) / IMAGE_LIST / "images"
os.makedirs(MOVIE_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

OUT_FIREBASE_PATH = OUTPUT_DIR / "validation" / PROJECT / EXPERIMENT / str(EPOCH) / IMAGE_LIST / 'firebase.json'
OUT_CSV_PATH = OUTPUT_DIR / "validation" / PROJECT / EXPERIMENT / str(EPOCH) / IMAGE_LIST / 'predictions.csv'

#########

if __name__ == '__main__':
    BLANK_SPACE = " "

    out_csv_list = []
    out_labels_dict = {}
    out_len_dict = {}
    firebase_prediction_dict = {}

    validation_list = []
    with open(VALIDATION_IMAGE_FILE, 'r') as labels_f:
        for line in labels_f:
            validation_list.append(line[:-1])

    keypoint_names, keypoint_cols = get_keypoint_names_and_colors_from_json(JSON_KEYS_PATH)

    keypoint_names_3d = ["curve-lv-endo",
                         "curve-rv-endo",
                         "curve-la-endo",
                         "curve-ra-endo",
                         "curve-lv-endo-jump",
                         "lv-apex-endo",
                         "lv-apex-epi",
                         "mv-post-hinge",
                         "mv-ant-hinge",
                         "rv-apex-endo",
                         "tv-ant-hinge",
                         "tv-sep-hinge"]

    keypoint_cols_3d = []
    for keypoint_name in keypoint_names_3d:
        idx = keypoint_names.index(keypoint_name)
        keypoint_cols_3d.append(keypoint_cols[idx])

    keypoint_sd = [CURVE_SD if 'curve' in keypoint_name else DOT_SD for keypoint_name in keypoint_names]
    keypoint_sd = torch.tensor(keypoint_sd, dtype=torch.float, device=DEVICE)
    keypoint_sd = keypoint_sd.unsqueeze(1).expand(-1, 2)

    keypoint_sd_3d = [CURVE_SD if 'curve' in keypoint_name else DOT_SD for keypoint_name in keypoint_names_3d]
    keypoint_sd_3d = torch.tensor(keypoint_sd_3d, dtype=torch.float, device=DEVICE)
    keypoint_sd_3d = keypoint_sd_3d.unsqueeze(1).expand(-1, 2)

    with open(OUTPUT_DIR / "checkpoints" / "unity" / EXPERIMENT / "cfg.yaml") as f:
        net_cfg_3d = yaml.safe_load(f)

    net_cfg_3d['training']['mixed_precision'] = False  # So we don't have to use pytorch-nightly
    net_cfg_3d['training']['data_parallel'] = False
    net_cfg_3d['resume']['path'] = str(OUTPUT_DIR / "checkpoints" / "unity" / EXPERIMENT / JAMES_WEIGHTS_NAME)

    model_3d = unet.UNet(n_channels=1, n_classes=len(keypoint_names_3d))
    #state_dict_3d = load_and_fix_state_dict(net_cfg_3d['resume']['path'], device=DEVICE)
    state_dict_3d = torch.load(net_cfg_3d['resume']['path'])
    #model_3d.load_state_dict(state_dict_3d['model'])
    model_3d.load_state_dict(state_dict_3d)
    model_3d.to(DEVICE)
    model_3d.eval()

    infer_dataset = UnityInferSet3d(filehash_list=validation_list,
                                    png_cache_dir=PNG_CACHE_3D_DIR,
                                    image_crop_size=IMAGE_CROP_SIZE,
                                    image_out_size=(320, 320),
                                    device="cpu",
                                    name=None)

    infer_dataloader = DataLoader(infer_dataset,
                                  batch_size=SINGLE_INFER_BATCH_SIZE,
                                  shuffle=False,
                                  num_workers=SINGLE_INFER_WORKERS,
                                  pin_memory=False,
                                  sampler=None)

    for batch_num, batch in enumerate(infer_dataloader):

        image_t_3d = batch.image.to(device=DEVICE, dtype=torch.float32, non_blocking=True).div(255.0)

        batch_size, channels, frames, height, width = image_t_3d.shape
        real_frames = frames

        if batch_size != 1:
            print("too many batches")
            continue
        if channels != 1:
            print("Should be b+w")
            continue
        if frames < 32 and frames >=16:
            print("Frames between 16 and 32 - looping")
            blank = torch.zeros((batch_size, channels, 32-frames, height, width), dtype=torch.float32, device=DEVICE)
            loop = image_t_3d[:, :, :32-frames, :, :]
            image_t_3d = torch.cat((image_t_3d, loop), dim=2)
            frames = 32
        if frames < 16:
            print("Not enough frames")
            continue

        y_pred_3d = torch.empty((frames, len(keypoint_names_3d), height, width), dtype=torch.float32, device=DEVICE)

        label_height_shifts = batch.label_height_shift.to(device=DEVICE)
        label_width_shifts = batch.label_width_shift.to(device=DEVICE)

        unity_i_codes = batch.unity_i_code
        unity_i_code = unity_i_codes[0]

        unity_f_codes = []
        for i in range(real_frames):
            unity_f_codes.append(f"{unity_i_code}-{i:04}")

        label_zero_shifts = torch.zeros_like(label_height_shifts)

        path_shift = torch.stack([label_height_shifts, label_width_shifts, label_zero_shifts], dim=-1).unsqueeze(1).unsqueeze(1)

        frame_starts = []
        frame_ends = []
        i = 0
        while True:
            if i >= frames:
                break
            frame_start = i
            frame_end = i + 32
            if frame_end >= frames:
                frame_end = frames
                frame_start = frames - 32
                frame_starts.append(frame_start)
                frame_ends.append(frame_end)
                break
            frame_starts.append(frame_start)
            frame_ends.append(frame_end)
            i = i + 16

        for i, (frame_start, frame_end) in enumerate(zip(frame_starts, frame_ends)):
            with torch.no_grad():
                x = image_t_3d[:, :, frame_start:frame_end, :, :].add(-0.5).div(0.25)

                out = model_3d(x)
                out = out.squeeze(0).permute(1, 0, 2, 3)

                out = torch.clamp(out, 0, 1)
                #out = gaussian_blur2d_norm(y_pred=out, kernel_size=(25, 25), sigma=keypoint_sd_3d)

                if i == 0:
                    y_pred_3d[frame_start:frame_end, ...] = out
                else:
                    y_pred_3d[frame_start+8:frame_end, ...] = out[8:, ...]

        y_pred_3d = y_pred_3d[:real_frames, :, :, :]

        image_t_3d = image_t_3d.squeeze(0).squeeze(0).unsqueeze(1)
        image_t_3d = image_t_3d[:real_frames, :, :, :]

        frames = real_frames

        y_pred_3d = torch.nn.functional.interpolate(y_pred_3d, size=IMAGE_CROP_SIZE, mode='bilinear')
        image_t_3d = torch.nn.functional.interpolate(image_t_3d, size=IMAGE_CROP_SIZE, mode='bilinear')


        ###
        for i, unity_f_code in enumerate(unity_f_codes):
            y_pred_out = image_logit_overlay_alpha(logits=y_pred_3d[[i], ...], images=image_t_3d[[i], ...], cols=keypoint_cols_3d)
            y_pred_out = y_pred_out.mul_(255).permute(0, 2, 3, 1).type(torch.uint8).cpu().numpy()

            out_path = MOVIE_DIR / unity_i_code / "raw" / (unity_f_code + ".png")
            os.makedirs(Path(out_path).parent, exist_ok=True)
            print(out_path)
            imageio.imwrite(out_path, y_pred_out[0, ...])
        cmd = "ffmpeg -y -pattern_type glob -i '" + f"{str(MOVIE_DIR / unity_i_code / 'raw')}" + "//*.png' -c:v libx264 -pix_fmt yuv420p -f mp4 " + f"{str(MOVIE_DIR / unity_i_code)}-raw.mp4"
        os.system(cmd)
        del y_pred_out
        ###

        for unity_f_code in unity_f_codes:
            out_labels_dict[unity_f_code] = {}
            out_labels_dict[unity_f_code]['labels'] = {}

        labels_to_process = ['curve-lv-endo',
                             "lv-apex-endo",
                             "mv-post-hinge",
                             "mv-ant-hinge",
                             ]

        for label in labels_to_process:

            path_crop = heatmap_to_label(y_pred=y_pred_3d, keypoint_names=keypoint_names_3d, label=label, period=50, temporal_smooth=False)

            if label.startswith('curve'):
                path_lens = get_path_len(path_crop)
                path_lens = path_lens.squeeze(1).cpu().detach().numpy()
            else:
                path_lens = torch.zeros(path_crop.shape[0]).cpu().detach().numpy()

            path_csv = pd.DataFrame({"FileHash": unity_i_code,
                                     "unity_code": unity_f_codes,
                                     "len": path_lens})

            gls_out_path = OUTPUT_DIR / "validation" / PROJECT / EXPERIMENT / str(EPOCH) / IMAGE_LIST / "gls" / f"{unity_i_code}.csv"
            os.makedirs(gls_out_path.parent, exist_ok=True)
            path_csv.to_csv(gls_out_path)

            # This is for if downscales
            #path_crop[..., 0:2] = path_crop[..., 0:2]

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

        for i, unity_f_code in enumerate(unity_f_codes):

            label_data = out_labels_dict[unity_f_code]['labels']

            heatmaps, weights = make_labels_and_masks(image_in_size=IMAGE_CROP_SIZE,
                                                      image_out_size=IMAGE_CROP_SIZE,
                                                      keypoint_names=keypoint_names_3d,
                                                      label_data=label_data,
                                                      label_data_idx=None,
                                                      label_height_shift=label_height_shifts[0],
                                                      label_width_shift=label_width_shifts[0],
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
            heatmaps_mix = image_logit_overlay_alpha(logits=heatmaps, images=image_t_3d[[i], ...], cols=keypoint_cols_3d)
            heatmaps_mix = heatmaps_mix.mul(255).permute(0, 2, 3, 1).type(torch.uint8).cpu().numpy()

            out_path = MOVIE_DIR / unity_i_code / "mix" / (unity_f_code + ".png")
            os.makedirs(Path(out_path).parent, exist_ok=True)
            print(out_path)
            imageio.imwrite(out_path, heatmaps_mix[0, ...])

        cmd = "ffmpeg -y -pattern_type glob -i '" + f"{str(MOVIE_DIR / unity_i_code / 'mix')}" + "//*.png' -c:v libx264 -pix_fmt yuv420p -f mp4 " + f"{str(MOVIE_DIR / unity_i_code)}-mix.mp4"
        os.system(cmd)

        continue

    for unity_f_code, data in out_labels_dict.items():
        for label in labels_to_process:
            out_csv = {"file": unity_f_code + ".png",
                       "label": label,
                       "user": "scantensus-echo",
                       "time": "",
                       "project": IMAGE_LIST,
                       "type": data['labels'][label]['type'],
                       "value_y": data['labels'][label]['y'],
                       "value_x": data['labels'][label]['x'],
                       "conf": data['labels'][label]['conf']}

            out_csv_list.append(out_csv)

    out_csv_pd = pd.DataFrame(out_csv_list)
    out_csv_pd.to_csv(OUT_CSV_PATH)
