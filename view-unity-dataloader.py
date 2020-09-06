import os
import time

from pathlib import Path

import imageio

import torch
import torch.nn
import torch.nn.functional

from torch.utils.data import DataLoader

from Scantensus.utils.json import get_keypoint_names_and_colors_from_json

from ScantensusPT.datasets.unity import UnityDataset, UnityMakeHeatmaps

from ScantensusPT.utils.image import image_logit_overlay_alpha

HOST = 'thready1'
PROJECT = "unity"

DOT_SD = 4
CURVE_SD = 2

DOT_WEIGHT_SD = DOT_SD * 5
CURVE_WEIGHT_SD = CURVE_SD * 5

DOT_WEIGHT = 80
CURVE_WEIGHT = 20

IMAGE_CROP_SIZE = (640, 640)
IMAGE_OUT_SIZE = (608, 608)

PRE_POST = False

###############
if HOST == "server":
    DATA_DIR = Path("/") / "mnt" / "Storage" / "scantensus-data"
    OUTPUT_DIR = Path("/") / "mnt" / "Storage" / "matt-output"
    DEVICE = "cpu"
elif HOST == "thready1":
    DATA_DIR = Path("/") / "home" / "matthew" / "scantensus-data"
    OUTPUT_DIR = Path("/") / "home" / "matthew" / "matt-output"
    DEVICE = "cuda"
elif HOST == "thready3":
    DATA_DIR = Path("/") / "home" / "matthew" / "scantensus-data"
    OUTPUT_DIR = Path("/") / "home" / "matthew" / "matt-output"
    DEVICE = "cuda"
elif HOST == "matt-laptop":
    DATA_DIR = Path("/") / "Volumes" / "Matt-Data" / "Projects-Clone" / "scantensus-data"
    OUTPUT_DIR = Path("/") / "Volumes" / "Matt-Temp" / "matt-output"
    DEVICE = "cpu"
else:
    raise Exception
################

########
PNG_CACHE_DIR = DATA_DIR / "png-cache"

JSON_KEYS_PATH = DATA_DIR / "labels" / PROJECT / "keys.json"
DB_TRAIN_PATH = DATA_DIR / "labels" / PROJECT / "labels-val.json"
DB_VAL_PATH = DATA_DIR / "labels" / PROJECT / "labels-val.json"

DL_OUTPUT_DIR = OUTPUT_DIR / "labelled_images" / PROJECT

#########

#########
os.makedirs(DL_OUTPUT_DIR, exist_ok=True)
############

#########
DATA_DIR = str(DATA_DIR)
DB_TRAIN_PATH = str(DB_TRAIN_PATH)
DB_VAL_PATH = str(DB_VAL_PATH)
###########

keypoint_names, keypoint_cols = get_keypoint_names_and_colors_from_json(JSON_KEYS_PATH)

train_data = UnityDataset(labels_path=DB_VAL_PATH,
                          png_cache_dir=PNG_CACHE_DIR,
                          keypoint_names=keypoint_names,
                          transform=False,
                          image_crop_size=IMAGE_CROP_SIZE,
                          image_out_size=IMAGE_OUT_SIZE,
                          device=DEVICE,
                          name='train')

train_dataloader = DataLoader(train_data,
                              batch_size=1,
                              shuffle=False,
                              num_workers=0,
                              pin_memory=False)

make_heatmaps = UnityMakeHeatmaps(keypoint_names=keypoint_names,
                                  image_crop_size=IMAGE_CROP_SIZE,
                                  image_out_size=IMAGE_OUT_SIZE,
                                  heatmap_scale_factors=(4, 2),
                                  dot_sd=DOT_SD,
                                  curve_sd=CURVE_SD,
                                  dot_weight_sd=DOT_WEIGHT_SD,
                                  curve_weight_sd=CURVE_WEIGHT_SD,
                                  dot_weight=DOT_WEIGHT,
                                  curve_weight=CURVE_WEIGHT,
                                  sub_pixel=True,
                                  device=DEVICE)


for step, batch in enumerate(train_dataloader):
    print(step)
    if step >= 100:
        break

    image = batch.image.to(device=DEVICE, dtype=torch.float32).div(255.0).add(-0.5)
    if PRE_POST:
        image = image[:, 3:6, :, :]

    unity_f_code = batch.unity_f_code[0]

    label_data = batch.label_data
    label_height_shift = batch.label_height_shift
    label_width_shift = batch.label_width_shift
    transform_matrix = batch.transform_matrix

    heatmaps, weights = make_heatmaps(label_data=label_data,
                                      label_height_shift=label_height_shift,
                                      label_width_shift=label_width_shift,
                                      transform_matrix=transform_matrix, )

    heatmaps = heatmaps[1].to(device=DEVICE, dtype=torch.float32).div(255.0)
    weights = weights[1].to(device=DEVICE, dtype=torch.float32)

    print(f"{unity_f_code}")
    out_img_path = DL_OUTPUT_DIR / f"{step}-{unity_f_code}-image.png"
    out_heatmaps_path = DL_OUTPUT_DIR / f"{step}-{unity_f_code}-heatmaps.png"
    out_weights_path = DL_OUTPUT_DIR / f"{step}-{unity_f_code}-weights.png"
    out_heatmaps_mix_path = DL_OUTPUT_DIR / f"{step}-{unity_f_code}-heatmaps_mix.png"
    out_weights_mix_path = DL_OUTPUT_DIR / f"{step}-{unity_f_code}-weights_mix.png"

    out_img = image.add(0.5).permute(0, 2, 3, 1).squeeze(0).mul(255).type(torch.uint8).cpu().detach().numpy()
    imageio.imwrite(out_img_path, out_img)

    heatmaps = torch.nn.functional.interpolate(heatmaps, scale_factor=2, mode='bilinear', align_corners=True)
    weights = torch.nn.functional.interpolate(weights, scale_factor=2, mode='bilinear', align_corners=True)

    out_heatmaps = image_logit_overlay_alpha(logits=heatmaps, images=None, cols=keypoint_cols)
    out_weights, _ = weights.max(dim=1, keepdim=True)
    out_heatmaps_mix = image_logit_overlay_alpha(logits=heatmaps, images=image.add(0.5), cols=keypoint_cols)
    #out_weights_mix = image_logit_overlay_alpha(logits=weights, images=image, cols=keypoint_cols)

    out_heatmaps = out_heatmaps.permute(0, 2, 3, 1).squeeze(0).mul(255).type(torch.uint8).cpu().detach().numpy()
    out_weights = out_weights.permute(0, 2, 3, 1).squeeze(0).type(torch.uint8).cpu().detach().numpy()
    out_heatmaps_mix = out_heatmaps_mix.permute(0, 2, 3, 1).squeeze(0).mul(255).type(torch.uint8).cpu().detach().numpy()
    #out_weights_mix = out_weights_mix.permute(0, 2, 3, 1).squeeze(0).mul(255).type(torch.uint8).cpu().detach().numpy()

    imageio.imwrite(out_heatmaps_path, out_heatmaps)
    imageio.imwrite(out_weights_path, out_weights)
    imageio.imwrite(out_heatmaps_mix_path, out_heatmaps_mix)
    #imageio.imwrite(out_weights_mix_path, out_weights_mix)




