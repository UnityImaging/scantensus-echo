import math
import random
from collections import namedtuple

import logging
import json

import torch
import numpy as np

from torch.utils.data import Dataset

from Scantensus.datasets.unity import UnityO
from Scantensus.utils.heatmaps import make_curve_labels, interpolate_curve
from ScantensusPT.utils.image import read_image_and_crop_into_tensor
from ScantensusPT.transforms.helpers import get_random_transform_parm, get_affine_matrix
from ScantensusPT.utils.heatmaps import render_gaussian_dot_u, render_gaussian_curve_u

UnityInferSet3dT = namedtuple("UnityInferSet3dT", "image unity_i_code label_height_shift label_width_shift")
UnityTrainSet3dT = namedtuple("UnityTrainSet3dT", "image unity_f_code label_data label_height_shift label_width_shift transform_matrix")


class UnityInferSet3d(Dataset):

    def __init__(self,
                 filehash_list,
                 png_cache_dir,
                 image_crop_size=(640, 640),
                 image_out_size=(320, 320),
                 device="cpu",
                 name=None):

        super().__init__()

        self.filehash_list = filehash_list

        self.png_cache_dir = png_cache_dir
        self.image_crop_size = image_crop_size
        self.image_out_size = image_out_size
        self.name = name
        self.device = device
        self.rgb_convert = torch.tensor([], device=device, dtype=torch.float32)

    def __len__(self):
        return len(self.filehash_list)

    def __getitem__(self, idx):

        filehash = self.filehash_list[idx]

        image_crop_size = self.image_crop_size
        device = self.device

        unity_o = UnityO(unity_code=filehash, png_cache_dir=self.png_cache_dir)

        image_paths = unity_o.get_all_frames_path()
        images = []

        label_height_shift = 0
        label_width_shift = 0
        for image_path in image_paths:
            image, label_height_shift, label_width_shift = read_image_and_crop_into_tensor(image_path=image_path, image_crop_size=image_crop_size, device=device, name="main")
            if self.image_out_size != self.image_crop_size:
                image = image.to(torch.float32).unsqueeze(0)
                image = torch.nn.functional.interpolate(image, size=self.image_out_size, mode="bilinear")
                image = image.squeeze(0).to(torch.uint8)

            images.append(image[[0], ...])  # as we only can cope with b&w

        image = torch.cat(images).unsqueeze(0)

        return UnityInferSet3dT(image=image,
                                unity_i_code=unity_o.unity_i_code,
                                label_height_shift=label_height_shift,
                                label_width_shift=label_width_shift)



class UnityDataset(Dataset):

    def __init__(self,
                 labels_path,
                 png_cache_dir,
                 keypoint_names,
                 transform=False,
                 image_crop_size=(608, 608),
                 image_out_size=(512, 512),
                 pre_post=False,
                 device="cpu",
                 name=None):

        super().__init__()

        self.logger = logging.getLogger("dataset")

        self.png_cache_dir = png_cache_dir
        self.keypoint_names = keypoint_names
        self.transform = transform
        self.image_crop_size = image_crop_size
        self.image_out_size = image_out_size
        self.pre_post = pre_post
        self.device = device
        self.name = name

        with open(labels_path, 'r') as json_f:
            self.db_raw = json.load(json_f)

        self.image_fn_list = list(self.db_raw.keys())

    def __len__(self):
        return len(self.image_fn_list)

    def __getitem__(self, idx):
        device = self.device
        image_crop_size = self.image_crop_size
        image_out_size = self.image_out_size
        transform = self.transform

        unity_code = self.image_fn_list[idx]

        unity_o = UnityO(unity_code=unity_code, png_cache_dir=self.png_cache_dir)

        image_path = unity_o.get_frame_path()
        image = label_height_shift = label_width_shift = None
        image_pre = label_height_shift_pre = label_width_shift_pre = None
        image_post = label_height_shift_post = label_width_shift_post = None

        image, label_height_shift, label_width_shift = read_image_and_crop_into_tensor(image_path=image_path, image_crop_size=image_crop_size, device=device, name="main")

        if self.pre_post:
            image_path_pre = unity_o.get_frame_path(frame_offset=-5)
            image_path_post = unity_o.get_frame_path(frame_offset=+5)
            image_pre, label_height_shift_pre, label_width_shift_pre = read_image_and_crop_into_tensor(image_path=image_path_pre, image_crop_size=image_crop_size, device=device, name="pre")
            image_post, label_height_shift_post, label_width_shift_post = read_image_and_crop_into_tensor(image_path=image_path_post, image_crop_size=image_crop_size, device=device, name="post")

            image = torch.cat((image_pre, image, image_post))

        label_data = self.db_raw[unity_code]['labels']

        if label_height_shift is None:
            label_data = None

        in_out_height_ratio = image_crop_size[0] / image_out_size[0]
        in_out_width_ratio = image_crop_size[1] / image_out_size[1]

        if transform:
            translate_h, translate_w, scale_h, scale_w, rotation_theta, shear_theta = get_random_transform_parm(translate=True,
                                                                                                                scale=True,
                                                                                                                rotate=True,
                                                                                                                shear=True)

            transform_matrix = get_affine_matrix(tx=translate_w,
                                                 ty=translate_h,
                                                 sx=scale_w * in_out_width_ratio,
                                                 sy=scale_h * in_out_height_ratio,
                                                 rotation_theta=rotation_theta,
                                                 shear_theta=shear_theta,
                                                 device=device)

            transform_matrix_inv = transform_matrix.inverse()

        else:
            transform_matrix = get_affine_matrix(tx=0,
                                                 ty=0,
                                                 sx=in_out_width_ratio,
                                                 sy=in_out_height_ratio,
                                                 rotation_theta=0,
                                                 shear_theta=0,
                                                 device=device)

            transform_matrix_inv = transform_matrix.inverse()

        image = image.float().div(255)
        image = transform_image(image=image,
                                transform_matrix=transform_matrix_inv,
                                out_image_size=self.image_out_size)

        if transform:
            random_gamma = math.exp(random.triangular(-0.8, 0.8))
            image = image.pow(random_gamma)

            if self.pre_post:
                if label_height_shift_pre is not None and label_height_shift_post is not None:
                    if random.random() < 0.3:
                        image[3:6, ...] = 0

        image = image.mul(255).to(torch.uint8)

        return UnityTrainSet3dT(image=image,
                                unity_f_code=unity_code,
                                label_data=label_data,
                                label_height_shift=label_height_shift,
                                label_width_shift=label_width_shift,
                                transform_matrix=transform_matrix)


def apply_matrix_to_coords(transform_matrix: torch.Tensor, coord: torch.Tensor):

    if coord.dim() == 2:
        coord = coord.unsqueeze(0)

    batch_size = coord.shape[0]

    if transform_matrix.dim() == 2:
        transform_matrix = transform_matrix.unsqueeze(0)

    if transform_matrix.size()[1:] == (3, 3):
        transform_matrix = transform_matrix[:, :2, :]

    A_batch = transform_matrix[:, :, :2]
    if A_batch.size(0) != batch_size:
        A_batch = A_batch.repeat(batch_size, 1, 1)

    B_batch = transform_matrix[:, :, 2].unsqueeze(1)

    coord = coord.bmm(A_batch.transpose(1, 2)) + B_batch.expand(coord.shape)

    return coord


def transform_image(image: torch.Tensor, transform_matrix: torch.Tensor, out_image_size=(512,512)):

    device = image.device

    if image.dim() == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.dim() == 3:
        image = image.unsqueeze(0)

    batch_size = image.shape[0]

    out_image_h = out_image_size[0]
    out_image_w = out_image_size[1]

    identity_grid = torch.tensor([[[1, 0, 0], [0, 1, 0]]], dtype=torch.float32, device=device)
    intermediate_grid_shape = [batch_size, out_image_h * out_image_w, 2]

    grid = torch.nn.functional.affine_grid(identity_grid, [batch_size, 1, out_image_h, out_image_w], align_corners=False)
    grid = grid.reshape(intermediate_grid_shape)

    # For some reason it gives you w, h at the output of affine_grid. So switch here.
    grid = grid[..., [1, 0]]
    grid = apply_matrix_to_coords(transform_matrix=transform_matrix, coord=grid)
    grid = grid[..., [1, 0]]

    grid = grid.reshape([batch_size, out_image_h, out_image_w, 2])

    # There is no constant selection for padding mode - so border will have to do to weights.
    image = torch.nn.functional.grid_sample(image, grid, mode='bilinear', padding_mode="zeros", align_corners=False).squeeze(0)

    return image


def normalize_coord(coord: torch.Tensor, image_size: torch.Tensor):

    coord = (coord * 2 / image_size) - 1

    return coord


def unnormalize_coord(coord: torch.Tensor, image_size: torch.tensor):

    coord = (coord + 1) * image_size / 2

    return coord

class UnityMakeHeatmaps(torch.nn.Module):

    def __init__(self,
                 keypoint_names,
                 image_crop_size,
                 image_out_size,
                 heatmap_scale_factors=(2, 4),
                 dot_sd=4,
                 curve_sd=2,
                 dot_weight_sd=40,
                 curve_weight_sd=10,
                 dot_weight=40,
                 curve_weight=10,
                 sub_pixel=True,
                 device="cpu"):

        super().__init__()

        self.keypoint_names = keypoint_names
        self.image_crop_size = image_crop_size
        self.image_out_size = image_out_size
        self.heatmap_scale_factors = heatmap_scale_factors
        self.dot_sd = dot_sd
        self.curve_sd = curve_sd
        self.dot_weight_sd = dot_weight_sd
        self.curve_weight_sd = curve_weight_sd
        self.dot_weight = dot_weight
        self.curve_weight = curve_weight
        self.sub_pixel = True
        self.device = device

    def forward(self,
                label_data,
                label_height_shift,
                label_width_shift,
                transform_matrix):

        batch_size = len(transform_matrix)

        out_heatmaps = []
        out_weights = []
        for scale_factor in self.heatmap_scale_factors:
            heatmaps_batch = []
            weights_batch = []
            for i in range(batch_size):
                heatmaps, weights = make_labels_and_masks(image_in_size=self.image_crop_size,
                                                          image_out_size=self.image_out_size,
                                                          label_data=label_data,
                                                          label_data_idx=i,
                                                          keypoint_names=self.keypoint_names,
                                                          label_height_shift=label_height_shift[i],
                                                          label_width_shift=label_width_shift[i],
                                                          heatmap_scale_factor=scale_factor,
                                                          transform_matrix=transform_matrix[i],
                                                          dot_sd=self.dot_sd,
                                                          curve_sd=self.curve_sd,
                                                          dot_weight_sd=self.dot_weight_sd,
                                                          curve_weight_sd=self.curve_weight_sd,
                                                          dot_weight=self.dot_weight,
                                                          curve_weight=self.curve_weight,
                                                          sub_pixel=self.sub_pixel,
                                                          device=self.device)
                heatmaps_batch.append(heatmaps)
                weights_batch.append(weights)
            out_heatmaps.append(torch.stack(heatmaps_batch))
            out_weights.append(torch.stack(weights_batch))

        return out_heatmaps, out_weights


def make_labels_and_masks(image_in_size,
                          image_out_size,
                          keypoint_names,
                          label_data,
                          label_data_idx=None,
                          label_height_shift=0,
                          label_width_shift=0,
                          transform_matrix=None,
                          heatmap_scale_factor=1,
                          dot_sd=4,
                          curve_sd=2,
                          dot_weight_sd=40,
                          curve_weight_sd=10,
                          dot_weight=40,
                          curve_weight=10,
                          sub_pixel=True,
                          device="cpu"):

    # if you are using in a different thread, e.g. a dataloader, device2 and device must be cpu.
    # the reason for the device2 is that the accurate curve heatmap generator is slow unless on cuda
    # there is a fast cpu fall back, but rounds each point to the nearest pixel.

    device = torch.device(device)

    dot_sd_t = torch.tensor([dot_sd, dot_sd], dtype=torch.float, device=device)
    dot_weight_sd_t = torch.tensor([dot_weight_sd, dot_weight_sd], dtype=torch.float, device=device)

    curve_sd_t = torch.tensor([curve_sd, curve_sd], dtype=torch.float, device=device)
    curve_weight_sd_t = torch.tensor([curve_weight_sd, curve_weight_sd], dtype=torch.float, device=device)

    num_keypoints = len(keypoint_names)

    if transform_matrix is not None:
        transform_matrix = transform_matrix.to(device)
        target_out_height = image_out_size[0] // heatmap_scale_factor
        target_out_width = image_out_size[1] // heatmap_scale_factor
        image_out_size_t = torch.tensor(image_out_size, dtype=torch.float, device=device)
    else:
        target_out_height = image_in_size[0] // heatmap_scale_factor
        target_out_width = image_in_size[1] // heatmap_scale_factor
        image_out_size_t = None

    image_in_size_t = torch.tensor(image_in_size, dtype=torch.float, device=device)

    heatmaps = torch.zeros((num_keypoints, target_out_height, target_out_width), device=device, dtype=torch.uint8)
    weights = torch.zeros((num_keypoints, target_out_height, target_out_width), device=device, dtype=torch.uint8)

    if label_data is None:
        return heatmaps, weights

    for keypoint_idx, keypoint in enumerate(keypoint_names):

        # Either hand ove the ['labels'] from the labels file, or from the dataloader
        # which turns the string into a list of strings, hence the idx.

        keypoint_data = label_data.get(keypoint, None)

        if keypoint_data is None:
            heatmaps[keypoint_idx, ...] = 0
            weights[keypoint_idx, ...] = 0
            continue

        if label_data_idx is None:
            label_type = keypoint_data.get('type', None)
        else:
            label_type = label_data[keypoint]['type'][label_data_idx]

        if label_type is None:
            heatmaps[keypoint_idx, ...] = 0
            weights[keypoint_idx, ...] = 0
            continue

        if label_type == "off":
            heatmaps[keypoint_idx, ...] = 0
            weights[keypoint_idx, ...] = 1
            continue

        if label_type == "blurred":
            heatmaps[keypoint_idx, ...] = 0
            weights[keypoint_idx, ...] = 0
            continue

        if label_data_idx is None:
            y_str = str(keypoint_data["y"])
            x_str = str(keypoint_data["x"])
        else:
            y_str = str(keypoint_data["y"][label_data_idx])
            x_str = str(keypoint_data["x"][label_data_idx])

        ys = [(float(y)) for y in y_str.split()]
        xs = [(float(x)) for x in x_str.split()]

        if len(ys) != len(xs) or not all(np.isfinite(ys)) or not all(np.isfinite(xs)) or len(ys) == 2 or len(ys) == 0:
            print(f"problem with data {keypoint}, {ys}, {xs}")
            heatmaps[keypoint_idx, ...] = 0
            weights[keypoint_idx, ...] = 0
            continue

        coord = torch.tensor([ys, xs], device=device).transpose(0, 1)
        label_shift = torch.tensor([label_height_shift, label_width_shift], device=device)
        coord = coord + label_shift

        if transform_matrix is not None:
            coord = normalize_coord(coord=coord, image_size=image_in_size_t)
            coord = apply_matrix_to_coords(transform_matrix=transform_matrix, coord=coord)
            coord = unnormalize_coord(coord=coord, image_size=image_out_size_t)
        else:
            coord = coord.unsqueeze(0)

        coord = coord / heatmap_scale_factor

        if len(ys) == 1:
            out_heatmap = render_gaussian_dot_u(point=coord[0, 0, :],
                                                std=dot_sd_t,
                                                size=(target_out_height, target_out_width),
                                                mul=255)

            out_weight = render_gaussian_dot_u(point=coord[0, 0, :],
                                               std=dot_weight_sd_t,
                                               size=(target_out_height, target_out_width),
                                               mul=(dot_weight-1)).add(1)

        elif len(ys) >= 3:
            points_np = coord[0, :, :].cpu().numpy()

            if sub_pixel:

                curve_points_np = interpolate_curve(points_np, ratio=1)

                curve_points = torch.tensor(curve_points_np,
                                            dtype=torch.float,
                                            device=device)

                out_heatmap = render_gaussian_curve_u(points=curve_points,
                                                      std=curve_sd_t,
                                                      size=(target_out_height, target_out_width),
                                                      mul=255).to(device)

                out_weight = render_gaussian_curve_u(points=curve_points,
                                                     std=curve_weight_sd_t,
                                                     size=(target_out_height, target_out_width),
                                                     mul=(curve_weight-1)).add(1).to(device)

            else:
                curve_points = interpolate_curve(points_np, ratio=2)

                curve_kernel_size = 2 * ((math.ceil(curve_sd) * 5) // 2) + 1
                curve_weight_kernel_size = 2 * ((math.ceil(curve_weight_sd) * 5) // 2) + 1

                out_heatmap = make_curve_labels(points=curve_points,
                                                image_size=(target_out_height, target_out_width),
                                                kernel_sd=curve_sd,
                                                kernel_size=curve_kernel_size)

                out_heatmap = torch.tensor(out_heatmap, device=device)
                out_heatmap = out_heatmap.mul(255).to(torch.uint8)

                out_weight = make_curve_labels(points=curve_points,
                                               image_size=(target_out_height, target_out_width),
                                               kernel_sd=curve_weight_sd,
                                               kernel_size=curve_weight_kernel_size)

                out_weight = torch.tensor(out_weight, device=device)
                out_weight = out_weight.mul(curve_weight-1).add(1).to(torch.uint8)

        else:
            print(f"Error - no idea what problem with data was: {ys}, {xs}")
            continue

        heatmaps[keypoint_idx, ...] = out_heatmap
        weights[keypoint_idx, ...] = out_weight

    return heatmaps, weights
