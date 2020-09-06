import math
import random

import torch


def deg2rad(x):
    return x * math.pi / 180


def get_random_transform_parm(translate=False, scale=False, rotate=False, shear=False):

    if translate:
        translate_h = random.uniform(-0.25, 0.25)
        translate_w = random.uniform(-0.25, 0.25)
    else:
        translate_h = 0
        translate_w = 0

    if scale:
        scale_h = math.exp(random.triangular(-0.6, 0.6))
        scale_w = scale_h * math.exp(random.triangular(-0.2, 0.2))
    else:
        scale_h = 1
        scale_w = 1

    if rotate:
        rotation_deg = random.triangular(-40, 40)
    else:
        rotation_deg = 0

    rotation_theta = math.pi / 180 * rotation_deg

    if shear:
        shear_deg = random.triangular(-20, 20)
    else:
        shear_deg = 0

    shear_theta = math.pi / 180 * shear_deg

    return translate_h, translate_w, scale_h, scale_w, rotation_theta, shear_theta


def get_affine_matrix(tx=0, ty=0, sx=1, sy=1, rotation_theta=0, shear_theta=0, device="cpu"):

    tf_rotate = torch.tensor([[math.cos(rotation_theta), -math.sin(rotation_theta), 0],
                              [math.sin(rotation_theta), math.cos(rotation_theta), 0],
                              [0, 0, 1]],
                             dtype=torch.float,
                             device=device)

    tf_translate = torch.tensor([[1, 0, tx],
                                 [0, 1, ty],
                                 [0, 0, 1]],
                                dtype=torch.float,
                                device=device)

    tf_scale = torch.tensor([[sx, 0, 0],
                             [0, sy, 0],
                             [0, 0, 1]],
                            dtype=torch.float,
                            device=device)

    tf_shear = torch.tensor([[1, -math.sin(shear_theta), 0],
                             [0, math.cos(shear_theta), 0],
                             [0, 0, 1]],
                            dtype=torch.float,
                            device=device)

    matrix = tf_shear @ tf_scale @ tf_rotate @ tf_translate

    return matrix
