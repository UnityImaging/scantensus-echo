from typing import List

import math

import numpy as np

import scipy.ndimage
import scipy.interpolate

def get_blur_kernel(kernel_sd, kernel_size):

    if kernel_size % 2 != 1:
        raise Exception

    blur_kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    blur_kernel[kernel_size // 2, kernel_size // 2] = 1
    blur_kernel = scipy.ndimage.filters.gaussian_filter(blur_kernel, (kernel_sd, kernel_sd))
    blur_kernel = blur_kernel / np.max(blur_kernel)

    return blur_kernel

def make_dot_labels(y, x, image_size, kernel_sd, kernel_size):

    image_height, image_width = image_size
    kernel_size_half = kernel_size // 2

    blur_kernel = get_blur_kernel(kernel_sd=kernel_sd, kernel_size=kernel_size)

    y = int(y)
    x = int(x)

    out = np.zeros((image_height + kernel_size - 1, image_width + kernel_size - 1), dtype=np.float32)

    if y + 1 > image_height:
        pass
    elif x + 1 > image_height:
        pass
    elif y < 0:
        pass
    elif x < 0:
        pass
    else:
        out[y:y + kernel_size, x:x + kernel_size] = blur_kernel

    out = out[kernel_size_half:image_height + kernel_size_half, kernel_size_half:image_width + kernel_size_half]

    return out


def interpolate_curve(points: np.array, ratio=1.5):

    distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
    num_curve_points = distance[-1]
    distance = np.insert(distance, 0, 0)
    distance = distance / distance[-1]

    #There can not be any distances that are <=0
    dd = np.diff(distance)
    dd = dd > 0
    dd = np.insert(dd, 0, True)
    distance = distance[dd]
    points = points[dd]

    cs = scipy.interpolate.CubicSpline(distance, points, bc_type='natural')

    distances = np.linspace(0, 1, math.ceil(num_curve_points*ratio))

    curve = cs(distances)

    return curve

def make_curve_labels(points: np.ndarray, image_size, kernel_sd: int, kernel_size: int):

    image_height, image_width = image_size
    kernel_size_half = kernel_size // 2

    blur_kernel = get_blur_kernel(kernel_sd=kernel_sd, kernel_size=kernel_size)

    out = np.zeros((image_height + kernel_size - 1, image_width + kernel_size - 1), dtype=np.float32)

    points = points.astype(np.int)
    curve_ys = points[:, 0]
    curve_xs = points[:, 1]

    for curve_y, curve_x in zip(curve_ys, curve_xs):
        if curve_y + kernel_size_half + 1 > image_height:
            continue
        elif curve_x + kernel_size_half + 1 > image_height:
            continue
        elif curve_y - kernel_size_half < 0:
            continue
        elif curve_x - kernel_size_half < 0:
            continue
        else:
            out[curve_y:curve_y + kernel_size, curve_x:curve_x + kernel_size] = np.maximum(out[curve_y:curve_y + kernel_size, curve_x:curve_x + kernel_size], blur_kernel)

    out = out[kernel_size_half:image_height + kernel_size_half,
          kernel_size_half:image_width + kernel_size_half]

    return out

