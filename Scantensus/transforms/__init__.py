import numpy as np


def center_crop_or_pad(image: np.ndarray, output_size=(608,608), cval=0):

    in_h, in_w = image.shape[:2]
    out_h, out_w = output_size

    if in_h <= out_h:
        in_s_h = 0
        in_e_h = in_s_h + in_h
        out_s_h = (out_h - in_h) // 2
        out_e_h = out_s_h + in_h
    else:
        in_s_h = (in_h - out_h) // 2
        in_e_h = in_s_h + out_h
        out_s_h = 0
        out_e_h = out_s_h + out_h

    if in_w <= out_w:
        in_s_w = 0
        in_e_w = in_s_w + in_w
        out_s_w = (out_w - in_w) // 2
        out_e_w = out_s_w + in_w
    else:
        in_s_w = (in_w - out_w) // 2
        in_e_w = in_s_w + out_w
        out_s_w = 0
        out_e_w = out_s_w + out_w

    out_image = np.ones((out_h, out_w, image.shape[2]), dtype=image.dtype) * cval
    out_image[out_s_h:out_e_h, out_s_w:out_e_w, :] = image[in_s_h:in_e_h, in_s_w:in_e_w, :]

    return out_image


def center_crop_or_pad_4d(image: np.ndarray, output_size=(608,608), cval=0):

    in_b, in_h, in_w, in_c = image.shape
    out_h, out_w = output_size

    if in_h <= out_h:
        in_s_h = 0
        in_e_h = in_s_h + in_h
        out_s_h = (out_h - in_h) // 2
        out_e_h = out_s_h + in_h
    else:
        in_s_h = (in_h - out_h) // 2
        in_e_h = in_s_h + out_h
        out_s_h = 0
        out_e_h = out_s_h + out_h

    if in_w <= out_w:
        in_s_w = 0
        in_e_w = in_s_w + in_w
        out_s_w = (out_w - in_w) // 2
        out_e_w = out_s_w + in_w
    else:
        in_s_w = (in_w - out_w) // 2
        in_e_w = in_s_w + out_w
        out_s_w = 0
        out_e_w = out_s_w + out_w

    out_image = np.ones((in_b, out_h, out_w, in_c), dtype=image.dtype) * cval
    out_image[:, out_s_h:out_e_h, out_s_w:out_e_w, :] = image[:, in_s_h:in_e_h, in_s_w:in_e_w, :]

    return out_image


def uint_to_scaled_float(image):
    return image / 255.0

def normalize_image(image):
    return image - 0.5

def param2theta(param, w, h):
    # Param will have the translation in pixels. You need to provide the SOURCE image coord.
    # These param need to be the inverse.
    # Effectively this does https://discuss.pytorch.org/t/affine-transformation-matrix-paramters-conversion/19522/13
    # I.e
    # magic = [[2/w, 0, -1],[0, 2/h, -1],[0,0,1]]
    # magic @ param @ magic-1

    theta = np.zeros([2,3])
    theta[0,0] = param[0,0]
    theta[0,1] = param[0,1]*h/w
    theta[0,2] = param[0,2]*2/w + theta[0,0] + theta[0,1] - 1
    theta[1,0] = param[1,0]*w/h
    theta[1,1] = param[1,1]
    theta[1,2] = param[1,2]*2/h + theta[1,0] + theta[1,1] - 1
    return theta

