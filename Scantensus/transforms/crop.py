import numpy as np


def nchw_center_crop_or_pad(image: np.ndarray, output_size=(608,608), cval=0):

    in_n, in_c, in_h, in_w = image.shape
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

    out_image = np.ones((in_n, in_c, out_h, out_w), dtype=image.dtype) * cval
    out_image[:, :, out_s_h:out_e_h, out_s_w:out_e_w,] = image[:, :, in_s_h:in_e_h, in_s_w:in_e_w]

    return out_image


def nchw_patch_centre(image: np.ndarray, patch: np.ndarray):

    in_n, in_c, in_h, in_w = image.shape
    out_n, out_c, out_h, out_w = patch.shape

    if in_n != out_n:
        raise Exception

    if in_c != out_c:
        raise Exception

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

    image[:, :, in_s_h:in_e_h, in_s_w:in_e_w] = patch[:, :, out_s_h:out_e_h, out_s_w:out_e_w,]

    return