import numpy as np
import torch

import imageio


def image_logit_overlay_alpha(logits: torch.Tensor, images=None, cols=None, make_bg_bw=False):

    device = logits.device

    #NCHW

    overlay_intensity = 1.0

    if images is None:
        images = torch.zeros((logits.shape[0], 3, logits.shape[2], logits.shape[3]), device=device)

    cols = torch.tensor(cols, device=device, dtype=torch.float)

    logits = torch.clamp(logits, 0, 1)

    #This keeps the dims
    if make_bg_bw == True:
        out = images[:, [0], :, :]
    else:
        out = images

    #BC(3)HW
    cols = cols.unsqueeze(0).unsqueeze(3).unsqueeze(4)

    #BC(1 for cols)HW
    alpha = logits.unsqueeze(2) * cols

    #And now RGB takes the channels position.
    alpha = torch.sum(alpha, dim=1)
    alpha = torch.clamp(alpha, 0, 1)

    out = overlay_intensity * alpha + out * (1.0 - alpha)

    return out


def read_image_and_crop_into_tensor(image_path, image_crop_size=(608, 608), device="cpu", name=None):

    out_h = image_crop_size[0]
    out_w = image_crop_size[1]

    image = torch.zeros((3, out_h, out_w), dtype=torch.uint8, device=device)

    try:
        image_np = imageio.imread(image_path)
        if image_np.ndim == 2:
            image_np = np.stack((image_np, image_np, image_np), axis=-1)
    except Exception as e:
        print(f"Failed to load image: {image_path}")
        return image, 0, 0

    in_h = image_np.shape[0]
    in_w = image_np.shape[1]

    if in_h <= out_h:
        in_s_h = 0
        in_e_h = in_s_h + in_h
        out_s_h = (out_h - in_h) // 2
        out_e_h = out_s_h + in_h
        label_height_shift = out_s_h
    else:
        in_s_h = (in_h - out_h) // 2
        in_e_h = in_s_h + out_h
        out_s_h = 0
        out_e_h = out_s_h + out_h
        label_height_shift = -in_s_h

    if in_w <= out_w:
        in_s_w = 0
        in_e_w = in_s_w + in_w
        out_s_w = (out_w - in_w) // 2
        out_e_w = out_s_w + in_w
        label_width_shift = out_s_w
    else:
        in_s_w = (in_w - out_w) // 2
        in_e_w = in_s_w + out_w
        out_s_w = 0
        out_e_w = out_s_w + out_w
        label_width_shift = -in_s_w

    image[:, out_s_h:out_e_h, out_s_w:out_e_w] = torch.tensor(image_np[in_s_h:in_e_h, in_s_w:in_e_w, :], device=device).permute(2, 0, 1)

    return image, label_height_shift, label_width_shift
