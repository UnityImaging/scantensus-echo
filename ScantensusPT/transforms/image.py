import random
import math

import torch
import torch.nn
import torch.nn.functional

from .helpers import get_affine_matrix, deg2rad


class UintToScaledTensorFloat:

    def __init__(self, device='cpu'):

        self.device = device


    def __call__(self, image):

        image = torch.as_tensor(image.transpose((2, 0, 1)), device=self.device).float().div_(255)

        return image

class TensorRandomGamma:
    def __init__(self):
        pass

    def __call__(self, image: torch.Tensor):

        random_gamma = math.exp(random.triangular(-0.8, 0.8))

        image = image.pow(random_gamma)

        return image


class TensorRandomWarp:
    def __init__(self, output_size, translate=True, rotate=True, shear=True, scale=True):

        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

        self.translate = translate
        self.rotate = rotate
        self.shear = shear
        self.scale = scale

    def __call__(self, image: torch.Tensor):

        device = image.device

        image = image.unsqueeze(0)

        batch_size = image.shape[0]
        channels = image.shape[1]
        image_h = image.shape[2]
        image_w = image.shape[3]

        out_image_h = self.output_size[0]
        out_image_w = self.output_size[1]

        if self.scale:
            random_scale_h = math.exp(random.triangular(-0.6, 0.6))
            random_scale_w = random_scale_h * math.exp(random.triangular(-0.2, 0.2))
        else:
            random_scale_h = 1
            random_scale_w = 1

        scale_h = (image_h / out_image_h) * random_scale_h
        scale_w = (image_w / out_image_w) * random_scale_w

        if self.rotate:
            rotation_deg = random.triangular(-40, 40)
        else:
            rotation_deg = 0

        if self.shear:
            shear_deg = random.triangular(-20, 20)
        else:
            shear_deg = 0

        if self.translate:
            translate_h = random.uniform(-0.25,0.25)
            translate_w = random.uniform(-0.25,0.25)
        else:
            translate_h = 0
            translate_w = 0

        matrix = get_affine_matrix(tx=translate_w,
                                   ty=translate_h,
                                   sx=scale_w,
                                   sy=scale_h,
                                   rotation_deg=rotation_deg,
                                   shear_deg=shear_deg,
                                   device=device)

        matrix = matrix.inverse()

        if matrix.dim() == 2:
            matrix = matrix[:2, :]
            matrix = matrix.unsqueeze(0)
        elif matrix.dim() == 3:
            if matrix.size()[1:] == (3, 3):
                matrix = matrix[:, :2, :]

        A_batch = matrix[:, :, :2]
        if A_batch.size(0) != batch_size:
            A_batch = A_batch.repeat(batch_size, 1, 1)
        b_batch = matrix[:, :, 2].unsqueeze(1)

        identity_grid = torch.tensor([[[1, 0, 0], [0, 1, 0]]], dtype=torch.float32, device=device)
        intermediate_grid_shape = [batch_size, out_image_h * out_image_w, 2]

        grid = torch.nn.functional.affine_grid(identity_grid, [batch_size, 1, out_image_h, out_image_w])
        grid = grid.reshape(intermediate_grid_shape)

        ##Note addbmm is not usefult here as it also does a reduce
        grid = grid.bmm(A_batch.transpose(1, 2)) + b_batch.expand(intermediate_grid_shape)

        grid = grid.reshape([batch_size, out_image_h, out_image_w, 2])

        #There is no constant selection for padding mode - so border will have to do to weights.
        image = torch.nn.functional.grid_sample(image, grid, mode='bilinear', padding_mode="zeros").squeeze(0)

        return image


class TensorNormalizeImage:

    def __call__(self, image: torch.Tensor):

        image -= 0.5

        return image

