import random
import math
import numbers

import torch
import torch.nn
import torch.nn.functional

from .helpers import deg2rad, get_affine_matrix


class UintToScaledTensorFloat:

    def __call__(self, sample):

        image, heatmaps, weights = sample

        image = torch.from_numpy(image.transpose((2, 0, 1))).cpu().float().div_(255)
        heatmaps = torch.from_numpy(heatmaps.transpose((2, 0, 1))).cpu().float().div_(255)
        weights = torch.from_numpy(weights.transpose((2, 0, 1))).cpu().float()

        return image, heatmaps, weights

class TensorRandomGamma:
    def __init__(self):
        pass

    def __call__(self, sample):
        image, heatmaps, weights = sample

        random_gamma = math.exp(random.triangular(-0.8, 0.8))

        image = image.pow(random_gamma)

        return image, heatmaps, weights


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

    def __call__(self, sample):

        image, heatmaps, weights = sample

        image = image.unsqueeze(0)
        heatmaps = heatmaps.unsqueeze(0)
        weights = weights.unsqueeze(0)

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

        matrix = get_affine_matrix(tx=translate_w, ty=translate_h, sx=scale_w, sy=scale_h, rotation_deg=rotation_deg, shear_deg=shear_deg)
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

        identity_grid = torch.tensor([[[1, 0, 0], [0, 1, 0]]], dtype=torch.float32)
        intermediate_grid_shape = [batch_size, out_image_h * out_image_w, 2]

        grid = torch.nn.functional.affine_grid(identity_grid, [batch_size, 1, out_image_h, out_image_w])
        grid = grid.reshape(intermediate_grid_shape)

        ##Note addbmm is not usefult here as it also does a reduce
        grid = grid.bmm(A_batch.transpose(1, 2)) + b_batch.expand(intermediate_grid_shape)

        grid = grid.reshape([batch_size, out_image_h, out_image_w, 2])


        #There is no constant selection for padding mode - so border will have to do to weights.
        image = torch.nn.functional.grid_sample(image, grid, mode='bilinear', padding_mode="zeros").squeeze(0)
        heatmaps = torch.nn.functional.grid_sample(heatmaps, grid, mode='bilinear', padding_mode="zeros").squeeze(0)
        weights = torch.nn.functional.grid_sample(weights, grid, mode='bilinear', padding_mode="border").squeeze(0)

        return image, heatmaps, weights


class TensorRandomErase:

    def __init__(self, p=0.5, scale=(0.02, 0.07), ratio=(0.5, 2), value=0, inplace=False):
        #assert isinstance(value, (numbers.Number, str, tuple, list))
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            raise ValueError("range should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("range of scale should be between 0 and 1")
        if p < 0 or p > 1:
            raise ValueError("range of random erasing probability should be between 0 and 1")

        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace

    @staticmethod
    def get_params(img, scale, ratio, value=0):
        """Get parameters for ``erase`` for a random erasing.

        Args:
            img (Tensor): Tensor image of size (C, H, W) to be erased.
            scale: range of proportion of erased area against input image.
            ratio: range of aspect ratio of erased area.

        Returns:
            tuple: params (i, j, h, w, v) to be passed to ``erase`` for random erasing.
        """
        img_c, img_h, img_w = img.shape
        area = img_h * img_w

        for attempt in range(10):
            erase_area = random.uniform(scale[0], scale[1]) * area
            aspect_ratio = random.uniform(ratio[0], ratio[1])

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))

            if h < img_h and w < img_w:
                i = random.randint(0, img_h - h)
                j = random.randint(0, img_w - w)
                if isinstance(value, numbers.Number):
                    v = value
                elif isinstance(value, torch._six.string_classes):
                    v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
                elif isinstance(value, (list, tuple)):
                    v = torch.tensor(value, dtype=torch.float32).view(-1, 1, 1).expand(-1, h, w)
                return i, j, h, w, v

        # Return original image
        return 0, 0, img_h, img_w, img

    def __call__(self, sample):

        image, heatmaps, weights = sample

        value = random.uniform(0,1)

        if random.uniform(0, 1) < self.p:
            x, y, h, w, v = self.get_params(image, scale=self.scale, ratio=self.ratio, value=value)
            return erase(image, x, y, h, w, v, self.inplace), heatmaps, weights

        return image, heatmaps, weights


def erase(img, i, j, h, w, v, inplace=False):
    """ Erase the input Tensor Image with given value.

    Args:
        img (Tensor Image): Tensor image of size (C, H, W) to be erased
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the erased region.
        w (int): Width of the erased region.
        v: Erasing value.
        inplace(bool, optional): For in-place operations. By default is set False.

    Returns:
        Tensor Image: Erased image.
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError('img should be Tensor Image. Got {}'.format(type(img)))

    if not inplace:
        img = img.clone()

    img[:, i:i + h, j:j + w] = v
    return img

class TensorNormalizeImage:

    def __call__(self, sample):
        image, heatmaps, weights = sample

        image -= 0.5

        return image, heatmaps, weights

