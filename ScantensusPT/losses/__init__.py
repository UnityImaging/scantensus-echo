from typing import List

import torch
import torch.nn
from torch.nn import MSELoss
from .losses import MSEClampLoss, MSESumLoss
from ScantensusPT.utils.point import get_point
from ScantensusPT.utils.heatmaps import gaussian_blur2d_norm


class MattMSELoss(torch.nn.Module):

    def __init__(self, keep_layers=None):

        super().__init__()
        self.loss_fn = MSELoss(reduction='mean')

        if keep_layers is not None:
            self.keep_layers = keep_layers[None, :, None, None]
        else:
            self.keep_layers = None

    def forward(self, y_pred: List[torch.Tensor], y_true: List[torch.Tensor], y_weights: List[torch.Tensor]):

        out_a = self.loss_fn(y_pred[0],  y_true[0])
        if self.keep_layers is not None:
            out_a = out_a * self.keep_layers

        out_b = self.loss_fn(y_pred[1],  y_true[1])
        if self.keep_layers is not None:
            out_b = out_b * self.keep_layers

        return out_a + out_b


class MattMSEClampLoss(torch.nn.Module):

    def __init__(self, keep_layers=None):

        super().__init__()
        self.loss_fn = MSEClampLoss()
        if keep_layers is not None:
            self.keep_layers = keep_layers[None, :, None, None]
        else:
            self.keep_layers = None

    def forward(self, y_pred: List[torch.Tensor], y_true: List[torch.Tensor], y_weights: List[torch.Tensor]):

        out_a = self.loss_fn(y_pred[0],  y_true[0], y_weights[0])
        if self.keep_layers is not None:
            out_a = out_a * self.keep_layers
        out_a = torch.mean(out_a)

        out_b = self.loss_fn(y_pred[1],  y_true[1], y_weights[1])
        if self.keep_layers is not None:
            out_b = out_b * self.keep_layers
        out_b = torch.mean(out_b)

        return out_a + out_b


class MattMSESumLoss(torch.nn.Module):

    def __init__(self, keep_layers=None):

        super().__init__()
        self.loss_fn = MSESumLoss()
        if keep_layers is not None:
            self.keep_layers = keep_layers[None, :, None, None]
        else:
            self.keep_layers = None

    def forward(self, y_pred: List[torch.Tensor], y_true: List[torch.Tensor], y_weights: List[torch.Tensor]):

        out_a = self.loss_fn(y_pred[0],  y_true[0], y_weights[0])
        if self.keep_layers is not None:
            out_a = out_a * self.keep_layers
        out_a = torch.mean(out_a)

        out_b = self.loss_fn(y_pred[1],  y_true[1], y_weights[1])
        if self.keep_layers is not None:
            out_b = out_b * self.keep_layers
        out_b = torch.mean(out_b)

        return out_a + out_b

class LVIDMetric(torch.nn.Module):

    def __init__(self, keypoint_names, dot_sd=5):
        super().__init__()
        self.threshold = 0.05
        self.keypoint_names = keypoint_names
        self.dot_sd = dot_sd
        self.chan = [self.keypoint_names.index('lv-ivs-bottom'), self.keypoint_names.index('lv-pw-top')]

    def forward(self, y_pred: List[torch.Tensor], y_true: List[torch.Tensor], y_weights: List[torch.Tensor]):

        y_pred = y_pred[1]
        y_true = y_true[1]

        y_pred = y_pred[:, self.chan, ...]
        y_true = y_true[:, self.chan, ...]

        sigma = torch.tensor([self.dot_sd, self.dot_sd], device=y_pred.device, dtype=y_pred.dtype).unsqueeze(1).expand(-1, 2)
        y_pred = gaussian_blur2d_norm(y_pred=y_pred,
                                      kernel_size=(25, 25),
                                      sigma=sigma)

        p_pred = get_point(y_pred)
        p_true = get_point(y_true)

        dist_pred = torch.sqrt(torch.sum((p_pred[:, 0, 0:2] - p_pred[:, 1, 0:2])**2, dim=-1))
        dist_true = torch.sqrt(torch.sum((p_true[:, 0, 0:2] - p_true[:, 1, 0:2])**2, dim=-1))

        valid_pred = torch.prod(p_pred[..., 2] > self.threshold, dim=1)
        valid_true = torch.prod(p_true[..., 2] > self.threshold, dim=1)

        dist_pred = dist_pred * valid_pred
        dist_true = dist_true * valid_true

        dist_diff = torch.abs(dist_pred - dist_true)

        return torch.sum(dist_diff * valid_true) / torch.sum(valid_true)


