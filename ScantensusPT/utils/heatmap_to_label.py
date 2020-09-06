from typing import List

import torch

from .trace import *


def heatmap_to_label(y_pred: torch.Tensor,
                     label: str,
                     keypoint_names: List[str],
                     period: float = 50,
                     temporal_smooth=False):

    if label == "curve-lv-endo":
        return get_lv_endo_path(y_pred, keypoint_names, period=period, temporal_smooth=temporal_smooth)
    elif label == "curve-rv-endo":
        return get_rv_endo_path(y_pred, keypoint_names, period=period, temporal_smooth=temporal_smooth)

    elif label == "curve-la-endo":
        return get_la_endo_path(y_pred, keypoint_names, period=period, temporal_smooth=temporal_smooth)
    elif label == "curve-ra-endo":
        return get_ra_endo_path(y_pred, keypoint_names, period=period, temporal_smooth=temporal_smooth)

    elif label == "curve-lv-post-endo":
        return get_lv_post_endo_path(y_pred, keypoint_names, period=period, temporal_smooth=temporal_smooth)
    elif label == "curve-lv-post-epi":
        return get_lv_post_epi_path(y_pred, keypoint_names, period=period, temporal_smooth=temporal_smooth)
    elif label == "curve-lv-antsep-rv":
        return get_lv_antsep_rv_path(y_pred, keypoint_names, period=period, temporal_smooth=temporal_smooth)
    elif label == "curve-lv-antsep-endo":
        return get_lv_antsep_endo_path(y_pred, keypoint_names, period=period, temporal_smooth=temporal_smooth)
    elif not label.startswith("curve"):
        point = get_point(y_pred[:, [keypoint_names.index(label)], :, :], period=period, temporal_smooth=temporal_smooth)
        return point.unsqueeze(2)
    else:
        raise Exception
