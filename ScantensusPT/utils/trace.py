from typing import List

import torch
import torch.nn
import torch.nn.functional

from .point import get_point
from .path import get_path, get_path_via


def get_lv_endo_path(y_pred: torch.Tensor, keypoint_names, period, temporal_smooth):
    snake_n_knots = 31
    snake_n_points = 51

    lv_endo_idx = keypoint_names.index("curve-lv-endo")
    mv_ant_hinge_idx = keypoint_names.index("mv-ant-hinge")
    mv_post_hinge_idx = keypoint_names.index("mv-post-hinge")
    lv_apex_endo_idx = keypoint_names.index("lv-apex-endo")

    lv_endo_jump_idx = keypoint_names.index("curve-lv-endo-jump")

    lv_endo = torch.max(y_pred[:, [lv_endo_idx], :, :], y_pred[:, [lv_endo_jump_idx], :, :] ** 0.25)
    #lv_endo = y_pred[:, [lv_endo_idx], :, :]

    mv_ant_hinge = y_pred[:, [mv_ant_hinge_idx], :, :]
    mv_post_hinge = y_pred[:, [mv_post_hinge_idx], :, :]
    lv_apex_endo = y_pred[:, [lv_apex_endo_idx], :, :]

    mv_ant_hinge_point = get_point(mv_ant_hinge, period=period, temporal_smooth=temporal_smooth)
    mv_post_hinge_point = get_point(mv_post_hinge, period=period, temporal_smooth=temporal_smooth)
    lv_apex_endo_point = get_point(lv_apex_endo, period=period, temporal_smooth=temporal_smooth)

    out = get_path_via(logits=lv_endo,
                       p1s=mv_ant_hinge_point,
                       p2s=lv_apex_endo_point,
                       p3s=mv_post_hinge_point,
                       snake_n_knots=snake_n_knots,
                       snake_n_points=snake_n_points,
                       period=period,
                       temporal_smooth=temporal_smooth)

    return out

def get_lv_endo_path_simple(y_pred: torch.Tensor, keypoint_names, period, temporal_smooth):
    snake_n_knots = 31
    snake_n_points = 51

    lv_endo_idx = keypoint_names.index("curve-lv-endo")
    mv_ant_hinge_idx = keypoint_names.index("mv-ant-hinge")
    mv_post_hinge_idx = keypoint_names.index("mv-post-hinge")

    lv_endo = y_pred[:, [lv_endo_idx], :, :]
    mv_ant_hinge = y_pred[:, [mv_ant_hinge_idx], :, :]
    mv_post_hinge = y_pred[:, [mv_post_hinge_idx], :, :]

    mv_ant_hinge_point = get_point(mv_ant_hinge, period=period)
    mv_post_hinge_point = get_point(mv_post_hinge, period=period)

    out = get_path(logits=lv_endo,
                   p1s=mv_ant_hinge_point,
                   p2s=mv_post_hinge_point,
                   snake_n_knots=snake_n_knots,
                   snake_n_points=snake_n_points,
                   period=period,
                   temporal_smooth=temporal_smooth)

    return out

def get_lv_endo_path_2(y_pred: torch.Tensor, keypoint_names, period, temporal_smooth):
    snake_n_knots = 31
    snake_n_points = 51

    lv_endo_idx = keypoint_names.index("curve-lv-endo")
    mv_ant_hinge_idx = keypoint_names.index("mv-ant-hinge")
    mv_post_hinge_idx = keypoint_names.index("mv-post-hinge")
    lv_apex_endo_idx = keypoint_names.index("lv-apex-endo")

    lv_endo = y_pred[:, [lv_endo_idx], :, :]
    mv_ant_hinge = y_pred[:, [mv_ant_hinge_idx], :, :]
    mv_post_hinge = y_pred[:, [mv_post_hinge_idx], :, :]
    lv_apex_endo = y_pred[:, [lv_apex_endo_idx], :, :]

    mv_ant_hinge_point = get_point(mv_ant_hinge, period=period)
    mv_post_hinge_point = get_point(mv_post_hinge, period=period)
    lv_apex_endo_point = get_point(lv_apex_endo, period=period)

    out_left = get_path(logits=lv_endo,
                        p1s=mv_ant_hinge_point,
                        p2s=lv_apex_endo_point,
                        snake_n_knots=snake_n_knots,
                        snake_n_points=snake_n_points,
                        period=period,
                        temporal_smooth=temporal_smooth)

    out_right = get_path(logits=lv_endo,
                         p1s=lv_apex_endo_point,
                         p2s=mv_post_hinge_point,
                         snake_n_knots=snake_n_knots,
                         snake_n_points=snake_n_points,
                         period=period,
                         temporal_smooth=temporal_smooth)

    out = torch.cat((out_left, out_right[..., 1:, :]), dim=-2)

    return out


def get_rv_endo_path(y_pred: torch.Tensor, keypoint_names, period, temporal_smooth):
    snake_n_knots = 31
    snake_n_points = 31

    rv_endo_idx = keypoint_names.index("curve-rv-endo")
    tv_ant_hinge_idx = keypoint_names.index("tv-ant-hinge")
    tv_sep_hinge_idx = keypoint_names.index("tv-sep-hinge")
    rv_apex_endo_idx = keypoint_names.index("rv-apex-endo")

    rv_endo = y_pred[:, [rv_endo_idx], :, :]
    tv_ant_hinge = y_pred[:, [tv_ant_hinge_idx], :, :]
    tv_sep_hinge = y_pred[:, [tv_sep_hinge_idx], :, :]
    rv_apex_endo = y_pred[:, [rv_apex_endo_idx], :, :]

    tv_ant_hinge_point = get_point(tv_ant_hinge, period)
    rv_apex_endo_point = get_point(rv_apex_endo, period)
    tv_sep_hinge_point = get_point(tv_sep_hinge, period)

    tv_ant_hinge_coord = tv_ant_hinge_point[..., 0:2]
    rv_apex_endo_coord = rv_apex_endo_point[..., 0:2]
    tv_sep_hinge_coord = tv_sep_hinge_point[..., 0:2]

    mid_valve = ((2 * tv_ant_hinge_coord + tv_sep_hinge_coord) / 3)

    batch_size = rv_endo.shape[0]
    channels = rv_endo.shape[1]
    image_height = rv_endo.shape[2]
    image_width = rv_endo.shape[3]
    device = rv_endo.device
    yv, xv = torch.meshgrid([torch.arange(0, image_height, device=device), torch.arange(0, image_width, device=device)])
    yv = yv.float().unsqueeze(0).unsqueeze(0).repeat(batch_size, channels, 1, 1)
    xv = xv.float().unsqueeze(0).unsqueeze(0).repeat(batch_size, channels, 1, 1)

    p1 = mid_valve.unsqueeze(2).unsqueeze(3)
    p2 = rv_apex_endo_coord.unsqueeze(2).unsqueeze(3)

    #https://math.stackexchange.com/questions/274712/calculate-on-which-side-of-a-straight-line-is-a-given-point-located
    d = (xv - p1[..., 1]) * (p2[..., 0] - p1[..., 0]) - (yv - p1[..., 0]) * (p2[..., 1] - p1[..., 1])
    d = torch.sign(d)

    rv_left = rv_endo.clone()
    rv_left[d < 0] = 0

    out_left = get_path(logits=rv_left,
                        p1s=tv_ant_hinge_point,
                        p2s=rv_apex_endo_point,
                        snake_n_knots=snake_n_knots,
                        snake_n_points=snake_n_points + 1,
                        period=period,
                        temporal_smooth=temporal_smooth)

    out_right = get_path(logits=rv_endo,
                         p1s=rv_apex_endo_point,
                         p2s=tv_sep_hinge_point,
                         snake_n_knots=snake_n_knots,
                         snake_n_points=snake_n_points,
                         period=period,
                         temporal_smooth=temporal_smooth)

    out = torch.cat((out_left, out_right[..., 1:, :]), dim=-2)

    return out


def get_la_endo_path(y_pred: torch.Tensor, keypoint_names, period, temporal_smooth):

    snake_n_knots = 21
    snake_n_points = 31

    la_endo_idx = keypoint_names.index("curve-la-endo")
    mv_ant_hinge_idx = keypoint_names.index("mv-ant-hinge")
    mv_post_hinge_idx = keypoint_names.index("mv-post-hinge")

    la_endo = y_pred[:, [la_endo_idx], :, :]
    mv_ant_hinge = y_pred[:, [mv_ant_hinge_idx], :, :]
    mv_post_hinge = y_pred[:, [mv_post_hinge_idx], :, :]

    mv_ant_hinge_point = get_point(mv_ant_hinge, period=period)
    mv_post_hinge_point = get_point(mv_post_hinge, period=period)

    out = get_path(logits=la_endo,
                   p1s=mv_ant_hinge_point,
                   p2s=mv_post_hinge_point,
                   snake_n_knots=snake_n_knots,
                   snake_n_points=snake_n_points,
                   period=period,
                   temporal_smooth=temporal_smooth)

    return out


def get_ra_endo_path(y_pred: torch.Tensor, keypoint_names, period, temporal_smooth):

    snake_n_knots = 21
    snake_n_points = 31

    ra_endo_idx = keypoint_names.index("curve-ra-endo")
    tv_ant_hinge_idx = keypoint_names.index("tv-ant-hinge")
    tv_sep_hinge_idx = keypoint_names.index("tv-sep-hinge")

    ra_endo = y_pred[:, [ra_endo_idx], :, :]
    tv_ant_hinge = y_pred[:, [tv_ant_hinge_idx], :, :]
    tv_sep_hinge = y_pred[:, [tv_sep_hinge_idx], :, :]

    tv_ant_hinge_point = get_point(tv_ant_hinge, period=period)
    tv_sep_hinge_point = get_point(tv_sep_hinge, period=period)

    out = get_path(logits=ra_endo,
                   p1s=tv_ant_hinge_point,
                   p2s=tv_sep_hinge_point,
                   snake_n_knots=snake_n_knots,
                   snake_n_points=snake_n_points,
                   period=period,
                   temporal_smooth=temporal_smooth)

    return out

def get_lv_antsep_endo_path(y_pred: torch.Tensor, keypoint_names, period, temporal_smooth):

    snake_n_knots = 21
    snake_n_points = 11

    curve_idx = keypoint_names.index("curve-lv-antsep-endo")
    start_idx = keypoint_names.index("lv-antsep-endo-apex")
    end_idx = keypoint_names.index("ao-valve-top-inner")

    curve_map = y_pred[:, [curve_idx], :, :]
    start_map = y_pred[:, [start_idx], :, :]
    end_map = y_pred[:, [end_idx], :, :]

    start_point = get_point(start_map, period=period)
    end_point = get_point(end_map, period=period)

    out = get_path(logits=curve_map,
                   p1s=start_point,
                   p2s=end_point,
                   snake_n_knots=snake_n_knots,
                   snake_n_points=snake_n_points,
                   period=period,
                   temporal_smooth=temporal_smooth)

    return out


def get_lv_antsep_rv_path(y_pred: torch.Tensor, keypoint_names, period, temporal_smooth):

    snake_n_knots = 21
    snake_n_points = 11

    curve_idx = keypoint_names.index("curve-lv-antsep-rv")
    start_idx = keypoint_names.index("lv-antsep-rv-apex")
    end_idx = keypoint_names.index("rv-bottom-inner")

    curve_map = y_pred[:, [curve_idx], :, :]
    start_map = y_pred[:, [start_idx], :, :]
    end_map = y_pred[:, [end_idx], :, :]

    start_point = get_point(start_map, period=period)
    end_point = get_point(end_map, period=period)

    out = get_path(logits=curve_map,
                   p1s=start_point,
                   p2s=end_point,
                   snake_n_knots=snake_n_knots,
                   snake_n_points=snake_n_points,
                   period=period,
                   temporal_smooth=temporal_smooth)

    return out


def get_lv_post_endo_path(y_pred: torch.Tensor, keypoint_names, period, temporal_smooth):

    snake_n_knots = 21
    snake_n_points = 11

    curve_idx = keypoint_names.index("curve-lv-post-endo")
    start_idx = keypoint_names.index("lv-post-endo-apex")
    end_idx = keypoint_names.index("mv-post-hinge")

    curve_map = y_pred[:, [curve_idx], :, :]
    start_map = y_pred[:, [start_idx], :, :]
    end_map = y_pred[:, [end_idx], :, :]

    start_point = get_point(start_map, period=period)
    end_point = get_point(end_map, period=period)

    out = get_path(logits=curve_map,
                   p1s=start_point,
                   p2s=end_point,
                   snake_n_knots=snake_n_knots,
                   snake_n_points=snake_n_points,
                   period=period,
                   temporal_smooth=temporal_smooth)

    return out


def get_lv_post_epi_path(y_pred: torch.Tensor, keypoint_names, period, temporal_smooth):

    snake_n_knots = 21
    snake_n_points = 11

    curve_idx = keypoint_names.index("curve-lv-post-epi")
    start_idx = keypoint_names.index("lv-post-epi-apex")
    end_idx = keypoint_names.index("lv-post-epi-base")

    curve_map = y_pred[:, [curve_idx], :, :]
    start_map = y_pred[:, [start_idx], :, :]
    end_map = y_pred[:, [end_idx], :, :]

    start_point = get_point(start_map, period=period)
    end_point = get_point(end_map, period=period)

    out = get_path(logits=curve_map,
                   p1s=start_point,
                   p2s=end_point,
                   snake_n_knots=snake_n_knots,
                   snake_n_points=snake_n_points,
                   period=period,
                   temporal_smooth=temporal_smooth)

    return out


def get_label_path(y_pred: torch.Tensor, keypoint_names: List[str], label: str, period: float = 50, temporal_smooth = False):
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
    else:
        raise Exception
