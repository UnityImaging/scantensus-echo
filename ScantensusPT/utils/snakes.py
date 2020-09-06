import numpy as np

import scipy.interpolate
import skimage.graph

import torch
import torch.nn
import torch.nn.functional

from ScantensusPT.utils.segmentation import active_contour
from .path import get_path_len





def tensor_points_to_init_snake(points: torch.Tensor, snake_n: int):
    device = points.device

    points_shape = points.shape
    batch_size = points_shape[0]
    channels = points_shape[1]
    num_points = points_shape[2]

    distance = torch.empty((batch_size, channels, num_points), dtype=torch.float, device=device)
    out = torch.empty((batch_size, channels, snake_n, 2), dtype=torch.float, device=device)

    distance[:, :, 0] = 0
    distance[:, :, 1:] = torch.cumsum(torch.sqrt(torch.sum((points[..., 1:,:] - points[..., :-1,:]) ** 2, dim=-1)), dim=-1)

    distance = distance / distance[..., -1:]

    distance_np = distance.cpu().numpy()
    points_np = points.cpu().numpy()

    new_distances = np.linspace(0, 1, snake_n)

    for batch_num in range(batch_size):
        for channel_num in range(channels):
            if np.any(np.isnan(distance_np[batch_num, channel_num, :])):
                init_np = np.zeros((snake_n, 2), dtype=np.float)
            else:
                cs = scipy.interpolate.CubicSpline(distance_np[batch_num, channel_num, :], points_np[batch_num, channel_num, :, :], bc_type='natural')
                init_np = cs(new_distances)
            out[batch_num, channel_num, :, :] = torch.as_tensor(init_np)

    return out

def tensor_snake_to_snake_and_conf(snake: torch.Tensor, img: torch.Tensor):

    img_shape = img.shape
    img_height = img_shape[2]
    img_width = img_shape[3]
    img_size = torch.tensor([img_height, img_width], dtype=torch.float, device=img.device)

    snake_shrink = (2 * snake / img_size) - 1

    snake_conf = torch.nn.functional.grid_sample(img.transpose(2,3),
                                                 snake_shrink,
                                                 'bilinear',
                                                 'zeros')

    snake_conf = snake_conf.transpose(2,3)

    out = torch.cat((snake, snake_conf), dim=-1)

    return out

def get_lv_endo_snake(y_pred: torch.Tensor, keypoint_names):
    snake_n = 100

    lv_endo_idx = keypoint_names.index("curve-lv-endo")
    mv_ant_hinge_idx = keypoint_names.index("mv-ant-hinge")
    mv_post_hinge_idx = keypoint_names.index("mv-post-hinge")
    lv_apex_endo_idx = keypoint_names.index("lv-apex-endo")

    lv_endo = y_pred[:, lv_endo_idx, :, :].unsqueeze(1)
    mv_ant_hinge = y_pred[:, mv_ant_hinge_idx, :, :].unsqueeze(1)
    mv_post_hinge = y_pred[:, mv_post_hinge_idx, :, :].unsqueeze(1)
    lv_apex_endo = y_pred[:, lv_apex_endo_idx, :, :].unsqueeze(1)

    mv_ant_hinge_coord, mv_ant_hinge_val = get_point(mv_ant_hinge)
    lv_apex_endo_coord, lv_apex_endo_val = get_point(lv_apex_endo)
    mv_post_hinge_coord, mv_post_hinge_val = get_point(mv_post_hinge)

    seg_a_normal = torch.stack([(mv_ant_hinge_coord[..., 1] - lv_apex_endo_coord[..., 1]), -(mv_ant_hinge_coord[..., 0] - lv_apex_endo_coord[..., 0])], dim=-1)
    seg_a_mid = ((mv_ant_hinge_coord + lv_apex_endo_coord) / 2) + (seg_a_normal / 3)

    seg_b_normal = torch.stack([(lv_apex_endo_coord[..., 1] - mv_post_hinge_coord[..., 1]), -(lv_apex_endo_coord[..., 0] - mv_post_hinge_coord[..., 0])], dim=-1)
    seg_b_mid = ((lv_apex_endo_coord + mv_post_hinge_coord) / 2) + (seg_b_normal / 3)

    # BC(yx) -> BC(point_num)(yx)
    points = torch.stack((mv_ant_hinge_coord,
                          seg_a_mid,
                          lv_apex_endo_coord,
                          seg_b_mid,
                          mv_post_hinge_coord), dim=2)

    init = tensor_points_to_init_snake(points, snake_n)

    snake = active_contour(lv_endo,
                           init,
                           alpha=0.01,
                           beta=0.5,
                           w_edge=0,
                           w_line=5,
                           boundary_condition="fixed",
                           max_px_move=1,
                           coordinates="rc")

    out = tensor_snake_to_snake_and_conf(snake, lv_endo)

    return out

def get_lv_endo_path(y_pred: torch.Tensor, keypoint_names):
    snake_n_knots = 20
    snake_n_points = 100

    lv_endo_idx = keypoint_names.index("curve-lv-endo")
    mv_ant_hinge_idx = keypoint_names.index("mv-ant-hinge")
    mv_post_hinge_idx = keypoint_names.index("mv-post-hinge")
    lv_apex_endo_idx = keypoint_names.index("lv-apex-endo")

    lv_endo = y_pred[:, lv_endo_idx, :, :].unsqueeze(1)
    mv_ant_hinge = y_pred[:, mv_ant_hinge_idx, :, :].unsqueeze(1)
    mv_post_hinge = y_pred[:, mv_post_hinge_idx, :, :].unsqueeze(1)
    lv_apex_endo = y_pred[:, lv_apex_endo_idx, :, :].unsqueeze(1)

    mv_ant_hinge_coord, mv_ant_hinge_val = get_point(mv_ant_hinge)
    lv_apex_endo_coord, lv_apex_endo_val = get_point(lv_apex_endo)
    mv_post_hinge_coord, mv_post_hinge_val = get_point(mv_post_hinge)

    out = get_path(logits=lv_endo, p1s=mv_ant_hinge_coord, p2s=mv_post_hinge_coord, snake_n_knots=snake_n_knots, snake_n_points=snake_n_points)

    return out



def get_rv_endo_snake(y_pred: torch.Tensor, keypoint_names):
    snake_n = 100

    rv_endo_idx = keypoint_names.index("curve-rv-endo")
    tv_ant_hinge_idx = keypoint_names.index("tv-ant-hinge")
    tv_sep_hinge_idx = keypoint_names.index("tv-sep-hinge")
    rv_apex_endo_idx = keypoint_names.index("rv-apex-endo")

    rv_endo = y_pred[:, rv_endo_idx, :, :].unsqueeze(1)
    tv_ant_hinge = y_pred[:, tv_ant_hinge_idx, :, :].unsqueeze(1)
    tv_sep_hinge = y_pred[:, tv_sep_hinge_idx, :, :].unsqueeze(1)
    rv_apex_endo = y_pred[:, rv_apex_endo_idx, :, :].unsqueeze(1)

    tv_ant_hinge_coord, tv_ant_hinge_val = get_point(tv_ant_hinge)
    rv_apex_endo_coord, rv_apex_endo_val = get_point(rv_apex_endo)
    tv_sep_hinge_coord, tv_sep_hinge_val = get_point(tv_sep_hinge)

    seg_a_normal = torch.stack([(tv_ant_hinge_coord[..., 1] - rv_apex_endo_coord[..., 1]), -(tv_ant_hinge_coord[..., 0] - rv_apex_endo_coord[..., 0])], dim=-1)
    seg_a_mid = ((tv_ant_hinge_coord + rv_apex_endo_coord) / 2) + (seg_a_normal / 3)

    seg_b_normal = torch.stack([(rv_apex_endo_coord[..., 1] - tv_sep_hinge_coord[..., 1]), -(rv_apex_endo_coord[..., 0] - tv_sep_hinge_coord[..., 0])], dim=-1)
    seg_b_mid = ((rv_apex_endo_coord + tv_sep_hinge_coord) / 2) + (seg_b_normal / 3)

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
    d_left = (d+1)/2
    d_right = (1-d)/2 #not actually used as the straight line ness of hte right line is enough

    points_a = torch.stack((tv_ant_hinge_coord,
                       seg_a_mid,
                       rv_apex_endo_coord), dim=2)

    init_a = tensor_points_to_init_snake(points_a, snake_n//2)

    snake_a = active_contour(rv_endo * d_left,
                             init_a,
                             alpha=0.01,
                             beta=0.5,
                             w_edge=0,
                             w_line=5,
                             boundary_condition="fixed",
                             max_px_move=1,
                             coordinates="rc")

    points_b = torch.stack((rv_apex_endo_coord,
                            seg_b_mid,
                            tv_sep_hinge_coord), dim=2)

    init_b = tensor_points_to_init_snake(points_b, snake_n//4)

    snake_b = active_contour(rv_endo,
                             init_b,
                             alpha=0.01,
                             beta=0.5,
                             w_edge=0,
                             w_line=5,
                             boundary_condition="fixed",
                             max_px_move=1,
                             coordinates="rc")

    #Also - the next spline function failes if duplicated points.
    snake = torch.cat((snake_a, snake_b[..., 1:, :]), dim=-2)

    out = tensor_snake_to_snake_and_conf(snake, rv_endo)

    return out

def get_rv_endo_path(y_pred: torch.Tensor, keypoint_names):
    snake_n_knots = 10
    snake_n_points = 50

    rv_endo_idx = keypoint_names.index("curve-rv-endo")
    tv_ant_hinge_idx = keypoint_names.index("tv-ant-hinge")
    tv_sep_hinge_idx = keypoint_names.index("tv-sep-hinge")
    rv_apex_endo_idx = keypoint_names.index("rv-apex-endo")

    rv_endo = y_pred[:, rv_endo_idx, :, :].unsqueeze(1)
    tv_ant_hinge = y_pred[:, tv_ant_hinge_idx, :, :].unsqueeze(1)
    tv_sep_hinge = y_pred[:, tv_sep_hinge_idx, :, :].unsqueeze(1)
    rv_apex_endo = y_pred[:, rv_apex_endo_idx, :, :].unsqueeze(1)

    tv_ant_hinge_coord, tv_ant_hinge_val = get_point(tv_ant_hinge)
    rv_apex_endo_coord, rv_apex_endo_val = get_point(rv_apex_endo)
    tv_sep_hinge_coord, tv_sep_hinge_val = get_point(tv_sep_hinge)

    seg_a_normal = torch.stack([(tv_ant_hinge_coord[..., 1] - rv_apex_endo_coord[..., 1]), -(tv_ant_hinge_coord[..., 0] - rv_apex_endo_coord[..., 0])], dim=-1)
    seg_a_mid = ((tv_ant_hinge_coord + rv_apex_endo_coord) / 2) + (seg_a_normal / 3)

    seg_b_normal = torch.stack([(rv_apex_endo_coord[..., 1] - tv_sep_hinge_coord[..., 1]), -(rv_apex_endo_coord[..., 0] - tv_sep_hinge_coord[..., 0])], dim=-1)
    seg_b_mid = ((rv_apex_endo_coord + tv_sep_hinge_coord) / 2) + (seg_b_normal / 3)

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
    d_left = (d+1)/2
    d_right = (1-d)/2 #not actually used as the straight line ness of hte right line is enough

    rv_left = rv_endo.clone()
    rv_left[d<0] = 0

    out_left = get_path(logits=rv_left,
                   p1s=tv_ant_hinge_coord,
                   p2s=rv_apex_endo_coord,
                   snake_n_knots=snake_n_knots,
                   snake_n_points=snake_n_points+1)

    out_right = get_path(logits=rv_endo,
                   p1s=rv_apex_endo_coord,
                   p2s=tv_sep_hinge_coord,
                   snake_n_knots=snake_n_knots,
                   snake_n_points=snake_n_points)

    out = torch.cat((out_left, out_right[..., 1:, :]), dim=-2)

    return out


def get_la_endo_snake(y_pred: torch.Tensor, keypoint_names):
    snake_n = 50
    normal_ratio = 3

    la_endo_idx = keypoint_names.index("curve-la-endo")
    mv_ant_hinge_idx = keypoint_names.index("mv-ant-hinge")
    mv_post_hinge_idx = keypoint_names.index("mv-post-hinge")

    la_endo = y_pred[:, la_endo_idx, :, :].unsqueeze(1)
    mv_ant_hinge = y_pred[:, mv_ant_hinge_idx, :, :].unsqueeze(1)
    mv_post_hinge = y_pred[:, mv_post_hinge_idx, :, :].unsqueeze(1)

    mv_ant_hinge_coord, mv_ant_hinge_val = get_point(mv_ant_hinge)
    mv_post_hinge_coord, mv_post_hinge_val = get_point(mv_post_hinge)

    seg_a_normal = torch.stack([-(mv_ant_hinge_coord[..., 1] - mv_post_hinge_coord[..., 1]), (mv_ant_hinge_coord[..., 0] - mv_post_hinge_coord[..., 0])], dim=-1)

    seg_a_a = mv_ant_hinge_coord + (seg_a_normal * normal_ratio)
    seg_a_b = mv_post_hinge_coord + (seg_a_normal * normal_ratio)

    points = torch.stack((mv_ant_hinge_coord,
                       seg_a_a,
                       seg_a_b,
                       mv_post_hinge_coord), dim=2)

    init = tensor_points_to_init_snake(points, snake_n)

    snake = active_contour(la_endo,
                           init,
                           alpha=0.01,
                           beta=0.5,
                           w_edge=0,
                           w_line=5,
                           boundary_condition="fixed",
                           max_px_move=1,
                           coordinates="rc")

    out = tensor_snake_to_snake_and_conf(snake, la_endo)

    return out

def get_la_endo_path(y_pred: torch.Tensor, keypoint_names):

    snake_n_knots = 20
    snake_n_points = 100

    la_endo_idx = keypoint_names.index("curve-la-endo")
    mv_ant_hinge_idx = keypoint_names.index("mv-ant-hinge")
    mv_post_hinge_idx = keypoint_names.index("mv-post-hinge")

    la_endo = y_pred[:, la_endo_idx, :, :].unsqueeze(1)
    mv_ant_hinge = y_pred[:, mv_ant_hinge_idx, :, :].unsqueeze(1)
    mv_post_hinge = y_pred[:, mv_post_hinge_idx, :, :].unsqueeze(1)

    mv_ant_hinge_coord, mv_ant_hinge_val = get_point(mv_ant_hinge)
    mv_post_hinge_coord, mv_post_hinge_val = get_point(mv_post_hinge)

    out = get_path(logits=la_endo,
                   p1s=mv_ant_hinge_coord,
                   p2s=mv_post_hinge_coord,
                   snake_n_knots=snake_n_knots,
                   snake_n_points=snake_n_points)

    return out


def get_ra_endo_snake(y_pred: torch.Tensor, keypoint_names):

    snake_n = 50
    normal_ratio = 3

    ra_endo_idx = keypoint_names.index("curve-ra-endo")
    tv_ant_hinge_idx = keypoint_names.index("tv-ant-hinge")
    tv_sep_hinge_idx = keypoint_names.index("tv-sep-hinge")

    ra_endo = y_pred[:, ra_endo_idx, :, :].unsqueeze(1)
    tv_ant_hinge = y_pred[:, tv_ant_hinge_idx, :, :].unsqueeze(1)
    tv_sep_hinge = y_pred[:, tv_sep_hinge_idx, :, :].unsqueeze(1)

    tv_ant_hinge_coord, tv_ant_hinge_val = get_point(tv_ant_hinge)
    tv_sep_hinge_coord, tv_sep_hinge_val = get_point(tv_sep_hinge)

    seg_a_normal = torch.stack([-(tv_ant_hinge_coord[..., 1] - tv_sep_hinge_coord[..., 1]), (tv_ant_hinge_coord[..., 0] - tv_sep_hinge_coord[..., 0])], dim=-1)

    seg_a_a = tv_ant_hinge_coord + (seg_a_normal * normal_ratio)
    seg_a_b = tv_sep_hinge_coord + (seg_a_normal * normal_ratio)

    points = torch.stack((tv_ant_hinge_coord,
                       seg_a_a,
                       seg_a_b,
                       tv_sep_hinge_coord), dim=2)

    init = tensor_points_to_init_snake(points, snake_n)

    snake = active_contour(ra_endo,
                           init,
                           alpha=0.01,
                           beta=0.5,
                           w_edge=0,
                           w_line=5,
                           boundary_condition="fixed",
                           max_px_move=1,
                           coordinates="rc")

    out = tensor_snake_to_snake_and_conf(snake, ra_endo)

    return out

def get_ra_endo_path(y_pred: torch.Tensor, keypoint_names):

    snake_n_knots = 20
    snake_n_points = 100
    normal_ratio = 3

    ra_endo_idx = keypoint_names.index("curve-ra-endo")
    tv_ant_hinge_idx = keypoint_names.index("tv-ant-hinge")
    tv_sep_hinge_idx = keypoint_names.index("tv-sep-hinge")

    ra_endo = y_pred[:, ra_endo_idx, :, :].unsqueeze(1)
    tv_ant_hinge = y_pred[:, tv_ant_hinge_idx, :, :].unsqueeze(1)
    tv_sep_hinge = y_pred[:, tv_sep_hinge_idx, :, :].unsqueeze(1)

    tv_ant_hinge_coord, tv_ant_hinge_val = get_point(tv_ant_hinge)
    tv_sep_hinge_coord, tv_sep_hinge_val = get_point(tv_sep_hinge)

    seg_a_normal = torch.stack([-(tv_ant_hinge_coord[..., 1] - tv_sep_hinge_coord[..., 1]), (tv_ant_hinge_coord[..., 0] - tv_sep_hinge_coord[..., 0])], dim=-1)

    seg_a_a = tv_ant_hinge_coord + (seg_a_normal * normal_ratio)
    seg_a_b = tv_sep_hinge_coord + (seg_a_normal * normal_ratio)

    out = get_path(logits=ra_endo,
                   p1s=tv_ant_hinge_coord,
                   p2s=tv_sep_hinge_coord,
                   snake_n_knots=snake_n_knots,
                   snake_n_points=snake_n_points)

    return out

def get_lv_antsep_endo_path(y_pred: torch.Tensor, keypoint_names):

    snake_n = 20

    curve_idx = keypoint_names.index("curve-lv-antsep-endo")
    start_idx = keypoint_names.index("lv-antsep-endo-apex")
    end_idx = keypoint_names.index("lv-antsep-endo-base")

    curve_map = y_pred[:, curve_idx, :, :].unsqueeze(1)
    start_map = y_pred[:, start_idx, :, :].unsqueeze(1)
    end_map = y_pred[:, end_idx, :, :].unsqueeze(1)

    start_coord, start_val = get_point(start_map)
    end_coord, end_val = get_point(end_map)


    costs = torch.clamp(curve_map, 0, 0.8)
    costs = (1 - (costs / 0.8)) * 100

    costs = costs.cpu().detach().numpy()
    p1s = start_coord.int().cpu().detach().numpy()
    p2s = end_coord.int().cpu().detach().numpy()

    out_points = torch.zeros((y_pred.shape[0], 1, snake_n, 2), dtype=torch.float, device=y_pred.device)

    for i, (cost, p1, p2) in enumerate(zip(costs, p1s, p2s)):
        cost = cost[0,...]
        p1 = p1[0, ...]
        p2 = p2[0, ...]
        path, _ = skimage.graph.route_through_array((cost) ** 4, p1, p2)
        path = torch.tensor(path, dtype=torch.float).unsqueeze(0).unsqueeze(0)
        path = tensor_points_to_init_snake(path, snake_n)

        out_points[i, ...] = path

    out = tensor_snake_to_snake_and_conf(out_points, curve_map)

    return out

def get_lv_antsep_rv_path(y_pred: torch.Tensor, keypoint_names):

    snake_n = 20

    curve_idx = keypoint_names.index("curve-lv-antsep-rv")
    start_idx = keypoint_names.index("lv-antsep-rv-apex")
    end_idx = keypoint_names.index("lv-antsep-rv-base")

    curve_map = y_pred[:, curve_idx, :, :].unsqueeze(1)
    start_map = y_pred[:, start_idx, :, :].unsqueeze(1)
    end_map = y_pred[:, end_idx, :, :].unsqueeze(1)

    start_coord, start_val = get_point(start_map)
    end_coord, end_val = get_point(end_map)


    costs = torch.clamp(curve_map, 0, 0.8)
    costs = (1 - (costs / 0.8)) * 100

    costs = costs.cpu().detach().numpy()
    p1s = start_coord.int().cpu().detach().numpy()
    p2s = end_coord.int().cpu().detach().numpy()

    out_points = torch.zeros((y_pred.shape[0], 1, snake_n, 2), dtype=torch.float, device=y_pred.device)

    for i, (cost, p1, p2) in enumerate(zip(costs, p1s, p2s)):
        cost = cost[0,...]
        p1 = p1[0, ...]
        p2 = p2[0, ...]
        path, _ = skimage.graph.route_through_array((cost) ** 4, p1, p2)
        path = torch.tensor(path, dtype=torch.float).unsqueeze(0).unsqueeze(0)
        path = tensor_points_to_init_snake(path, snake_n)

        out_points[i, ...] = path

    out = tensor_snake_to_snake_and_conf(out_points, curve_map)

    return out

def get_lv_post_endo_path(y_pred: torch.Tensor, keypoint_names):

    snake_n = 20

    curve_idx = keypoint_names.index("curve-lv-post-endo")
    start_idx = keypoint_names.index("lv-post-endo-apex")
    end_idx = keypoint_names.index("lv-post-endo-base")

    curve_map = y_pred[:, curve_idx, :, :].unsqueeze(1)
    start_map = y_pred[:, start_idx, :, :].unsqueeze(1)
    end_map = y_pred[:, end_idx, :, :].unsqueeze(1)

    start_coord, start_val = get_point(start_map)
    end_coord, end_val = get_point(end_map)


    costs = torch.clamp(curve_map, 0, 0.8)
    costs = (1 - (costs / 0.8)) * 100

    costs = costs.cpu().detach().numpy()
    p1s = start_coord.int().cpu().detach().numpy()
    p2s = end_coord.int().cpu().detach().numpy()

    out_points = torch.zeros((y_pred.shape[0], 1, snake_n, 2), dtype=torch.float, device=y_pred.device)

    for i, (cost, p1, p2) in enumerate(zip(costs, p1s, p2s)):
        cost = cost[0,...]
        p1 = p1[0, ...]
        p2 = p2[0, ...]
        path, _ = skimage.graph.route_through_array((cost) ** 4, p1, p2)
        path = torch.tensor(path, dtype=torch.float).unsqueeze(0).unsqueeze(0)
        path = tensor_points_to_init_snake(path, snake_n)

        out_points[i, ...] = path

    out = tensor_snake_to_snake_and_conf(out_points, curve_map)

    return out

def get_lv_post_epi_path(y_pred: torch.Tensor, keypoint_names):

    snake_n = 20

    curve_idx = keypoint_names.index("curve-lv-post-epi")
    start_idx = keypoint_names.index("lv-post-epi-apex")
    end_idx = keypoint_names.index("lv-post-epi-base")

    curve_map = y_pred[:, curve_idx, :, :].unsqueeze(1)
    start_map = y_pred[:, start_idx, :, :].unsqueeze(1)
    end_map = y_pred[:, end_idx, :, :].unsqueeze(1)

    start_coord, start_val = get_point(start_map)
    end_coord, end_val = get_point(end_map)


    costs = torch.clamp(curve_map, 0, 0.8)
    costs = (1 - (costs / 0.8)) * 100

    costs = costs.cpu().detach().numpy()
    p1s = start_coord.int().cpu().detach().numpy()
    p2s = end_coord.int().cpu().detach().numpy()

    out_points = torch.zeros((y_pred.shape[0], 1, snake_n, 2), dtype=torch.float, device=y_pred.device)

    for i, (cost, p1, p2) in enumerate(zip(costs, p1s, p2s)):
        cost = cost[0,...]
        p1 = p1[0, ...]
        p2 = p2[0, ...]
        path, _ = skimage.graph.route_through_array((cost) ** 4, p1, p2)
        path = torch.tensor(path, dtype=torch.float).unsqueeze(0).unsqueeze(0)
        path = tensor_points_to_init_snake(path, snake_n)

        out_points[i, ...] = path

    out = tensor_snake_to_snake_and_conf(out_points, curve_map)

    return out
