import numpy as np

import scipy.interpolate
import skimage.graph

import torch
import torch.nn
import torch.nn.functional

import scipy.signal


def get_path_len(points):
    # in is [batch, channels, points, [y, x, conf]]
    # no neet to supply conf
    # out is [batch, channels]
    distance = torch.cumsum(torch.sqrt(torch.sum((points[..., 1:, 0:2] - points[..., :-1, 0:2]) ** 2, dim=-1)), dim=-1)
    return distance[..., -1]


def get_path(logits: torch.Tensor,
             p1s: torch.Tensor,
             p2s: torch.Tensor,
             snake_n_knots: int,
             snake_n_points: int,
             period: float,
             temporal_smooth=False):

    batch_size, channels, height, width = logits.shape

    if temporal_smooth:
        new_logits = torch.empty_like(logits)
        if batch_size > 1:
            new_logits[0, ...] = logits[0, ...] * 0.75 + logits[1, ...] * 0.25
            new_logits[-1, ...] = logits[-1, ...] * 0.75 + logits[-2, ...] * 0.25

            if batch_size > 2:
                new_logits[1:-1, ...] = logits[1:-1, ...] * 0.5 + logits[0:-2, ...] * 0.25 + logits[2:, ...] * 0.25

            logits = new_logits

    costs = (1 - logits) ** 4

    logits_np = logits.cpu().detach().numpy()

    costs = costs.detach().cpu().numpy()
    p1s = p1s.detach().cpu().numpy()
    p2s = p2s.detach().cpu().numpy()

    out_points = torch.zeros((batch_size, 1, snake_n_points, 3), dtype=torch.float, device=logits.device)

    for i in range(batch_size):
        for j in range(channels):
            try:
                logit = logits_np[i, j, ...]
                cost = costs[i, j, ...]
                p1 = p1s[i, j, ...]
                p2 = p2s[i, j, ...]

                if p1[..., 2] < 0.05:
                    raise Exception

                if p2[..., 2] < 0.05:
                    raise Exception

                path, total_cost = skimage.graph.route_through_array(cost, p1[..., 0:2].astype(np.int32), p2[..., 0:2].astype(np.int32))

                mean_score = 1 - (total_cost/len(path))**(1/4)

                if mean_score < 0.01:
                    raise Exception

                path = np.array(path).T

                path_weight = logit[path[0], path[1]]
                path_weight = path_weight ** 2
                path_weight[0] = 10
                path_weight[-1] = 10

                tck, u = scipy.interpolate.splprep(x=path, w=path_weight, k=3, t=snake_n_knots)

                u = np.linspace(0, 1, snake_n_points)

                new_points = scipy.interpolate.splev(u, tck)

                new_points = np.array(new_points).T

                path = torch.tensor(new_points, dtype=torch.float)

                out_points[i, j, :, 0:2] = path
                out_points[i, j, :, 2] = mean_score
            except Exception as e:
                out_points[i, j, :, 0:2] = float('NaN')
                out_points[i, j, :, 2] = float('NaN')
                continue

    return out_points


def get_path_via(logits: torch.Tensor,
                 p1s: torch.Tensor,
                 p2s: torch.Tensor,
                 p3s: torch.Tensor,
                 snake_n_knots: int,
                 snake_n_points: int,
                 period: float,
                 temporal_smooth=False):

    filter_width = 2 * (int(period) // 8) + 1
    filter_width = max(filter_width, 5)

    batch_size, channels, height, width = logits.shape
    device = logits.device

    if temporal_smooth:
        new_logits = torch.empty_like(logits)
        if batch_size > 1:
            new_logits[0, ...] = logits[0, ...] * 0.75 + logits[1, ...] * 0.25
            new_logits[-1, ...] = logits[-1, ...] * 0.75 + logits[-2, ...] * 0.25

            if batch_size > 2:
                new_logits[1:-1, ...] = logits[1:-1, ...] * 0.5 + logits[0:-2, ...] * 0.25 + logits[2:, ...] * 0.25

            logits = new_logits

    costs = (1.0 - logits) ** 8

    logits_np = logits.detach().cpu().numpy()
    
    costs = costs.detach().cpu().numpy()
    p1s = p1s.detach().cpu().numpy()
    p2s = p2s.detach().cpu().numpy()
    p3s = p3s.detach().cpu().numpy()

    out_points = torch.zeros((batch_size, channels, snake_n_points, 3), dtype=torch.float, device=device)

    for i in range(batch_size):
        for j in range(channels):
            try:
                logit = logits_np[i, j, ...]
                cost = costs[i, j, ...]
                p1 = p1s[i, j, ...]
                p2 = p2s[i, j, ...]
                p3 = p3s[i, j, ...]

                if p1[..., 2] < -0.05:
                    print(f"p1 out")
                    raise Exception

                if p2[..., 2] < -0.05:
                    print(f"p2 out")
                    raise Exception

                if p3[..., 2] < -0.05:
                    print(f"p3 out")
                    raise Exception

                path_a, total_cost_a = skimage.graph.route_through_array(cost, p1[..., 0:2].astype(np.int32), p2[..., 0:2].astype(np.int32))
                path_b, total_cost_b = skimage.graph.route_through_array(cost, p2[..., 0:2].astype(np.int32), p3[..., 0:2].astype(np.int32))

                mean_score = 1 - ((total_cost_a+total_cost_b)/(len(path_a)+len(path_b)))**(1/4)
                mean_score = float(mean_score)

                if mean_score < 0.01:
                    raise Exception

                path_a.extend(path_b[1:])

                path = np.array(path_a).T

                path_weight = logit[path[0], path[1]]
                path_weight = path_weight ** 2
                path_weight[0] = 10
                path_weight[-1] = 10

                tck, u = scipy.interpolate.splprep(x=path, w=path_weight, k=3, t=snake_n_knots)

                u = np.linspace(0, 1, snake_n_points)

                new_points = scipy.interpolate.splev(u, tck)

                new_points = np.array(new_points).T

                path = torch.tensor(new_points, dtype=torch.float, device=device)

                out_points[i, j, :, 0:2] = path
                out_points[i, j, :, 2] = mean_score
            except Exception as e:
                out_points[i, j, :, 0:2] = float('NaN')
                out_points[i, j, :, 2] = float('NaN')
                continue

    return out_points
