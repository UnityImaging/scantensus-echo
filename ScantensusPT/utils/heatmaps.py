from typing import Tuple, Optional, List

import torch
import torch.nn.functional as F

@torch.jit.script
def create_meshgrid(
        height: int,
        width: int,
        normalized_coordinates: bool = True,
        device: Optional[torch.device] = torch.device('cpu')) -> torch.Tensor:
    """Generates a coordinate grid for an image.

    When the flag `normalized_coordinates` is set to True, the grid is
    normalized to be in the range [-1,1] to be consistent with the pytorch
    function grid_sample.
    http://pytorch.org/docs/master/nn.html#torch.nn.functional.grid_sample

    Args:
        height (int): the image height (rows).
        width (int): the image width (cols).
        normalized_coordinates (bool): whether to normalize
          coordinates in the range [-1, 1] in order to be consistent with the
          PyTorch function grid_sample.

    Return:
        torch.Tensor: returns a grid tensor with shape :math:`(1, H, W, 2)`.
    """
    # generate coordinates
    ys: Optional[torch.Tensor] = None
    xs: Optional[torch.Tensor] = None

    if normalized_coordinates:
        ys = torch.linspace(-1, 1, height, device=device, dtype=torch.float)
        xs = torch.linspace(-1, 1, width, device=device, dtype=torch.float)
    else:
        ys = torch.linspace(0, height - 1, height, device=device, dtype=torch.float)
        xs = torch.linspace(0, width - 1, width, device=device, dtype=torch.float)

    # generate grid by stacking coordinates
    base_grid: torch.Tensor = torch.stack(torch.meshgrid([ys, xs]))  # 2xHxW

    return torch.unsqueeze(base_grid, dim=0).permute(0, 2, 3, 1)  # 1xHxWx2


@torch.jit.script
def render_gaussian_dot(
        mean: torch.Tensor,
        std: torch.Tensor,
        size: Tuple[int, int]
) -> torch.Tensor:
    r"""Renders the PDF of a 2D Gaussian distribution.

    mean is y, x
    std is y,x
    size is height,width

    Shape:
        - `mean`: :math:`(*, 2)`
        - `std`: :math:`(*, 2)`. Should be able to be broadcast with `mean`.
        - Output: :math:`(*, H, W)`
    """
    if not (std.dtype == mean.dtype and std.device == mean.device):
        raise TypeError("Expected inputs to have the same dtype and device")

    height, width = size

    mean = mean.unsqueeze(-2).unsqueeze(-2)
    std = std.unsqueeze(-2).unsqueeze(-2)

    grid: torch.Tensor = create_meshgrid(height=height,
                                         width=width,
                                         normalized_coordinates=False,
                                         device=mean.device)
    grid = grid.to(mean.dtype)

    delta = (grid - mean)
    k = -0.5 * (delta / std) ** 2
    gauss = torch.exp(torch.sum(k, dim=-1))

    return gauss

@torch.jit.script
def render_gaussian_curve(
        mean: torch.Tensor,
        std: torch.Tensor,
        size: Tuple[int, int]
) -> torch.Tensor:
    r"""Renders the PDF of a 2D Gaussian distribution.

    mean is y, x
    std is y,x
    size is height,width

    Shape:
        - `mean`: :math:`(*, 2)`
        - `std`: :math:`(*, 2)`. Should be able to be broadcast with `mean`.
        - Output: :math:`(*, H, W)`
    """
    if not (std.dtype == mean.dtype and std.device == mean.device):
        raise TypeError("Expected inputs to have the same dtype and device")

    height, width = size

    mean = mean.unsqueeze(-2).unsqueeze(-2)
    std = std.unsqueeze(-2).unsqueeze(-2)

    grid: torch.Tensor = create_meshgrid(height=height,
                                         width=width,
                                         normalized_coordinates=False,
                                         device=mean.device)
    grid = grid.to(mean.dtype)

    delta = (grid - mean)
    k = ((delta / std) ** 2)
    k = -0.5 * k.sum(dim=-1).min(dim=-3)[0]
    gauss = torch.exp(k)

    return gauss


@torch.jit.script
def render_gaussian_dot_u(
        point: torch.Tensor,
        std: torch.Tensor,
        size: Tuple[int, int],
        mul: float = 255.0
) -> torch.Tensor:

    gauss = render_gaussian_dot(mean=point, std=std, size=size)
    return gauss.mul(mul).to(torch.uint8)

@torch.jit.script
def render_gaussian_dot_f(
        point: torch.Tensor,
        std: torch.Tensor,
        size: Tuple[int, int],
        mul: float = 1.0
) -> torch.Tensor:

    gauss = render_gaussian_dot(mean=point, std=std, size=size)
    return gauss.mul(mul)

@torch.jit.script
def render_gaussian_curve_u(
        points: torch.Tensor,
        std: torch.Tensor,
        size: Tuple[int, int],
        mul: float = 255.0
) -> torch.Tensor:

    gauss = render_gaussian_curve(mean=points, std=std, size=size)
    return gauss.mul(mul).to(torch.uint8)

@torch.jit.script
def render_gaussian_curve_f(
        points: torch.Tensor,
        std: torch.Tensor,
        size: Tuple[int, int],
        mul: float = 1.0
) -> torch.Tensor:
    gauss = render_gaussian_curve(mean=points, std=std, size=size)
    return gauss.mul(mul)





def gaussian(window_size: int, sigma: torch.tensor):
    sigma = sigma.unsqueeze(-1)

    x = torch.arange(window_size, device=sigma.device).float() - window_size // 2
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp((-x.pow(2.0) / (2 * sigma ** 2)))
    return gauss / gauss.sum(dim=-1, keepdim=True)


def get_gaussian_kernel2d(
        kernel_size: Tuple[int, int],
        sigma: torch.Tensor,
        force_even: bool = False) -> torch.Tensor:

    ksize_y, ksize_x = kernel_size
    sigma_y = sigma[..., 0]
    sigma_x = sigma[..., 1]

    kernel_x: torch.Tensor = gaussian(ksize_x, sigma_x)
    kernel_y: torch.Tensor = gaussian(ksize_y, sigma_y)
    kernel_2d: torch.Tensor = torch.matmul(
        kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-2)
    )
    return kernel_2d

def compute_padding(kernel_size: Tuple[int, int]) -> List[int]:
    """Computes padding tuple."""
    # 4 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    assert len(kernel_size) == 2, kernel_size
    computed = [k // 2 for k in kernel_size]

    # for even kernels we need to do asymetric padding :(
    return [computed[1] - 1 if kernel_size[0] % 2 == 0 else computed[1],
            computed[1],
            computed[0] - 1 if kernel_size[1] % 2 == 0 else computed[0],
            computed[0]]

def gaussian_blur2d(x: torch.tensor,
                    kernel_size: Tuple[int, int],
                    sigma: torch.Tensor,
                    ):
    if x.ndim != 4:
        raise Exception

    b, c, h, w = x.shape

    filter = get_gaussian_kernel2d(kernel_size, sigma)
    filter = filter.unsqueeze(1)

    filter_height, filter_width = filter.shape[-2:]
    padding_shape: List[int] = compute_padding((filter_height, filter_width))
    input_pad: torch.Tensor = F.pad(x, padding_shape, mode='constant')

    out = F.conv2d(input_pad, filter, groups=c, padding=0, stride=1)

    return out

def gaussian_blur2d_norm(y_pred: torch.Tensor,
                         kernel_size: Tuple[int, int],
                         sigma: torch.Tensor
                         ):

    max_y_pred = torch.max(y_pred.reshape(y_pred.shape[0], y_pred.shape[1], -1), dim=2, keepdim=True)[0].unsqueeze(3)

    y_pred = gaussian_blur2d(x=y_pred, kernel_size=kernel_size, sigma=sigma)

    max_y__pred = torch.max(y_pred.reshape(y_pred.shape[0], y_pred.shape[1], -1), dim=2, keepdim=True)[0].unsqueeze(3)
    min_y__pred = torch.min(y_pred.reshape(y_pred.shape[0], y_pred.shape[1], -1), dim=2, keepdim=True)[0].unsqueeze(3)

    y_pred = ((y_pred - min_y__pred) / (max_y__pred - min_y__pred)) * max_y_pred

    return y_pred