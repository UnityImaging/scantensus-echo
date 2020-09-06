from warnings import warn

import numpy as np

import torch
import torch.nn
import torch.nn.functional

def active_contour(image: torch.Tensor,
                   snake: torch.Tensor,
                   alpha=0.01,
                   beta=0.1,
                   w_line=0,
                   w_edge=1,
                   gamma=0.01,
                   bc=None,
                   max_px_move=1.0,
                   max_iterations=2500,
                   convergence=0.5,
                   *,
                   boundary_condition='periodic',
                   coordinates=None):

    """Active contour model.
    Active contours by fitting snakes to features of images. Supports single
    and multichannel 2D images. Snakes can be periodic (for segmentation) or
    have fixed and/or free ends.
    The output snake has the same length as the input boundary.
    As the number of points is constant, make sure that the initial snake
    has enough points to capture the details of the final contour.
    Parameters
    ----------
    image : (N, M) or (N, M, 3) ndarray
        Input image.
    snake : (N, 2) ndarray
        Initial snake coordinates. For periodic boundary conditions, endpoints
        must not be duplicated.
    alpha : float, optional
        Snake length shape parameter. Higher values makes snake contract
        faster.
    beta : float, optional
        Snake smoothness shape parameter. Higher values makes snake smoother.
    w_line : float, optional
        Controls attraction to brightness. Use negative values to attract toward
        dark regions.
    w_edge : float, optional
        Controls attraction to edges. Use negative values to repel snake from
        edges.
    gamma : float, optional
        Explicit time stepping parameter.
    bc : deprecated; use ``boundary_condition``
        DEPRECATED. See ``boundary_condition`` below.
    max_px_move : float, optional
        Maximum pixel distance to move per iteration.
    max_iterations : int, optional
        Maximum iterations to optimize snake shape.
    convergence: float, optional
        Convergence criteria.
    boundary_condition : string, optional
        Boundary conditions for the contour. Can be one of 'periodic',
        'free', 'fixed', 'free-fixed', or 'fixed-free'. 'periodic' attaches
        the two ends of the snake, 'fixed' holds the end-points in place,
        and 'free' allows free movement of the ends. 'fixed' and 'free' can
        be combined by parsing 'fixed-free', 'free-fixed'. Parsing
        'fixed-fixed' or 'free-free' yields same behaviour as 'fixed' and
        'free', respectively.
    coordinates : {'rc' or 'xy'}, optional
        Whether to use rc or xy coordinates. The 'xy' option (current default)
        will be removed in version 0.18.
    Returns
    -------
    snake : (N, 2) ndarray
        Optimised snake, same shape as input parameter.
    References
    ----------
    .. [1]  Kass, M.; Witkin, A.; Terzopoulos, D. "Snakes: Active contour
            models". International Journal of Computer Vision 1 (4): 321
            (1988). :DOI:`10.1007/BF00133570`
    Examples
    --------
    >>> from skimage.draw import circle_perimeter
    >>> from skimage.filters import gaussian
    Create and smooth image:
    >>> img = np.zeros((100, 100))
    >>> rr, cc = circle_perimeter(35, 45, 25)
    >>> img[rr, cc] = 1
    >>> img = gaussian(img, 2)
    Initialize spline:
    >>> s = np.linspace(0, 2*np.pi, 100)
    >>> init = 50 * np.array([np.sin(s), np.cos(s)]).T + 50
    Fit spline to image:
    >>> snake = active_contour(img, init, w_edge=0, w_line=1, coordinates='rc')  # doctest: +SKIP
    >>> dist = np.sqrt((45-snake[:, 0])**2 + (35-snake[:, 1])**2)  # doctest: +SKIP
    >>> int(np.mean(dist))  # doctest: +SKIP
    25
    """

    if bc is not None:
        message = ('The keyword argument `bc` to `active_contour` has been '
                   'renamed. Use `boundary_condition=` instead. `bc` will be '
                   'removed in scikit-image v0.18.')
        warn(message, stacklevel=2)
        boundary_condition = bc

    if coordinates is None:
        message = ('The coordinates used by `active_contour` will change '
                   'from xy coordinates (transposed from image dimensions) to '
                   'rc coordinates in scikit-image 0.18. Set '
                   "`coordinates='rc'` to silence this warning. "
                   "`coordinates='xy'` will restore the old behavior until "
                   '0.18, but will stop working thereafter.')
        warn(message, category=FutureWarning, stacklevel=2)
        coordinates = 'xy'

    if coordinates == 'rc':
        snake_rc = snake
    else:
        snake_rc = snake[:, [1,0]]

    max_iterations = int(max_iterations)
    if max_iterations <= 0:
        raise ValueError("max_iterations should be >0.")
    convergence_order = 10
    valid_bcs = ['periodic', 'free', 'fixed', 'free-fixed', 'fixed-free', 'fixed-fixed', 'free-free']
    if boundary_condition not in valid_bcs:
        raise ValueError("Invalid boundary condition.\n" +
                         "Should be one of: "+", ".join(valid_bcs)+'.')

    if image.ndim == 2:
        img = image.float().unsqueeze(0).unsqueeze(0)
    else:
        img = image.float()

    if snake_rc.ndim == 2:
        snake_rc = snake_rc.float().unsqueeze(0).unsqueeze(0)

    device = img.device
    batch_size = img.shape[0]
    img_height = img.shape[2]
    img_width = img.shape[3]

    n = snake_rc.shape[2]

    # Find edges using sobel:
    if w_edge != 0:
        raise Exception("Sorry - couldn't be bothered to implement sobel filter thing")

    img = img * w_line

    eye_n = torch.eye(n, device=device)

    # Build snake shape matrix for Euler equation
    a = torch.roll(eye_n, -1, dims=0) + \
        torch.roll(eye_n, -1, dims=1) - \
        2*eye_n  # second order derivative, central difference

    b = torch.roll(eye_n, -2, dims=0) + \
        torch.roll(eye_n, -2, dims=1) - \
        4*torch.roll(eye_n, -1, dims=0) - \
        4*torch.roll(eye_n, -1, dims=1) + \
        6*eye_n  # fourth order derivative, central difference

    A = -alpha*a + beta*b

    # Impose boundary conditions different from periodic:
    sfixed = False
    if boundary_condition.startswith('fixed'):
        A[0, :] = 0
        A[1, :] = 0
        A[1, 0] = 1
        A[1, 1] = -2
        A[1, 2] = 1

        sfixed = True

    efixed = False
    if boundary_condition.endswith('fixed'):
        A[-1, :] = 0
        A[-2, :] = 0
        A[-2, -3] = 1
        A[-2, -2] = -2
        A[-2, -1] = 1

        efixed = True

    sfree = False
    if boundary_condition.startswith('free'):
        A[0, :] = 0
        A[0, 0] = 1
        A[0, 1] = -2
        A[0, 2] = 1

        A[1, :] = 0
        A[1, 0] = -1
        A[1, 1] = 3
        A[1, 2] = -3
        A[1, 3] = 1

        sfree = True

    efree = False
    if boundary_condition.endswith('free'):
        A[-1, :] = 0
        A[-1, -3] = 1
        A[-1, -2] = -2
        A[-1, -1] = 1

        A[-2, :] = 0
        A[-2, -4] = -1
        A[-2, -3] = 3
        A[-2, -2] = -3
        A[-2, -1] = 1

        efree = True

    # Only one inversion is needed for implicit spline energy minimization:
    inv = torch.inverse(A + gamma * eye_n)

    # Explicit time stepping for image energy minimization:
    save = torch.empty((batch_size, convergence_order, n, 2), dtype=torch.float, device=device)

    x_sobel = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=device, dtype=torch.float).view(1, 1, 3, 3)
    y_sobel = x_sobel.transpose(2,3)
    sobel_filter = torch.cat((y_sobel, x_sobel), dim=0) # for out_chan, in_chan, height, width
    img_g = torch.nn.functional.conv2d(img, sobel_filter, padding=1) / 4.0

    # I don't quite know why we transpose it - perhaps it is so the matrix stuff works.
    # But you have to - the gradient is 90 degress off the image.
    img_g = img_g.transpose(2,3)

    img_size = torch.tensor([img_height, img_width], dtype=torch.float, device=device)

    for i in range(max_iterations):

        snake_rc_shrink = (2 * snake_rc / img_size) - 1

        f_g = torch.nn.functional.grid_sample(img_g, snake_rc_shrink, "bilinear", 'zeros')
        f_g = f_g.permute(0,2,3,1)

        if sfixed:
            f_g[..., 0, :] = 0
        if efixed:
            f_g[..., -1, :] = 0
        if sfree:
            f_g[..., 0, :] *= 2
        if efree:
            f_g[..., -1, :] *= 2

        yxn = inv @ (gamma * snake_rc + f_g)

        dyx = max_px_move * torch.clamp(yxn-snake_rc, -1, 1)

        if sfixed:
            dyx[..., 0, :] = 0
        if efixed:
            dyx[..., -1, :] = 0

        snake_rc = snake_rc + dyx

        # Convergence criteria needs to compare to a number of previous
        # configurations since oscillations can occur.

        j = i % (convergence_order+1)
        if j < convergence_order:
            save[:, j, :, :] = snake_rc.squeeze(1)
        else:
            dist = torch.sum(torch.abs(save-snake_rc), dim=3)
            dist = torch.max(dist, dim=2)[0]
            dist = torch.min(dist, dim=1)[0]
            dist = torch.max(dist) # this checks all the batches for the worst.
            if dist < convergence:
                break

    if coordinates == 'xy':
        return snake_rc[..., [1,0]]
    else:
        return snake_rc
