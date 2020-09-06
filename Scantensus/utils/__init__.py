import numpy as np


###From https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python/17201686#17201686

def matlab_style_gauss2D_kernel(shape, sigma):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


##From https://stackoverflow.com/questions/55643675/how-do-i-implement-gaussian-blurring-layer-in-keras

def get_blur_kernel(shape=(5,5), sigma=1, channels=3):

    kernel_weights = matlab_style_gauss2D_kernel(shape=shape, sigma=sigma)
    kernel_weights = np.expand_dims(kernel_weights, axis=-1)
    kernel_weights = np.repeat(kernel_weights, channels, axis=-1)  # apply the same filter on all the input channels
    kernel_weights = np.expand_dims(kernel_weights, axis=-1)

    return kernel_weights

