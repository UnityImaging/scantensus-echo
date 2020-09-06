import numpy as np

def param2theta(param, w, h):
    # Param will have the translation in pixels. You need to provide the SOURCE image coord.
    # These param need to be the inverse.
    # Effectively this does https://discuss.pytorch.org/t/affine-transformation-matrix-paramters-conversion/19522/13
    # I.e
    # magic = [[2/w, 0, -1],[0, 2/h, -1],[0,0,1]]
    # magic @ param @ magic-1

    theta = np.zeros([2,3])
    theta[0,0] = param[0,0]
    theta[0,1] = param[0,1]*h/w
    theta[0,2] = param[0,2]*2/w + theta[0,0] + theta[0,1] - 1
    theta[1,0] = param[1,0]*w/h
    theta[1,1] = param[1,1]
    theta[1,2] = param[1,2]*2/h + theta[1,0] + theta[1,1] - 1
    return theta
