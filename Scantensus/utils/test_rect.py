import numpy as np
import scipy
import torch

import imageio
from scipy.interpolate import RectBivariateSpline


d = np.load("/Volumes/Matt-Data/test.npz")

endo = d[0,...,19]

intp = RectBivariateSpline(np.arange(endo.shape[1]),
                           np.arange(endo.shape[0]),
                           endo.T, kx=2, ky=2, s=0)

f_x = intp(np.arange(endo.shape[1]), np.arange(endo.shape[0]), dx=1, grid=True)
f_y = intp(np.arange(endo.shape[1]), np.arange(endo.shape[0]), dy=1, grid=True)

print()

g_y, g_x = np.gradient(endo)


