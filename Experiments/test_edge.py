import os
from pathlib import Path

import imageio

import numpy as np

from scipy import ndimage as ndi

from skimage import feature

import matplotlib.pyplot as plt
import skimage.segmentation
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

from skimage.segmentation import morphological_geodesic_active_contour

path = "/Volumes/Matt-Data/Projects-Clone/scantensus-data/validation/unity-plax-1/01-4aaf9972beb73809068541b208ebb5716878dd33bd76796f8f428b52b7debd8f/01-4aaf9972beb73809068541b208ebb5716878dd33bd76796f8f428b52b7debd8f-0000.png"


img = imageio.imread(path)

