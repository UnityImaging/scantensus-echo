import os
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import skimage.segmentation
from Scantensus.utils.segmentation import active_contour
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
import skimage.filters
import skimage.graph
from skimage.segmentation import morphological_geodesic_active_contour

import torch
import torch.nn
import torch.nn.functional
import torch.optim

HOST = 'matt-laptop'

###############
if HOST == "thready":
    DATA_DIR = Path("/") / "mnt" / "Storage" / "scantensus-data"
    OUTPUT_DIR = Path("/") / "mnt" / "Storage" / "matt-output"
elif HOST == "matt-laptop":
    DATA_DIR = Path("/") / "Users" / "matthew" / "Box" / "scantensus-data"
    OUTPUT_DIR = Path("/") / "Volumes" / "Matt-Data" / "matt-output"
else:
    raise Exception
################

JSON_KEYS_PATH = DATA_DIR / "labels" / "unity" / "keys.json"



d = np.load("/Volumes/Matt-Data/snake_test.npz")

epi = d[0,...,20]

endo = d[0,...,19]

endo_inv = 1 - endo
endo_inv = np.clip(endo_inv, 0, 1)

costs = endo_inv

path, cost = skimage.graph.route_through_array(costs, [400,290], [400, 410])
