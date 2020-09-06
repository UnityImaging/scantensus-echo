import matplotlib.pyplot as plt

import numpy as np
import random
from skimage.transform import warp, AffineTransform

test = np.random.random(100*100*10).reshape((100,100,10))

transform = AffineTransform(#scale=(0.8, 1.1),
                            #rotation=np.deg2rad(10),
                            #shear=random.uniform(-0.1, 0.1),
                            translation=(20,20))


image = warp(test, transform.inverse, output_shape=(100, 100))