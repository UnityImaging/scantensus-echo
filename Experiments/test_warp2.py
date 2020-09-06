import numpy as np
import torch
import math

import skimage.transform
import torch.nn.functional

import matplotlib.pyplot as plt

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

image_np = np.random.random(50*3).reshape((1,10,5,3))

image_t = torch.nn.functional.interpolate(torch.from_numpy(image_np.transpose((0,3,1,2))), scale_factor=(10,10))
image_np = image_t.detach().numpy().transpose(0,2,3,1)



tf = skimage.transform.AffineTransform(scale=(0.3, 0.3),
                                       rotation=np.deg2rad(45),
                                       shear=0,
                                       translation=(0, 0))


rotation = 0
rotation_theta = math.pi / 180 * rotation
tf_rotate = torch.FloatTensor([[math.cos(rotation_theta), -math.sin(rotation_theta), 0],
                               [math.sin(rotation_theta), math.cos(rotation_theta), 0],
                               [0, 0, 1]])

tx = 0
ty = 0
tf_translate = torch.FloatTensor([[1, 0, tx],
                                  [0, 1, ty],
                                  [0, 0, 1]])

sx = 0.5
sy = 1

tf_scale = torch.FloatTensor([[sx, 0, 0],
                              [0, sy, 0],
                              [0, 0, 1]])

shear_deg = 45
shear_theta = math.pi / 180 * shear_deg

tf_shear = torch.FloatTensor([[1, -math.sin(shear_theta), 0],
                              [0, math.cos(shear_theta), 0],
                              [0, 0, 1]])

matrix = tf_shear @ tf_scale @ tf_rotate @ tf_translate


matrix = matrix.inverse()


if matrix.dim() == 2:
    matrix = matrix[:2, :]
    matrix = matrix.unsqueeze(0)
elif matrix.dim() == 3:
    if matrix.size()[1:] == (3, 3):
        matrix = matrix[:, :2, :]

A_batch = matrix[:,:,:2]
if A_batch.size(0) != image_t.size(0):
    A_batch = A_batch.repeat(image_t.size(0),1,1)
b_batch = matrix[:,:,2].unsqueeze(1)

identity = torch.tensor([[[1,0,0],[0,1,0]]]).float()
grid = torch.nn.functional.affine_grid(identity,[1,1,100,100])
grid = grid.reshape([1,10000,2])

grid = grid.bmm(A_batch.transpose(1,2)) + b_batch.expand_as(grid)
grid = grid.reshape([1,100,100,2])

t_image_new_t = torch.nn.functional.grid_sample(image_t.float(),grid)
t_image_new_np = t_image_new_t.detach().numpy().transpose(0,2,3,1)

#sk_image_new_np = skimage.transform.warp(image_np[0,...], tf_final_inverse, output_shape=(200, 200), cval=0)

print("hello")
