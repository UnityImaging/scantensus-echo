import os

import numpy as np
import matplotlib.pyplot as plt

import imageio

import cv2

import scipy.stats


test_fl = "/Volumes/Matt-Temp/test_png/01-2bded1bdcca6de6a067433560938b8f6911d09f39990f81e5370a669b96970a7-0065.png"

img = imageio.imread(test_fl)[...,0]

plt.imshow(img)
plt.show()

img_edge = cv2.Canny(img, 100, 250, 50, L2gradient=True)
plt.imshow(img_edge)
plt.show()

img_edge = cv2.Laplacian(img, cv2.CV_32F, ksize=21)
plt.imshow(img_edge)
plt.show()



if False:
    box_height = 5
    box_width = 5
    img_height = img.shape[0]
    img_width = img.shape[1]

    out_v = np.zeros_like(img).astype(np.float32)

    for i in np.arange(box_height, img_height - box_height):
        print(i)
        for j in np.arange(box_width, img_width - box_width):
            upper_box = img[i - box_height:i, j - 2: j+3]
            lower_box = img[i:i + box_height, j - 2: j+3]

            a = scipy.stats.entropy(upper_box.flatten(), qk=lower_box.flatten())
            b = scipy.stats.entropy(lower_box.flatten(), qk=upper_box.flatten())

            out_v[i,j] = a+b

    out_v[np.isnan(out_v)] = 0

    out_h = np.zeros_like(img).astype(np.float32)







    for i in np.arange(box_height, img_height - box_height):
        print(i)
        for j in np.arange(box_width, img_width - box_width):
            left_box = img[i-2: i+3, j-5: j]
            right_box = img[i-2:i+3, j: j+5]

            a = scipy.stats.entropy(left_box.flatten(), qk=right_box.flatten())
            b = scipy.stats.entropy(right_box.flatten(), qk=left_box.flatten())

            out_h[i,j] = a+b

    out_v[np.isnan(out_v)] = 1
    out_h[np.isnan(out_h)] = 1

    plt.imshow(out_v)
    plt.show()
    plt.imshow(out_h)
    plt.show()
    plt.imshow(out_v+out_h)
    plt.show()

lead = 10
spread = 2
box_size = lead * (2 * spread + 1)
I = np.exp(2 * (img / 255))

h_features = np.zeros_like(I)
v_features = np.zeros_like(I)

for row in np.arange(spread, I.shape[0] - spread):
    for col in np.arange(lead, I.shape[1] - lead):
        var1 = np.sum(I[row - spread:row + spread + 1, col - lead:col]) / 2. * box_size
        var2 = np.sum(I[row - spread:row + spread + 1, col: col + lead]) / 2. * box_size
        i2var1 = 1. / (2. * var1)
        i2var2 = 1. / (2. * var2)
        J = 0.5 * np.exp(-i2var1) * (-np.log(2 * var1) + np.log(2 * var2) - 1 - i2var1 + var1 / var2 + i2var2) + \
            0.5 * np.exp(-i2var2) * (-np.log(2 * var2) + np.log(2 * var1) - 1 - i2var2 + var2 / var1 + i2var1)
        h_features[row, col] = J
        h_features[row, col] = var1 - var2

for col in np.arange(spread, I.shape[1] - spread):
    for row in np.arange(lead, I.shape[0] - lead):
        var1 = np.sum(I[row - lead:row, col - spread:col + spread + 1]) / 2. * box_size
        var2 = np.sum(I[row:row + lead, col - spread:col + spread + 1]) / 2. * box_size
        i2var1 = 1. / (2. * var1)
        i2var2 = 1. / (2. * var2)
        J = 0.5 * np.exp(-i2var1) * (-np.log(2 * var1) + np.log(2 * var2) - 1 - i2var1 + var1 / var2 + i2var2) + \
            0.5 * np.exp(-i2var2) * (-np.log(2 * var2) + np.log(2 * var1) - 1 - i2var2 + var2 / var1 + i2var1)
        v_features[row, col] = J
        v_features[row, col] = var1 - var2

plt.imshow(v_features)
plt.show()
plt.imshow(h_features)
plt.show()
plt.imshow(h_features + v_features)
plt.show()

print()