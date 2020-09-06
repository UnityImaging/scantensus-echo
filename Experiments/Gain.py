from pathlib import Path

import imageio

import matplotlib.pyplot as plt


IMAGE_PATH = Path("/") / "Volumes" / "Matt-Data" / "Projects-Code" / "scantensus-experiments" / "exp1" / "test.png"

img = imageio.imread(IMAGE_PATH)
img = img / 255.0

plt.imshow(img)
plt.show()