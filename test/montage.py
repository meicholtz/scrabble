from skimage.util import montage
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(6*3*3).reshape((6,3,3))
print(x)

y = montage(x, grid_shape=(2, 3))
print(y)

plt.imshow(y)
plt.show()
