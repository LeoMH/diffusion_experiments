import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import cv2

def monotonically_increasing(l):
    return all(x <= y for x, y in zip(l, l[1:]))

# data = np.array([[0,0,0,0,0,0,1,1],
#                  [0,0,0,0,1,1,0,0],
#                  [0,0,0,1,0,0,0,0],
#                  [0,0,0,1,0,0,0,0],
#                  [0,0,1,0,0,0,0,0],
#                  [0,1,0,0,0,0,0,0],
#                  [1,0,0,0,0,0,0,0],
#                  [1,0,0,0,0,0,0,0]])

img_path = ("activity_test.tiff")
data_raw = cv2.imread(img_path, 0)
data_255 = cv2.bitwise_not(data_raw)
data = data_255 / 255.0

data = data[::-1]

idx = np.argwhere(data>0.4)

x = idx[:,1]
y = idx[:,0]

unique_x, inverse = np.unique(x, return_inverse=True)

mean_y = np.bincount(inverse, weights=y) / np.bincount(inverse)


x_values = np.hstack([np.array(0), unique_x, np.array(unique_x.shape[0]+1)])
y_values = np.hstack([np.array(0.0), mean_y, np.array(data.shape[0])])
print(y_values.shape)

print(monotonically_increasing(y_values))

xmin, xmax = 0.0, 658.0
a, b = 1e-12, 1.0

y_scaled = a + (y_values - xmin) * (b - a) / (xmax - xmin)

print(y_scaled)

plt.plot(x_values, y_scaled)
plt.show()