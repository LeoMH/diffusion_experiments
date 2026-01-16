import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import cv2

def monotonically_increasing(l):
    return all(x <= y for x, y in zip(l, l[1:]))

def make_monotone(data: np.ndarray):
    rising = data[0] < data[-1]
    last_value = 0
    new_data = np.copy(data)
    for i in range(1, data.shape[0]):
        if data[i] >= data[last_value] if rising else data[i] <= data[last_value]:
            new_data[last_value : i + 1] = np.linspace(
                data[last_value], data[i], i - last_value + 1
            )
            last_value = i
    return new_data

def write_column_first(path, data: np.ndarray, col_name: str):
    data = np.asarray(data).ravel()
    df = pd.DataFrame({col_name: data})
    df.to_csv(path, index=False)

def write_column(path, data: np.ndarray, col_name: str):
    csv_file_path = path
    df = pd.read_csv(csv_file_path) 
    column = data 
    df[f'{col_name}'] = column 
    df.to_csv(csv_file_path, index=False)

# data = np.array([[0,0,0,0,0,0,1,1],
#                  [0,0,0,0,1,1,0,0],
#                  [0,0,0,1,0,0,0,0],
#                  [0,0,0,1,0,0,0,0],
#                  [0,0,1,0,0,0,0,0],
#                  [0,1,0,0,0,0,0,0],
#                  [1,0,0,0,0,0,0,0],
#                  [1,0,0,0,0,0,0,0]])

img_path = ("images/al_fe_system/aFe_800K_log.tiff")
data_raw = cv2.imread(img_path, 0)
data_255 = cv2.bitwise_not(data_raw)
data = data_255 / 255.0

data = data[::-1]

idx = np.argwhere(data>0.4)

x = idx[:,1]
y = idx[:,0]

unique_x, inverse = np.unique(x, return_inverse=True)

mean_y = np.bincount(inverse, weights=y) / np.bincount(inverse)

# mean_y = mean_y[::-1] # reverse array for Al activities
# x_values = np.hstack([np.array(0), unique_x, np.array(unique_x.shape[0]+1)])
# y_values = np.hstack([np.array(0.0), mean_y, np.array(np.max(mean_y))])
x_values = unique_x
y_values = mean_y

print(y_values.shape)

monotone = monotonically_increasing(y_values)

print(monotone)

if monotone == False:
    y_values = make_monotone(y_values)

print(monotonically_increasing(y_values))

xmin, xmax = np.min(y_values), np.max(y_values)
a, b = -16.42908729285, -0.00103153185

y_scaled_log = a + (y_values - xmin) * (b - a) / (xmax - xmin)

plt.plot(x_values, y_scaled_log)
plt.show()

y_scaled = np.exp(y_scaled_log)
plt.plot(x_values, y_scaled)
plt.show()

csv_path = "lookup/Activities_Al-Fe_edited.csv"
name_col = "a-Fe(800K)"
write_column(csv_path, y_scaled, name_col)