import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import numpy as np


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

# 650K

infile = "lookup/Activities_Al-Fe_650K.csv"

df = pd.read_csv(infile, sep=",")

original_data = np.array(df["a-Al(A1)"])
org_a = original_data[:626]
org_b = original_data[626:]
original_data_log = np.log(original_data)
print(original_data_log.shape)

monotone = make_monotone(np.array(df["a-Al(A1)"]))
monotone2 = make_monotone(np.array(df["a-Al(A1)"])[::-1])[::-1]

alpha = 0.5
total = monotone * alpha + monotone2 * (1 - alpha)

monotone_log = np.log(monotone)
monotone2_log = np.log(monotone2)
total_log = np.log(total)

plt.plot(monotone_log, label="Rising", marker=".")
plt.plot(monotone2_log, label="Falling")
# plt.plot(total_log, label="Combined")
plt.plot(original_data_log, label="Original")
# plt.plot(np.array(df["a-Fe-alpha(A2)"]), label="Raw")
plt.legend()

plt.show()

# csv_path = "lookup\Activities_Al-Fe_edited.csv"
# name_col = "a-Al(650K)"
# write_column_first(csv_path, total, name_col)

# original_data_b = np.array(df["a-Fe-alpha(A2)"])
# original_data_b_log = np.log(original_data_b)

# monotone_b = make_monotone(np.array(df["a-Fe-alpha(A2)"]))
# monotone2_b = make_monotone(np.array(df["a-Fe-alpha(A2)"])[::-1])[::-1]

# alpha_b = 0.5
# total_b = monotone_b * alpha_b + monotone2_b * (1 - alpha_b)

# plt.plot(monotone_b, label="Rising", marker=".")
# plt.plot(monotone2_b, label="Falling")
# plt.plot(total_b, label="Combined")
# plt.plot(original_data_b_log, label="Original")
# plt.plot(np.array(df["a-Fe-alpha(A2)"]), label="Raw")
# plt.legend()

# plt.show()

# csv_path = "lookup\Activities_Al-Fe_edited.csv"
# name_col = "a-Fe-alpha(650K)"
# write_column(csv_path, total_b, name_col)

# 800K

# infile = "lookup\Activities_Al-Fe_800K.csv"

# df = pd.read_csv(infile, sep=",")

# original_data = np.array(df["a-Al(A1)"])

# monotone = make_monotone(np.array(df["a-Al(A1)"]))
# monotone2 = make_monotone(np.array(df["a-Al(A1)"])[::-1])[::-1]

# alpha = 0.5
# total = monotone * alpha + monotone2 * (1 - alpha)

# plt.plot(monotone, label="Rising", marker=".")
# plt.plot(monotone2, label="Falling")
# plt.plot(total, label="Combined")
# plt.plot(original_data, label="Original")
# # plt.plot(np.array(df["a-Fe-alpha(A2)"]), label="Raw")
# plt.legend()

# plt.show()

# csv_path = "lookup\Activities_Al-Fe_edited.csv"
# name_col = "a-Al(800K)"
# write_column(csv_path, total, name_col)

# original_data_b = np.array(df["a-Fe-alpha(A2)"])

# monotone_b = make_monotone(np.array(df["a-Fe-alpha(A2)"]))
# monotone2_b = make_monotone(np.array(df["a-Fe-alpha(A2)"])[::-1])[::-1]

# alpha_b = 0.5
# total_b = monotone_b * alpha_b + monotone2_b * (1 - alpha_b)

# plt.plot(monotone_b, label="Rising", marker=".")
# plt.plot(monotone2_b, label="Falling")
# plt.plot(total_b, label="Combined")
# plt.plot(original_data_b, label="Original")
# # plt.plot(np.array(df["a-Fe-alpha(A2)"]), label="Raw")
# plt.legend()

# plt.show()

# csv_path = "lookup\Activities_Al-Fe_edited.csv"
# name_col = "a-Fe-alpha(800K)"
# write_column(csv_path, total_b, name_col)