import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

infile = "lookup\Activities_Al-Fe_edited.csv"

df = pd.read_csv(infile, sep=",")

aAl_650 = np.array(df["a-Al(650K)"])
aAl_800 = np.array(df["a-Al(800K)"])

# T = 700

val = 1/3

aAl_700 = aAl_650 + (aAl_800 - aAl_650) * val

# T = 750

val2 = 2/3

aAl_750 = aAl_650 + (aAl_800 - aAl_650) * val2

# Plot

plt.plot(aAl_650, label="650")
plt.plot(aAl_700, label="700")
plt.plot(aAl_750, label="750")
plt.plot(aAl_800, label="800")
plt.legend(loc=10)

plt.show()

# activities for Fe

aFe_650 = 1 - aAl_650
aFe_700 = 1 - aAl_700
aFe_750 = 1 - aAl_750
aFe_800 = 1 - aAl_800

csv_path = "lookup\Activities_Al-Fe_complete.csv"
name_col = "a-Al(650K)"
write_column_first(csv_path, aAl_650, name_col)
name_col = "a-Fe(650K)"
write_column(csv_path, aFe_650, name_col)
name_col = "a-Al(700K)"
write_column(csv_path, aAl_700, name_col)
name_col = "a-Fe(700K)"
write_column(csv_path, aFe_700, name_col)
name_col = "a-Al(750K)"
write_column(csv_path, aAl_750, name_col)
name_col = "a-Fe(750K)"
write_column(csv_path, aFe_750, name_col)
name_col = "a-Al(800K)"
write_column(csv_path, aAl_800, name_col)
name_col = "a-Fe(800K)"
write_column(csv_path, aFe_800, name_col)