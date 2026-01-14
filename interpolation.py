import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

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

infile = "lookup/Activities_Al-Fe_edited.csv"

df = pd.read_csv(infile, sep=",")

aAl_650_d = np.array(df["a-Al(650K)"])
aAl_800_d = np.array(df["a-Al(800K)"])

aAl_650 = np.log(aAl_650_d)
aAl_800 = np.log(aAl_800_d)

A = (np.log(aAl_650_d/aAl_800_d))/((1/650)-(1/800))
B = np.log(aAl_650_d) - A * (1/650)

# T = 700

aAl_700 = (A/700) + B
aAl_700_d = np.exp(aAl_700)
 
# T = 750

aAl_750 = (A/750) + B
aAl_750_d = np.exp(aAl_750)

# Plot

plt.plot(aAl_650, label="650")
plt.plot(aAl_700, label="700")
plt.plot(aAl_750, label="750")
plt.plot(aAl_800, label="800")
plt.legend(loc=10)

plt.show()

# activities for Fe

aFe_650_d = np.array(df["a-Fe(650K)"])
aFe_800_d = np.array(df["a-Fe(800K)"])

aFe_650 = np.log(aFe_650_d)
aFe_800 = np.log(aFe_800_d)

A = (np.log(aFe_650_d/aFe_800_d))/((1/650)-(1/800))
B = np.log(aFe_650_d) - A * (1/650)

# T = 700

aFe_700 = (A/700) + B
aFe_700_d = np.exp(aFe_700)
 
# T = 750

aFe_750 = (A/750) + B
aFe_750_d = np.exp(aFe_750)

# Plot

plt.plot(aFe_650, label="650")
plt.plot(aFe_700, label="700")
plt.plot(aFe_750, label="750")
plt.plot(aFe_800, label="800")
plt.legend(loc=10)

plt.show()

csv_path = "lookup/Activities_Al-Fe_complete.csv"
name_col = "a-Al(650K)"
write_column_first(csv_path, aAl_650_d, name_col)
name_col = "a-Fe(650K)"
write_column(csv_path, aFe_650_d, name_col)
name_col = "a-Al(700K)"
write_column(csv_path, aAl_700_d, name_col)
name_col = "a-Fe(700K)"
write_column(csv_path, aFe_700_d, name_col)
name_col = "a-Al(750K)"
write_column(csv_path, aAl_750_d, name_col)
name_col = "a-Fe(750K)"
write_column(csv_path, aFe_750_d, name_col)
name_col = "a-Al(800K)"
write_column(csv_path, aAl_800_d, name_col)
name_col = "a-Fe(800K)"
write_column(csv_path, aFe_800_d, name_col)