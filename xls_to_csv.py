import pandas as pd

read_file = pd.read_excel("lookup\Activities_Al-Fe_650K.xls", "Sheet1")

read_file.to_csv("lookup\Activities_Al-Fe_650K.csv", index=None, header=True)

read_file = pd.read_excel("lookup\Activities_Al-Fe_800K.xls", "Sheet1")

read_file.to_csv("lookup\Activities_Al-Fe_800K.csv", index=None, header=True)