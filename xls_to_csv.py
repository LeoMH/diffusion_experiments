import pandas as pd

read_file = pd.read_excel("lookup/Activities_Cu-Sn_226°C-234°C.xls", "Sheet1")

read_file.to_csv("lookup/Activities_Cu-Sn_226°C-234°C.csv", index=None, header=True)