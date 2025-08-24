import pandas as pd

lookup_path_al = "/home/leo/Coding/Diffusion/Übertragungstabelle_Al_Factsage900°C.csv"
lookup_path_fe = "/home/leo/Coding/Diffusion/Übertragungstabelle_Fe_Factsage900°C.csv"

lookup_al = pd.read_csv(lookup_path_al, sep=";", decimal=".")["900"].to_list()
lookup_fe = pd.read_csv(lookup_path_fe, sep=";", decimal=".")["900"].to_list()


with open("lookup.h", "w") as f:
    f.write("double lookup_al[] = {%s};\n" % ",".join([str(x) for x in lookup_al]))
    f.write("double lookup_fe[] = {%s};\n" % ",".join([str(x) for x in lookup_fe]))
