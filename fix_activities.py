import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def make_monotone(data: np.ndarray, rising=True):
    last_value = 0
    new_data = np.copy(data)
    for i in range(1, data.shape[0]):
        if data[i] >= data[last_value] if rising else data[i] <= data[last_value]:
            new_data[last_value : i + 1] = np.linspace(
                data[last_value], data[i], i - last_value + 1
            )
            last_value = i
    return new_data


infile = "/home/leo/Downloads/Equilib_Al-Fe _650K_bearbeitet.csv"

df = pd.read_csv(infile, sep=",")

monotone = make_monotone(np.array(df["a-Fe-alpha(A2)"]))
monotone2 = make_monotone(np.array(df["a-Fe-alpha(A2)"])[::-1], rising=False)[::-1]

alpha = 0.5
total = monotone * alpha + monotone2 * (alpha - 1)

plt.plot(monotone, label="Rising", marker=".")
# plt.plot(monotone2, label="Falling")
# plt.plot(total, label="Combined")
plt.plot(np.array(df["a-Fe-alpha(A2)"]), label="Raw")
plt.legend()

plt.show()
