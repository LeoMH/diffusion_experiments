# from simulation_wrapper import SimulationParams, simulation_cuda
import csv
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
matplotlib.use("agg")
import pandas as pd
from diffusion_cuda.simulation_wrapper import SimulationParams, simulation_cuda
import numpy as np
import os
import math
import ctypes

from parse_image_air import parse_image_air

EXPERIMENT_NAME = "cu_sn_cooldown"

def convert_lookup(arr : np.array):
    arrays = np.split(arr, 1001)
    return np.vstack(arrays)

# call the function
def run_simulation():
    directory = f"result/{EXPERIMENT_NAME}"
    os.makedirs(directory, exist_ok=True)

    # define your key colors (hex or RGB tuples):
    hellgrau = "#D3D3D3"    # light gray
    hellblau = "#ADD8E6"    # light blue
    blau = "#1303fc"
    pflaume  = "#8E4585"    # plum
    gruen    = "#00FF00"    # green
    gelb     = "#FFFF00"    # yellow
    
    # list of (value, color) control points
    cdict = [
        (0.00, gelb),      # at v=0.0
        (0.60, gruen),     # at v=0.6
        (0.70, pflaume),   # at v=0.7
        (0.80, hellblau),  # at v=0.8
        (1.00, hellgrau),  # at v=1.0
    ]

    cdict2 = [
        (0.00, gelb),
        (0.05, gruen),
        (0.10, pflaume),
        (0.15, blau),
        (0.20, hellblau),
        (1.00, hellgrau),
    ]

    # build the colormap
    cmap = LinearSegmentedColormap.from_list("custom_map", cdict)

    # cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_under("black")
    cmap.set_over("black")

    image_path = "images/cu_sn_temp_test.tiff"
    X = parse_image_air(image_path)[..., np.newaxis]  # Add a new axis to make it 3D
    
    plt.imshow(X[...,0], cmap=cmap,  vmin=0, vmax=1)
    plt.colorbar()
    plt.title(f"Simulation result at t = 0 s")
    plt.savefig(f"{directory}/simulation_begin.png",dpi=600)
    np.save(f"{directory}/simulation_begin.npy", X)

    t = 0
    temps = [234.00, 233.00, 232.00, 231.00, 230.00, 229.00, 228.00, 227.00, 226.00]
    temps_int = [234, 233, 232, 231, 230, 229, 228, 227, 226]

    lookup_path = "lookup/Activities_Cu-Sn_226°C-234°C.csv"
    lookup_cu_temp = pd.read_csv(lookup_path, sep=",", decimal=".")["aCu"].to_numpy()
    lookup_cu_temp = convert_lookup(lookup_cu_temp)
    lookup_sn_temp = pd.read_csv(lookup_path, sep=",", decimal=".")["aSn"].to_numpy()
    lookup_sn_temp = convert_lookup(lookup_sn_temp)
    lookup_sn_temp = lookup_sn_temp[::-1]
    

    # create an instance of the SimulationParams structure (T=500)
    sp = SimulationParams()
    sp.m_A = 0.0270
    sp.m_B = 0.0635
    sp.D_A = 4e-9
    sp.D_B = 4e-9
    sp.dd = 5e-10
    num_cells = X.shape[0] # / sp.dd ?
    sp.num_temps = num_cells

    for i in range(0,8):
        t = i * 1250

        # in ausgelagerter Funktion: Temperaturen interpolieren + lookup berechnen
        t1 = temps[i]
        t2 = temps[i+1]
        lookup_cu_t1 = lookup_cu_temp[:,i]
        lookup_cu_t2 = lookup_cu_temp[:,i+1]
        lookup_sn_t1 = lookup_sn_temp[:,i]
        lookup_sn_t2 = lookup_sn_temp[:,i+1]

        new_temps = temp_interpolation(t1, t2, num_cells)
        lookup_cu = create_lookup(t1, t2, lookup_cu_t1, lookup_cu_t2, new_temps)
        lookup_sn = create_lookup(t1, t2, lookup_sn_t1, lookup_sn_t2, new_temps)
        np.savetxt(f"testSn_{i}.txt", lookup_sn)
        np.savetxt(f"testCu_{i}.txt", lookup_cu)
        

        # TODO: Simulations Parameter bestimmen
        sp.p_A = 2595.0
        sp.p_B = 8675.0
        
        # delta_t und sp.timespan bestimmen
        delta_t = ((sp.dd * sp.dd) / sp.D_A)
        sp.timespan = t * delta_t
        # sp.timespan = 0.375
        # print(f"timespan: {sp.timespan}, delta_t: {delta_t}")

        #TODO: sp.temps bestimmen
        idx = np.arange(num_cells, dtype=np.uintp)
        temp_idx = idx[:, None, None]
        temp_idx = np.broadcast_to(temp_idx, X.shape)
        # temp_idx_ctypes = temp_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_ulong))
        # sp.temps = temp_idx

        # run simulation cuda
        result, X = simulation_cuda(sp, X, lookup_cu, lookup_sn, temp_idx)

        # print(f"Min X: {np.min(X)}")

        # Ergebnisse speichern
        plt.imshow(X[...,0], cmap=cmap, vmin=0, vmax=1)
        plt.title(f"Simulation result at t = {t} s\nTemperature = {t1} -- {t2}")
        plt.savefig(f"{directory}/simulation_result_{t}.png", dpi=600)
        np.save(f"{directory}/simulation_result_{t}.npy", X)

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

# t1, t2: temp values of outer bounds, float
# num_cells: number of horizontal cells equals number of temps to be returned, int
def temp_interpolation(t1, t2, num_cells):
    new_temps = np.linspace(t1, t2, num_cells)
    return new_temps

# t1, t2: temp values of outer bounds, float
# a_t1, a_t2: lists of activities for t1 and t2, np_1darray[Any]
# new_temps: list of new temperature values, list[float]
def create_lookup(t1, t2, a_t1, a_t2, new_temps):
    A = (np.log(a_t1/a_t2))/((1/t1)-(1/t2))
    B = np.log(a_t1) - (A/t1)
    result = a_t1
    for i in new_temps:
        if(i == t1 or i == t2):
            continue
        new_column = np.exp((A/i)+B)
        result = np.column_stack((result, new_column))
    return result

if __name__ == "__main__":
    result = run_simulation()
    print(f"Simulation result: {result}")
