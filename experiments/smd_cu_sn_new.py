# from simulation_wrapper import SimulationParams, simulation_cuda
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
matplotlib.use("agg")
import pandas as pd
from diffusion_cuda.simulation_wrapper import SimulationParams, simulation_cuda
import numpy as np
import os

from parse_image_air import parse_image_air

EXPERIMENT_NAME = "smd_cu_sn_new"

# call the function
def run_simulation():
    directory = f"result/{EXPERIMENT_NAME}"
    os.makedirs(directory, exist_ok=True)
    lookup_path = "lookup/Activities_Cu-Sn_226°C-234°C.csv"
    lookup_cu_temp = pd.read_csv(lookup_path, sep=",", decimal=".")["aCu"].to_numpy()
    lookup_cu_temp = convert_lookup(lookup_cu_temp)[:,0]
    lookup_sn = pd.read_csv(lookup_path, sep=",", decimal=".")["aSn"].to_numpy()
    lookup_sn = convert_lookup(lookup_sn)[:,0]
    lookup_cu = lookup_cu_temp[::-1]
    
    # create an instance of the SimulationParams structure (T=500)
    sp = SimulationParams()
    sp.timespan = 10
    sp.m_A = 0.0270
    sp.m_B = 0.0635
    sp.p_A = 2595.0
    sp.p_B = 8675.0
    sp.D_A = 4e-9
    sp.D_B = 4e-9
    sp.dd = 2e-6

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
        (0.20, gruen),
        (0.40, pflaume),
        (0.60, blau),
        (0.80, hellblau),
        (1.00, hellgrau),
    ]

    # build the colormap
    cmap = LinearSegmentedColormap.from_list("custom_map", cdict2)

    # cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_under("black")
    cmap.set_over("black")

    # image_path = "Cu-Sinterstruktur(200x200µm).tif"
    # image_path = "alubeispiel_cropped.tiff"
    # image_path = "air_image.tiff"
    image_path = "images/Geometrie2_SMD_(2µm).tiff"
    # X = parse_image(image_path)[..., np.newaxis]  # Add a new axis to make it 3D
    X = parse_image_air(image_path)[..., np.newaxis]  # Add a new axis to make it 3D
    
    timesteps = [4000, 2420, 3885, 6236, 10008, 16065, 25785, 41387, 66431, 106626, 171144, 274701, 440919, 707713, 1135940, 1823282, 2966525, 4697324, 7539609] # total: 20000000
    
    plt.imshow(X[...,0], cmap=cmap,  vmin=0, vmax=1)
    plt.colorbar()
    plt.title(f"Simulation result at t=0")
    plt.savefig(f"{directory}/simulation_begin.png",dpi=600)
    np.save(f"{directory}/simulation_begin.npy", X)

    # plot results 
    
    t = 0
    for i in timesteps:
        t += 1
        delta_t = ((sp.dd * sp.dd) / sp.D_A)
        sp.timespan = i * delta_t
        result, X = simulation_cuda(sp, X, lookup_sn, lookup_cu)
        plt.imshow(X[...,0], cmap=cmap, vmin=0, vmax=1)
        plt.title(f"Simulation result at t={t}")
        plt.savefig(f"{directory}/simulation_result_{t}.png", dpi=600)
        np.save(f"{directory}/simulation_result_{t}.npy", X)
   
def convert_lookup(arr : np.array):
    arrays = np.split(arr, 1001)
    return np.vstack(arrays)

if __name__ == "__main__":
    result = run_simulation()
    print(f"Simulation result: {result}")