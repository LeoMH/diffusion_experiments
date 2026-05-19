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

EXPERIMENT_NAME = "al_cu_spray_coat"

# call the function
def run_simulation():
    directory = f"result/{EXPERIMENT_NAME}"
    os.makedirs(directory, exist_ok=True)
    lookup_path = "lookup/Al-Cu_500-800°C_V3.csv"
    lookup_al_temp = pd.read_csv(lookup_path, sep=";", decimal=".")["a-Al(600)"].to_numpy()
    lookup_cu = pd.read_csv(lookup_path, sep=";", decimal=".")["a-Cu(600)"].to_numpy()
    lookup_al = lookup_al_temp[::-1]
    
    # create an instance of the SimulationParams structure (T=500)
    sp = SimulationParams()
    sp.timespan = 10
    sp.m_A = 0.0270
    sp.m_B = 0.0635
    sp.p_A = 2560.0
    sp.p_B = 8625.0
    sp.D_A = 4e-9
    sp.D_B = 4e-9
    sp.dd = 1e-8
    # define your key colors (hex or RGB tuples):
    hellgrau = "#D3D3D3"    # light gray
    hellblau = "#ADD8E6"    # light blue
    blau = "#1303fc"
    pflaume  = "#8E4585"    # plum
    gruen    = "#00FF00"    # green
    gelb     = "#FFFF00"    # yellow
    purple = "#76329c"
    gold = "#febf01"
    brightblue = "#b7e6fb"
    
    # list of (value, color) control points
    cdict = [
        (0.00, gold),      # at v=0.0
        (0.40, purple),     # at v=0.4
        (1.00, brightblue),  # at v=1.0
    ]

    # build the colormap
    cmap = LinearSegmentedColormap.from_list("custom_map", cdict)

    # cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_under("black")
    cmap.set_over("black")

    # image_path = "Cu-Sinterstruktur(200x200µm).tif"
    # image_path = "alubeispiel_cropped.tiff"
    # image_path = "air_image.tiff"
    image_path = "images/geometric model_Al-Cu_REM-image.tiff"
    # X = parse_image(image_path)[..., np.newaxis]  # Add a new axis to make it 3D
    X = parse_image_air(image_path)[..., np.newaxis]  # Add a new axis to make it 3D
    
    plt.imshow(X[...,0], cmap=cmap,  vmin=0, vmax=1)
    plt.colorbar()
    plt.title(f"Simulation result at t=0")
    plt.savefig(f"{directory}/simulation_begin.png",dpi=600)
    np.save(f"{directory}/simulation_begin.npy", X)

    # plot results 
    
    t = 0
    for i in range(0,204000,4000):
        t += 1
        delta_t = ((sp.dd * sp.dd) / sp.D_A)
        sp.timespan = i * delta_t
        result, X = simulation_cuda(sp, X, lookup_al, lookup_cu)
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