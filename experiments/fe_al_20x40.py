# from simulation_wrapper import SimulationParams, simulation_cuda
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from diffusion_cuda.simulation_wrapper import SimulationParams, simulation_cuda
import numpy as np
import os
import csv

from parse_image_air import parse_image_air

EXPERIMENT_NAME = "fe_al_20x40"

# call the function
def run_simulation():
    directory = f"result/{EXPERIMENT_NAME}"
    os.makedirs(directory, exist_ok=True)
    lookup_path = "lookup/Lookup_AL-Fe_650°C_neu.csv"
    lookup_al_temp = pd.read_csv(lookup_path, sep=";", decimal=".")["aAl"].to_numpy()
    lookup_fe = pd.read_csv(lookup_path, sep=";", decimal=".")["aFe"].to_numpy()
    lookup_al = lookup_al_temp[::-1]
    
    # create an instance of the SimulationParams structure (T=500)
    sp = SimulationParams()
    sp.timespan = 10
    sp.m_A = 0.0270 # al
    sp.m_B = 0.0558 # fe
    sp.p_A = 2625.0
    sp.p_B = 7860.0
    sp.D_A = 5.3e-15
    sp.D_B = 2.7e-14
    sp.dd = 5e-10

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

    # image_path = "Cu-Sinterstruktur(200x200µm).tif"
    # image_path = "alubeispiel_cropped.tiff"
    # image_path = "air_image.tiff"
    image_path = "images/fe_al_40x20-1.tiff"
    # X = parse_image(image_path)[..., np.newaxis]  # Add a new axis to make it 3D
    X = parse_image_air(image_path)[..., np.newaxis]  # Add a new axis to make it 3D
    
    timesteps = [0, 4000, 2420, 3885, 6236, 10008, 16065, 25785, 41387, 66431, 106626, 171144, 274701, 440919, 707713, 1135940, 1823282, 2966525, 4697324, 7539609] # total: 20000000
    plt.clf()
    plt.grid(True)
    plt.imshow(X[...,0], cmap=cmap,  vmin=0, vmax=1)
    plt.colorbar()
    plt.title(f"Simulation result at t=0")
    plt.savefig(f"{directory}/simulation_begin.png",dpi=600)
    np.save(f"{directory}/simulation_begin.npy", X)

    # plot results 
    d_a_mean = 1.6e-14
    t = 0
    
    for i in timesteps:
        t += 1
        delta_t = ((sp.dd * sp.dd) / d_a_mean)
        sp.timespan = i * delta_t
        result, X = simulation_cuda(sp, X, lookup_al, lookup_fe)
        plt.imshow(X[...,0], cmap=cmap, vmin=0, vmax=1)
        plt.title(f"Simulation result at t={t}")
        plt.savefig(f"{directory}/simulation_result_{t}.png", dpi=600)
        np.save(f"{directory}/simulation_result_{t}.npy", X)
        with open(f'{directory}/result.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow([t, *X[...,0][8]])
   

if __name__ == "__main__":
    result = run_simulation()
    print(f"Simulation result: {result}")