# from simulation_wrapper import SimulationParams, simulation_cuda
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt2
from matplotlib.colors import LinearSegmentedColormap
matplotlib.use("agg")
import pandas as pd
from diffusion_cuda.simulation_wrapper import SimulationParams, simulation_cuda
import numpy as np
import os
import csv

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
    image_path = "images/al-cu_spray_coat.tiff"
    # X = parse_image(image_path)[..., np.newaxis]  # Add a new axis to make it 3D
    X = parse_image_air(image_path)[..., np.newaxis]  # Add a new axis to make it 3D
    t = 1

    # --- Plot 1: The Image/Heatmap ---
    fig1, ax1 = plt.subplots()
    im = ax1.imshow(X[..., 0], cmap=cmap, vmin=0, vmax=1)
    fig1.colorbar(im, ax=ax1) # Attach colorbar specifically to this image
    ax1.set_title(f"Simulation result at t={t}")
    fig1.savefig(f"{directory}/simulation_result_{t}.png", dpi=600)
    plt.close(fig1) # Closes only fig1

    # --- Plot 2: The Line Graph ---
    value_data_collection = 330
    x_axis = int(X.shape[1])
    Y = X[..., 0][value_data_collection, 250:700]
    x_axis_points = np.arange(250,700)

    fig2, ax2 = plt.subplots()
    ax2.plot(x_axis_points, Y)
    ax2.set_title(f"Simulation result at t={t}")
    fig2.savefig(f"{directory}/row_simulation_result_{t}.png", dpi=600)
    plt.close(fig2) # Closes only fig2

    # Save your data
    np.save(f"{directory}/simulation_result_{t}.npy", X)
    

    # test
    print(len(Y))

    # plot results 
    span = 400000
    with open(f'{directory}/result_row.csv', 'w') as csvfile: # clear existing file of content
        csvfile.close()
    with open(f'{directory}/result_row.csv', 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow([f"t={t}", *Y])
        csvfile.close()

    for i in range(1,51):
        t += 1
        delta_t = ((sp.dd * sp.dd) / sp.D_A)
        sp.timespan = span * delta_t
        result, X = simulation_cuda(sp, X, lookup_al, lookup_cu)
        # --- Plot 1: The Image/Heatmap ---
        fig1, ax1 = plt.subplots()
        im = ax1.imshow(X[..., 0], cmap=cmap, vmin=0, vmax=1)
        fig1.colorbar(im, ax=ax1) # Attach colorbar specifically to this image
        ax1.set_title(f"Simulation result at t={t}")
        fig1.savefig(f"{directory}/simulation_result_{t}.png", dpi=600)
        plt.close(fig1) # Closes only fig1
        Y = X[...,0][value_data_collection,250:700]
        # --- Plot 2: The Line Graph ---
        fig2, ax2 = plt.subplots()
        ax2.plot(x_axis_points, Y)
        ax2.set_title(f"Simulation result at t={t}")
        fig2.savefig(f"{directory}/row_simulation_result_{t}.png", dpi=600)
        plt.close(fig2) # Closes only fig2
        # Save your data
        np.save(f"{directory}/simulation_result_{t}.npy", X) 
        with open(f'{directory}/result_row.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow([f"t={t}", *Y])
            csvfile.close()
   
def convert_lookup(arr : np.array):
    arrays = np.split(arr, 1001)
    return np.vstack(arrays)

if __name__ == "__main__":
    result = run_simulation()
    print(f"Simulation result: {result}")