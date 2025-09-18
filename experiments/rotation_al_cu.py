# from simulation_wrapper import SimulationParams, simulation_cuda
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from diffusion_cuda.simulation_wrapper import SimulationParams, simulation_cuda
import numpy as np
import os

from diffusion_experiments import parse_image_air

EXPERIMENT_NAME = "rotation_al_cu"

# call the function
def run_simulation():
    directory = f"result/{EXPERIMENT_NAME}"
    os.makedirs(directory, exist_ok=True)
    lookup_path = os.path.join("..", "lookup", "Aktivitäten_Al-Cu_500-800°C.csv")
    lookup_al = pd.read_csv(lookup_path, sep=";", decimal=".")["a-Al(500)"].to_numpy()
    lookup_cu_temp = pd.read_csv(lookup_path, sep=";", decimal=".")["a-Cu(500)"].to_numpy()
    lookup_cu = lookup_cu_temp[::-1]
    # create an instance of the SimulationParams structure (T=500)
    sp = SimulationParams()
    sp.timespan = 10
    sp.m_A = 0.0270
    sp.m_B = 0.0635
    sp.p_A = 2595.0
    sp.p_B = 8675.0
    sp.dd = 1e-6

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
    cmap = LinearSegmentedColormap.from_list("custom_map", cdict2)

    # cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_under("black")
    cmap.set_over("black")

    # image_path = "Cu-Sinterstruktur(200x200µm).tif"
    # image_path = "alubeispiel_cropped.tiff"
    # image_path = "air_image.tiff"
    image_path = "rot_al_cu.tiff"
    # X = parse_image(image_path)[..., np.newaxis]  # Add a new axis to make it 3D
    X = parse_image_air(image_path)[..., np.newaxis]  # Add a new axis to make it 3D
    plt.imshow(X[...,0].swapaxes(0, 1), cmap=cmap,  vmin=0, vmax=1)
    plt.colorbar()
    plt.title(f"Simulation result at t=0")
    plt.savefig(f"{directory}/simulation_begin.png",dpi=600)
    np.save(f"{directory}/simulation_begin.npy", X)

    # plot results 500
    for i in range(1,201):
        x = 180*i
        min = 3*i
        sp.timespan = 180
        dt = sp.timespan / 5e7
        coeff = (sp.dd * sp.dd) / dt
        sp.D_A = coeff
        sp.D_B = coeff
        result, X = simulation_cuda(sp, X, lookup_al, lookup_cu)
        plt.imshow(X[...,0].swapaxes(0, 1), cmap=cmap, vmin=0, vmax=1)
        plt.title(f"Simulation result at t={i}min")
        plt.savefig(f"{directory}/500/simulation_result_{i}.png", dpi=600)
        np.save(f"{directory}/500/simulation_result_{i}.npy", X)

    # update lookpus and SimulationParams for 600
    lookup_al = pd.read_csv(lookup_path, sep=";", decimal=".")["a-Al(600)"].to_numpy()
    lookup_cu_temp = pd.read_csv(lookup_path, sep=";", decimal=".")["a-Cu(600)"].to_numpy()
    lookup_cu = lookup_cu_temp[::-1]
    sp.p_A = 2560.0
    sp.p_B = 8625.0

    # plot results 600
    for i in range(1,201):
        x = 180*i
        min = 3*i
        sp.timespan = 180
        dt = sp.timespan / 5e7
        coeff = (sp.dd * sp.dd) / dt
        sp.D_A = coeff
        sp.D_B = coeff
        result, X = simulation_cuda(sp, X, lookup_al, lookup_cu)
        plt.imshow(X[...,0].swapaxes(0, 1), cmap=cmap, vmin=0, vmax=1)
        plt.title(f"Simulation result at t={i}min")
        plt.savefig(f"{directory}/600/simulation_result_{i}.png", dpi=600)
        np.save(f"{directory}/600/simulation_result_{i}.npy", X)

    # update lookpus and SimulationParams for 700
    lookup_al = pd.read_csv(lookup_path, sep=";", decimal=".")["a-Al(600)"].to_numpy()
    lookup_cu_temp = pd.read_csv(lookup_path, sep=";", decimal=".")["a-Cu(600)"].to_numpy()
    lookup_cu = lookup_cu_temp[::-1]
    sp.p_A = 2360.0
    sp.p_B = 8575.0

    # plot results 700
    for i in range(1,201):
        x = 180*i
        min = 3*i
        sp.timespan = 180
        dt = sp.timespan / 5e7
        coeff = (sp.dd * sp.dd) / dt
        sp.D_A = coeff
        sp.D_B = coeff
        result, X = simulation_cuda(sp, X, lookup_al, lookup_cu)
        plt.imshow(X[...,0].swapaxes(0, 1), cmap=cmap, vmin=0, vmax=1)
        plt.title(f"Simulation result at t={i}min")
        plt.savefig(f"{directory}/700/simulation_result_{i}.png", dpi=600)
        np.save(f"{directory}/700/simulation_result_{i}.npy", X)
    
    # update lookpus and SimulationParams for 800
    lookup_al = pd.read_csv(lookup_path, sep=";", decimal=".")["a-Al(600)"].to_numpy()
    lookup_cu_temp = pd.read_csv(lookup_path, sep=";", decimal=".")["a-Cu(600)"].to_numpy()
    lookup_cu = lookup_cu_temp[::-1]
    sp.p_A = 2335.0
    sp.p_B = 8525.0

    # plot results 800
    for i in range(1,201):
        x = 180*i
        min = 3*i
        sp.timespan = 180
        dt = sp.timespan / 5e7
        coeff = (sp.dd * sp.dd) / dt
        sp.D_A = coeff
        sp.D_B = coeff
        result, X = simulation_cuda(sp, X, lookup_al, lookup_cu)
        plt.imshow(X[...,0].swapaxes(0, 1), cmap=cmap, vmin=0, vmax=1)
        plt.title(f"Simulation result at t={i}min")
        plt.savefig(f"{directory}/800/simulation_result_{i}.png", dpi=600)
        np.save(f"{directory}/800/simulation_result_{i}.npy", X)

if __name__ == "__main__":
    result = run_simulation()
    print(f"Simulation result: {result}")