# from simulation_wrapper import SimulationParams, simulation_cuda
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from diffusion_cuda.simulation_wrapper import SimulationParams, simulation_cuda
import numpy as np
import os

from parse_image_air import parse_image_air

EXPERIMENT_NAME = "al_fe_200x100"

# call the function
def run_simulation():
    directory = f"result/{EXPERIMENT_NAME}"
    os.makedirs(directory, exist_ok=True)
    lookup_path = "lookup\Activities_Al-Fe_complete.csv"
    lookup_al_temp = pd.read_csv(lookup_path, sep=";", decimal=".")["a-Al(650K)"].to_numpy()
    lookup_fe = pd.read_csv(lookup_path, sep=";", decimal=".")["a-Cu(650K)"].to_numpy()
    lookup_al = lookup_al_temp[::-1]
    
    # create an instance of the SimulationParams structure (T=650K)
    # TODO: update params
    sp = SimulationParams()
    sp.timespan = 10
    sp.m_A = 0.0270
    sp.m_B = 0.0559
    sp.p_A = 2625.0
    sp.p_B = 7860.0
    sp.D_A = 4e-9
    sp.D_B = 4e-9
    sp.dd = 10e-10

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
    image_path = "images/fe_al_200x100.tiff"
    # X = parse_image(image_path)[..., np.newaxis]  # Add a new axis to make it 3D
    X = parse_image_air(image_path)[..., np.newaxis]  # Add a new axis to make it 3D
    
    # TODO: set timesteps
    timesteps = []
    
    plt.imshow(X[...,0].swapaxes(0, 1), cmap=cmap,  vmin=0, vmax=1)
    plt.colorbar()
    plt.title(f"Simulation result at t=0")
    plt.savefig(f"{directory}/simulation_begin.png",dpi=600)
    np.save(f"{directory}/simulation_begin.npy", X)

    # plot results 650
    dir650 = f"result/{EXPERIMENT_NAME}/650"
    os.makedirs(dir650, exist_ok=True)
    
    t = 0
    for i in timesteps:
        print(X[...,0][5])
        t += 1
        delta_t = ((sp.dd * sp.dd) / sp.D_A)
        sp.timespan = i * delta_t
        result, X = simulation_cuda(sp, X, lookup_al, lookup_fe)
        plt.imshow(X[...,0].swapaxes(0, 1), cmap=cmap, vmin=0, vmax=1)
        plt.title(f"Simulation result at t={t}")
        plt.savefig(f"{dir650}/simulation_result_{t}.png", dpi=600)
        np.save(f"{dir650}/simulation_result_{t}.npy", X)
        
        

    # update lookpus and SimulationParams for 700
    # TODO
    X = parse_image_air(image_path)[..., np.newaxis]  # Add a new axis to make it 3D
    lookup_al_temp = pd.read_csv(lookup_path, sep=";", decimal=".")["a-Al(700)"].to_numpy()
    lookup_cu = pd.read_csv(lookup_path, sep=";", decimal=".")["a-Cu(700)"].to_numpy()
    lookup_al = lookup_al_temp[::-1]
    sp.p_A = 2560.0
    sp.p_B = 8625.0

    # plot results 600
    dir600 = f"result/{EXPERIMENT_NAME}/700"
    os.makedirs(dir600, exist_ok=True)
    t = 0
    for i in timesteps:
        print(X[...,0][5])
        t += 1
        delta_t = (sp.dd * sp.dd) / sp.D_A
        sp.timespan = i * delta_t
        result, X = simulation_cuda(sp, X, lookup_al, lookup_cu)
        plt.imshow(X[...,0].swapaxes(0, 1), cmap=cmap, vmin=0, vmax=1)
        plt.title(f"Simulation result at t={t}")
        plt.savefig(f"{dir600}/simulation_result_{t}.png", dpi=600)
        np.save(f"{dir600}/simulation_result_{t}.npy", X)

    # update lookpus and SimulationParams for 750
    # TODO
    X = parse_image_air(image_path)[..., np.newaxis]  # Add a new axis to make it 3D
    lookup_al_temp = pd.read_csv(lookup_path, sep=";", decimal=".")["a-Al(750)"].to_numpy()
    lookup_cu = pd.read_csv(lookup_path, sep=";", decimal=".")["a-Cu(750)"].to_numpy()
    lookup_al = lookup_al_temp[::-1]
    # sp.p_A = 2360.0
    # sp.p_B = 8575.0

    # plot results 700
    dir700 = f"result/{EXPERIMENT_NAME}/700"
    os.makedirs(dir700, exist_ok=True)
    t = 0
    for i in timesteps:
        print(X[...,0][5])
        t += 1
        delta_t = (sp.dd * sp.dd) / sp.D_A
        sp.timespan = i * delta_t
        result, X = simulation_cuda(sp, X, lookup_al, lookup_cu)
        plt.imshow(X[...,0].swapaxes(0, 1), cmap=cmap, vmin=0, vmax=1)
        plt.title(f"Simulation result at t={t}")
        plt.savefig(f"{dir700}/simulation_result_{t}.png", dpi=600)
        np.save(f"{dir700}/simulation_result_{t}.npy", X)
    
    # update lookpus and SimulationParams for 800
    # TODO
    X = parse_image_air(image_path)[..., np.newaxis]  # Add a new axis to make it 3D
    lookup_al_temp = pd.read_csv(lookup_path, sep=";", decimal=".")["a-Al(800)"].to_numpy()
    lookup_cu = pd.read_csv(lookup_path, sep=";", decimal=".")["a-Cu(800)"].to_numpy()
    lookup_al = lookup_al_temp[::-1]
    # sp.p_A = 2335.0
    # sp.p_B = 8525.0

    # plot results 800
    dir800 = f"result/{EXPERIMENT_NAME}/800"
    os.makedirs(dir800, exist_ok=True)
    t = 0
    for i in timesteps:
        print(X[...,0][5])
        t += 1
        delta_t = (sp.dd * sp.dd) / sp.D_A
        sp.timespan = i * delta_t
        result, X = simulation_cuda(sp, X, lookup_al, lookup_cu)
        plt.imshow(X[...,0].swapaxes(0, 1), cmap=cmap, vmin=0, vmax=1)
        plt.title(f"Simulation result at t={t}")
        plt.savefig(f"{dir800}/simulation_result_{t}.png", dpi=600)
        np.save(f"{dir800}/simulation_result_{t}.npy", X)

if __name__ == "__main__":
    result = run_simulation()
    print(f"Simulation result: {result}")