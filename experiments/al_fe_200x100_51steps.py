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

from parse_image_air import parse_image_air

EXPERIMENT_NAME = "al_fe_200x100_51steps"

# call the function
def run_simulation():
    directory = f"result/{EXPERIMENT_NAME}"
    os.makedirs(directory, exist_ok=True)
    lookup_path = "lookup/Activities_Al-Fe_complete.csv"
    lookup_al_temp = pd.read_csv(lookup_path, sep=",", decimal=".")["a-Al(650K)"].to_numpy()
    lookup_fe = pd.read_csv(lookup_path, sep=",", decimal=".")["a-Fe(650K)"].to_numpy()
    lookup_al = lookup_al_temp
    
    # create an instance of the SimulationParams structure (T=650K)
    # TODO: update density values for 650K
    sp = SimulationParams()
    sp.timespan = 10
    sp.m_A = 0.0270 # al
    sp.m_B = 0.0559 # fe
    sp.p_A = 2625.0 # al
    sp.p_B = 7860.0 # fe
    sp.D_A = 5.3e-15
    sp.D_B = 2.7e-14
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
    cmap = LinearSegmentedColormap.from_list("custom_map", cdict)

    # cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_under("black")
    cmap.set_over("black")

    # image_path = "Cu-Sinterstruktur(200x200µm).tif"
    # image_path = "alubeispiel_cropped.tiff"
    # image_path = "air_image.tiff"
    image_path = "images/fe_al_200x100.tiff"
    # X = parse_image(image_path)[..., np.newaxis]  # Add a new axis to make it 3D
    X = parse_image_air(image_path)[..., np.newaxis]  # Add a new axis to make it 3D
    
    plt.imshow(X[...,0].swapaxes(0, 1), cmap=cmap,  vmin=0, vmax=1)
    plt.colorbar()
    plt.title(f"Simulation result at t=0")
    plt.savefig(f"{directory}/simulation_begin.png",dpi=600)
    np.save(f"{directory}/simulation_begin.npy", X)

    d_a_mean = 1.6e-14

    it_steps = 6000
    steps_to_save = 51
    total_steps = it_steps*steps_to_save

    # plot results 650
    temperature = 650
    dir650 = f"result/{EXPERIMENT_NAME}/{temperature}"
    os.makedirs(dir650, exist_ok=True)
    t = 0
    with open(f'{dir650}/result_{temperature}.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow([t, *X[...,0][50]])
    for i in range(0,total_steps,it_steps):
        t += 1
        delta_t = ((sp.dd * sp.dd) / d_a_mean)
        sp.timespan = it_steps * delta_t
        result, X = simulation_cuda(sp, X, lookup_al, lookup_fe)
        plt.imshow(X[...,0].swapaxes(0, 1), cmap=cmap, vmin=0, vmax=1)
        plt.title(f"Simulation result at t={t}")
        plt.savefig(f"{dir650}/simulation_result_{t}.png", dpi=600)
        np.save(f"{dir650}/simulation_result_{t}.npy", X)
        with open(f'{dir650}/result_{temperature}.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow([t, *X[...,0][50]])
        
        

    # update lookpus and SimulationParams for 700
    
    X = parse_image_air(image_path)[..., np.newaxis]  # Add a new axis to make it 3D
    lookup_al_temp = pd.read_csv(lookup_path, sep=",", decimal=".")["a-Al(700K)"].to_numpy()
    lookup_fe = pd.read_csv(lookup_path, sep=",", decimal=".")["a-Fe(700K)"].to_numpy()
    lookup_al = lookup_al_temp
    sp.p_A = 2515.0

    # plot results 700
    temperature = 700
    dir700 = f"result/{EXPERIMENT_NAME}/{temperature}"
    os.makedirs(dir700, exist_ok=True)
    t = 0
    with open(f'{dir700}/result_{temperature}.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow([t, *X[...,0][50]])
    for i in range(0,total_steps,it_steps):
        t += 1
        delta_t = ((sp.dd * sp.dd) / d_a_mean)
        sp.timespan = it_steps * delta_t
        result, X = simulation_cuda(sp, X, lookup_al, lookup_fe)
        plt.imshow(X[...,0].swapaxes(0, 1), cmap=cmap, vmin=0, vmax=1)
        plt.title(f"Simulation result at t={t}")
        plt.savefig(f"{dir700}/simulation_result_{t}.png", dpi=600)
        np.save(f"{dir700}/simulation_result_{t}.npy", X)
        with open(f'{dir700}/result_{temperature}.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow([t, *X[...,0][50]])

    # update lookpus and SimulationParams for 750
    
    X = parse_image_air(image_path)[..., np.newaxis]  # Add a new axis to make it 3D
    lookup_al_temp = pd.read_csv(lookup_path, sep=",", decimal=".")["a-Al(750K)"].to_numpy()
    lookup_fe = pd.read_csv(lookup_path, sep=",", decimal=".")["a-Fe(750K)"].to_numpy()
    lookup_al = lookup_al_temp
    sp.p_A = 2605.0

    # plot results 750
    temperature = 750
    dir750 = f"result/{EXPERIMENT_NAME}/{temperature}"
    os.makedirs(dir750, exist_ok=True)
    t = 0
    with open(f'{dir750}/result_{temperature}.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow([t, *X[...,0][50]])
    for i in range(0,total_steps,it_steps):
        t += 1
        delta_t = ((sp.dd * sp.dd) / d_a_mean)
        sp.timespan = it_steps * delta_t
        result, X = simulation_cuda(sp, X, lookup_al, lookup_fe)
        plt.imshow(X[...,0].swapaxes(0, 1), cmap=cmap, vmin=0, vmax=1)
        plt.title(f"Simulation result at t={t}")
        plt.savefig(f"{dir750}/simulation_result_{t}.png", dpi=600)
        np.save(f"{dir750}/simulation_result_{t}.npy", X)
        with open(f'{dir750}/result_{temperature}.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow([t, *X[...,0][50]])
    
    # update lookpus and SimulationParams for 800
    
    X = parse_image_air(image_path)[..., np.newaxis]  # Add a new axis to make it 3D
    lookup_al_temp = pd.read_csv(lookup_path, sep=",", decimal=".")["a-Al(800K)"].to_numpy()
    lookup_fe = pd.read_csv(lookup_path, sep=",", decimal=".")["a-Fe(800K)"].to_numpy()
    lookup_al = lookup_al_temp
    sp.p_A = 2595.0

    # plot results 800
    temperature = 800
    dir800 = f"result/{EXPERIMENT_NAME}/{temperature}"
    os.makedirs(dir800, exist_ok=True)
    t = 0
    with open(f'{dir800}/result_{temperature}.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow([t, *X[...,0][50]])
    for i in range(0,total_steps,it_steps):
        t += 1
        delta_t = ((sp.dd * sp.dd) / d_a_mean)
        sp.timespan = it_steps * delta_t
        result, X = simulation_cuda(sp, X, lookup_al, lookup_fe)
        plt.imshow(X[...,0].swapaxes(0, 1), cmap=cmap, vmin=0, vmax=1)
        plt.title(f"Simulation result at t={t}")
        plt.savefig(f"{dir800}/simulation_result_{t}.png", dpi=600)
        np.save(f"{dir800}/simulation_result_{t}.npy", X)
        with open(f'{dir800}/result_{temperature}.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow([t, *X[...,0][50]])

if __name__ == "__main__":
    result = run_simulation()
    print(f"Simulation result: {result}")