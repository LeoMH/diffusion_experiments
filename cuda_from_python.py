# from simulation_wrapper import SimulationParams, simulation_cuda
from diffusion_cuda.simulation_wrapper import SimulationParams, simulation_cuda
from parse_image import parse_image
from parse_image_air import parse_image_air
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd

# call the function
def run_simulation():
    # load the lookup tables
    lookup_path_al = "lookup/Übertragungstabelle_Al_Factsage900°C.csv"
    lookup_path_fe = "lookup/Übertragungstabelle_Fe_Factsage900°C.csv"

    lookup_al = pd.read_csv(lookup_path_al, sep=";", decimal=".")["900"].to_numpy()
    lookup_fe = pd.read_csv(lookup_path_fe, sep=";", decimal=".")["900"].to_numpy()

    # create an instance of the SimulationParams structure
    sp = SimulationParams()
    sp.timespan = 10
    sp.m_A = 0.0270
    sp.m_B = 0.0559
    sp.p_A = 2700.0
    sp.p_B = 7900.0
    sp.D_A = 4e-9
    sp.D_B = 4e-9
    sp.dd = 2e-7

    # define your key colors (hex or RGB tuples):
    hellgrau = "#D3D3D3"    # light gray
    hellblau = "#ADD8E6"    # light blue
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

    # build the colormap
    cmap = LinearSegmentedColormap.from_list("custom_map", cdict)

    # cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_under("black")
    cmap.set_over("black")

    # image_path = "Cu-Sinterstruktur(200x200µm).tif"
    image_path = "./images/air_edited_scaled.tiff"
    # image_path = "air_image.tiff"
    # X = parse_image(image_path)[..., np.newaxis]  # Add a new axis to make it 3D
    X = parse_image_air(image_path)[..., np.newaxis]  # Add a new axis to make it 3D
    plt.imshow(X[...,0].swapaxes(0, 1), cmap=cmap,  vmin=0, vmax=1)
    plt.colorbar()
    # plt.savefig(f"result/simulation_begin.png",dpi=600)

    # timespan = 1000
    # dT = 1
    # t = 0
    # while t < timespan:
    #     sp.timespan = dT
    #     t += dT  
    #     # run the simulation
    #     result, X = simulation_cuda(sp, X, lookup_al, lookup_fe)

    #     # Plot the result
    #     plt.imshow(X[...,0].swapaxes(0, 1), cmap=cmap, vmin=0, vmax=1)
    #     plt.title(f"Simulation result at t={t}")
    #     plt.savefig(f"result/simulation_result_{t}.png", dpi=600)
    #     np.save(f"result/simulation_result_{t}.npy", X)
    #     dT *= 2
    # return result

    # video
    plt.title(f"Simulation result at t=0")
    plt.savefig(f"vid_result/000.png",dpi=600)
    timespan = 60
    dT = 1
    t = 0
    while t < timespan:
        sp.timespan = dT
        t += dT  
        # run the simulation
        result, X = simulation_cuda(sp, X, lookup_al, lookup_fe)

        # Plot the result
        plt.imshow(X[...,0].swapaxes(0, 1), cmap=cmap, vmin=0, vmax=1)
        plt.title(f"Simulation result at t={t}")
        plt.savefig(f"vid_result/{t:03d}.png", dpi=600)
    return result

if __name__ == "__main__":
    result = run_simulation()
    print(f"Simulation result: {result}")
