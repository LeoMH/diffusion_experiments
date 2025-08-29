# from simulation_wrapper import SimulationParams, simulation_cuda
from diffusion_cuda.simulation_wrapper import SimulationParams, simulation_cuda
import numpy as np
import os

EXPERIMENT_NAME = "same_coeff_ideal_mixture"

# call the function
def run_simulation():
    directory = f"result/{EXPERIMENT_NAME}"
    os.makedirs(directory, exist_ok=True)
    lookup = np.linspace(0.0, 1.0, 1001)
    lookup_al = lookup
    lookup_fe = lookup
    # create an instance of the SimulationParams structure
    sp = SimulationParams()
    sp.timespan = 10
    sp.m_A = 0.0270
    sp.m_B = 0.0559
    sp.p_A = 2500.0
    sp.p_B = 7800.0
    sp.D_A = 4e-9
    sp.D_B = 4e-9
    sp.dd = 2e-7

    X = np.zeros((496,1,1))
    X[:50] = 1.0

    # video
    dts900 = [1, 2, 7, 10, 30, 50, 100, 300, 500, 3000]
    t = 0
    for i in dts900:
        t = t + i
        sp.timespan = i
        # run the simulation
        result, X = simulation_cuda(sp, X, lookup_al, lookup_fe)
        np.save(f"{directory}/simulation_result_{t}.npy", X)

if __name__ == "__main__":
    result = run_simulation()
    print(f"Simulation result: {result}")
